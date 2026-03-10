#include "debug_vis.cuh"
#include "threads/cuda_threads.cuh"
#include "kernels/debug_vis_kernels.cuh"
// NEW: FOR STREAMING UDP
#include "udp_streamer_gst.hpp"


#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <vector>
#include <cstdio>
#include <cstring>


// -------------------------------- DEBUG VISUALIZATION -------------------------------- //

// Cached detections for debug visualization
static std::mutex g_det_mutex;
static std::vector<ActionVis> g_last_vis;
static int g_last_N = 0;

// helper to update cached detections used by debug visualization
void debugVis_update_cached_detections(const ActionVis* h_vis, int N) {
    std::lock_guard<std::mutex> lk(g_det_mutex);

    if (N <= 0 || h_vis == nullptr) {
        g_last_N = 0;
        g_last_vis.clear();
        return;
    }

    g_last_N = N;
    g_last_vis.assign(h_vis, h_vis + g_last_N);
}


// =====================================================================================
// GPU version of debug visualization thread
// - Always updates the displayed frame from GPU (d_bgr_undistorted)
// - Draws the LAST cached detections (so boxes remain stable when inference is skipped)
// =====================================================================================    

void debugVis_gpu(
    int num_cameras,
    int cap_width,
    int cap_height,
    cudaStream_t stream_vis,
    cudaEvent_t ev_frame_ready,
    unsigned char* d_bgr_undistorted,
    size_t frame_bytes
) {
    (void)frame_bytes; // not needed anymore in GPU path; kept for symmetry

    // ---- config ----
    const float debug_scale = 0.25f;
    const int out_w_small = static_cast<int>(cap_width * debug_scale);
    const int out_h_small = static_cast<int>(cap_height * debug_scale);

    // Side-by-side small image dimensions
    const int sbs_w_small = out_w_small * num_cameras;
    const int sbs_h_small = out_h_small;

    // ---- persistent GPU + pinned host buffers (allocated once) ----
    static unsigned char* d_sbs_small = nullptr;   // GPU packed SBS small BGR
    static unsigned char* h_sbs_small = nullptr;   // pinned host buffer
    static size_t alloc_bytes = 0;
    static int alloc_w = 0, alloc_h = 0, alloc_cams = 0;

    // ---- persistent device detections buffer ----
    static ActionVis* d_det = nullptr;
    static int det_cap = 0;

    const size_t needed = static_cast<size_t>(sbs_w_small) * sbs_h_small * 3;

    if (!d_sbs_small || alloc_bytes != needed || alloc_w != sbs_w_small ||
        alloc_h != sbs_h_small || alloc_cams != num_cameras)
    {
        if (d_sbs_small) cudaFree(d_sbs_small);
        if (h_sbs_small) cudaFreeHost(h_sbs_small);

        cudaMalloc(&d_sbs_small, needed);
        cudaMallocHost(&h_sbs_small, needed);

        alloc_bytes = needed;
        alloc_w = sbs_w_small;
        alloc_h = sbs_h_small;
        alloc_cams = num_cameras;
    }

    // TODO: move as parameter / use CLS_MAX_BATCH
    const int MAX_DETS = 64; 

    if (!d_det || det_cap != MAX_DETS) {
        if (d_det) cudaFree(d_det);
        cudaMalloc(&d_det, MAX_DETS * sizeof(ActionVis));
        det_cap = MAX_DETS;
    }

    // ---- 0) wait for newest frame on GPU ----
    cudaStreamWaitEvent(stream_vis, ev_frame_ready, 0);

    // ---- 1) GPU: downscale + pack side-by-side ----
    launch_sbs_downscale_bgr_u8(
        d_bgr_undistorted,
        cap_width, cap_height,
        num_cameras,
        d_sbs_small,
        out_w_small, out_h_small,
        stream_vis
    );

#if !DEBUG_ONLY_FRAME
    // ---- 2) GPU: draw boxes using cached detections ----
    int local_N = 0;
    static std::vector<ActionVis> local_det;
    local_det.clear();
    {
        std::lock_guard<std::mutex> lk(g_det_mutex);
        local_N = g_last_N;
        if (local_N > 0) {
            if (local_N > MAX_DETS) local_N = MAX_DETS;
            local_det.assign(g_last_vis.begin(), g_last_vis.begin() + local_N);
        }
    }

    // Upload detections to GPU (tiny copy, async)
    if (local_N > 0) {
        cudaMemcpyAsync(
            d_det,
            local_det.data(),
            local_N * sizeof(ActionVis),
            cudaMemcpyHostToDevice,
            stream_vis
        );
    }

    launch_draw_boxes_sbs_bgr_u8(
        d_sbs_small,
        sbs_w_small, sbs_h_small,
        num_cameras,
        (local_N > 0) ? d_det : nullptr,
        local_N,
        debug_scale,
        cap_width, cap_height,
        stream_vis
    );

#endif

    // ---- 3) D2H copy of final small SBS image ----
    cudaMemcpyAsync(h_sbs_small, d_sbs_small, needed, cudaMemcpyDeviceToHost, stream_vis);

    // NOTE: You can keep this as synchronize for now.
    // Later we can make main thread display via a "ready" event to avoid sync here.
    cudaStreamSynchronize(stream_vis);

    // ---- 4) Publish frame_front (CPU Mat) ----
    {
        std::lock_guard<std::mutex> lock(frame_copy_mutex);

        // allocate frame_front to correct size if needed
        if (frame_front.empty() ||
            frame_front.cols != sbs_w_small ||
            frame_front.rows != sbs_h_small ||
            frame_front.type() != CV_8UC3)
        {
            frame_front.create(sbs_h_small, sbs_w_small, CV_8UC3);
        }

        // copy into OpenCV Mat
        std::memcpy(frame_front.data, h_sbs_small, needed);
    }
}


// NEW: UDP STREAMING version
// =====================================================================================
// GPU version + UDP streaming (no OpenCV imshow needed)
// Streams the final SBS small BGR frame via UDP (H.264 in MPEG-TS)
// =====================================================================================

// NEW: for udp streaming

static int g_last_w = 0, g_last_h = 0;

void debugVis_gpu_udp(
    int num_cameras,
    int cap_width,
    int cap_height,
    cudaStream_t stream_vis,
    cudaEvent_t ev_frame_ready,
    unsigned char* d_bgr_undistorted,
    size_t frame_bytes
) {
    static GstUdpStreamer g_stream;
    static bool g_stream_started = false;

    (void)frame_bytes;

    // ---- config ----
    const float debug_scale = 0.25f;
    const int out_w_small = static_cast<int>(cap_width  * debug_scale);
    const int out_h_small = static_cast<int>(cap_height * debug_scale);

    const int sbs_w_small = out_w_small * num_cameras;
    const int sbs_h_small = out_h_small;

    const size_t needed = static_cast<size_t>(sbs_w_small) * sbs_h_small * 4; // 4 not 3 because BGRx

    // ---- persistent GPU + pinned host buffers (allocated once) ----
    static unsigned char* d_sbs_small = nullptr;
    static unsigned char* h_sbs_small = nullptr;
    static size_t alloc_bytes = 0;
    static int alloc_w = 0, alloc_h = 0, alloc_cams = 0;

    static ActionVis* d_det = nullptr;
    static int det_cap = 0;

    if (!d_sbs_small || alloc_bytes != needed || alloc_w != sbs_w_small ||
        alloc_h != sbs_h_small || alloc_cams != num_cameras)
    {
        if (d_sbs_small) cudaFree(d_sbs_small);
        if (h_sbs_small) cudaFreeHost(h_sbs_small);

        cudaMalloc(&d_sbs_small, needed);
        cudaMallocHost(&h_sbs_small, needed);

        alloc_bytes = needed;
        alloc_w = sbs_w_small;
        alloc_h = sbs_h_small;
        alloc_cams = num_cameras;

        // force restart streamer on size change
        g_stream_started = false;
    }

    const int MAX_DETS = 64;
    if (!d_det || det_cap != MAX_DETS) {
        if (d_det) cudaFree(d_det);
        cudaMalloc(&d_det, MAX_DETS * sizeof(ActionVis));
        det_cap = MAX_DETS;
    }

    // ---- start streamer once (or when resolution changes) ----
    if (!g_stream_started || g_last_w != sbs_w_small || g_last_h != sbs_h_small) {
        // host = your laptop (receiver), port = 5000
        const std::string host = "10.42.0.2";
        const int port = 5000;
        const int fps = 30;
        const int bitrate = 4 * 1000 * 1000; // 4 Mbps to start (we can tune)

        std::printf("[DEBUG_UDP] starting stream %dx%d @ %dfps -> %s:%d bitrate=%d\n",
                    sbs_w_small, sbs_h_small, fps, host.c_str(), port, bitrate);

        if (!g_stream.start(sbs_w_small, sbs_h_small, fps, host, port, bitrate)) {
            std::fprintf(stderr, "[DEBUG_UDP] ERROR: failed to start GstUdpStreamer\n");
            // If start fails, just return (don’t crash main loop)
            return;
        }
        g_stream_started = true;
        g_last_w = sbs_w_small;
        g_last_h = sbs_h_small;
    }

    // ---- 0) wait for newest frame on GPU ----
    cudaStreamWaitEvent(stream_vis, ev_frame_ready, 0);

    // ---- 1) GPU: downscale + pack side-by-side ----
    launch_sbs_downscale_bgrx_u8(
        d_bgr_undistorted,
        cap_width, cap_height,
        num_cameras,
        d_sbs_small,
        out_w_small, out_h_small,
        stream_vis
    );

#if !DEBUG_ONLY_FRAME
    // ---- 2) GPU: draw boxes using cached detections ----
    int local_N = 0;
    static std::vector<ActionVis> local_det;
    local_det.clear();
    {
        std::lock_guard<std::mutex> lk(g_det_mutex);
        local_N = g_last_N;
        if (local_N > 0) {
            if (local_N > MAX_DETS) local_N = MAX_DETS;
            local_det.assign(g_last_vis.begin(), g_last_vis.begin() + local_N);
        }
    }

    if (local_N > 0) {
        cudaMemcpyAsync(
            d_det,
            local_det.data(),
            local_N * sizeof(ActionVis),
            cudaMemcpyHostToDevice,
            stream_vis
        );
    }

    launch_draw_boxes_sbs_bgrx_u8(
        d_sbs_small,
        sbs_w_small, sbs_h_small,
        num_cameras,
        (local_N > 0) ? d_det : nullptr,
        local_N,
        debug_scale,
        cap_width, cap_height,
        stream_vis
    );
#endif

    // ---- 3) D2H copy ----
    cudaMemcpyAsync(h_sbs_small, d_sbs_small, needed, cudaMemcpyDeviceToHost, stream_vis);
    cudaStreamSynchronize(stream_vis);

    // ---- 4) push to UDP ----
    g_stream.push_bgr(reinterpret_cast<const uint8_t*>(h_sbs_small), needed);
}