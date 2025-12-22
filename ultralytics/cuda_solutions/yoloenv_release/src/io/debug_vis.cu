#include "debug_vis.cuh"
#include "threads/cuda_threads.cuh"
#include "kernels/debug_vis_kernels.cuh"


#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <vector>
#include <cstdio>


// -------------------------------- DEBUG VISUALIZATION -------------------------------- //

void debugVis_multistream_side_by_side(
    int num_cameras, 
    int cap_width, 
    int cap_height, 
    const int N,
    cudaStream_t stream2, 
    unsigned char* d_bgr_undistorted, 
    size_t frame_bytes,
    ActionVis* d_vis,  // currently unused, but okay to keep
    ActionVis* h_vis
) {
    // 1) Download current frames for ALL cameras from GPU for debug/display
    {
        std::lock_guard<std::mutex> lock2(frame_copy_mutex);

        const int vis_width  = cap_width * num_cameras;
        const int vis_height = cap_height;

        // Prepare per-camera CPU mats
        static std::vector<cv::Mat> cam_frames;
        if (cam_frames.size() != static_cast<size_t>(num_cameras)) {
            cam_frames.assign(num_cameras, cv::Mat());
        }

        for (int cam = 0; cam < num_cameras; ++cam) {
            if (cam_frames[cam].empty() ||
                cam_frames[cam].cols != cap_width ||
                cam_frames[cam].rows != cap_height ||
                cam_frames[cam].type() != CV_8UC3)
            {
                cam_frames[cam].create(cap_height, cap_width, CV_8UC3);
            }

            const unsigned char* src_cam =
                d_bgr_undistorted + static_cast<size_t>(cam) * frame_bytes;

            cudaMemcpyAsync(
                cam_frames[cam].data,
                src_cam,
                frame_bytes,
                cudaMemcpyDeviceToHost,
                stream2
            );
        }

        // Wait for all copies to complete
        cudaStreamSynchronize(stream2);

        // Prepare composite frame_back: [cam0 | cam1 | cam2 | ...]
        if (frame_back.empty() ||
            frame_back.cols != vis_width ||
            frame_back.rows != vis_height ||
            frame_back.type() != CV_8UC3)
        {
            frame_back.create(vis_height, vis_width, CV_8UC3);
        }

        for (int cam = 0; cam < num_cameras; ++cam) {
            cv::Rect roi(cam * cap_width, 0, cap_width, cap_height);
            cam_frames[cam].copyTo(frame_back(roi));
        }

        // 2) Downscale frame_back → frame_front for faster display
        const float debug_scale = 0.25f; // or 0.2 if you want smaller
        cv::resize(
            frame_back,
            frame_front,
            cv::Size(
                static_cast<int>(vis_width  * debug_scale),
                static_cast<int>(vis_height * debug_scale)
            ),
            0, 0, cv::INTER_AREA
        );
    }

    #if !DEBUG_ONLY_FRAME
        // 3) Draw overlays on frame_front (composite)
        static const cv::Scalar COLORS[3] = {
            cv::Scalar(0,255,0),   // green
            cv::Scalar(0,0,255),   // red
            cv::Scalar(0,255,255)  // yellow
        };
        static const char* CLABELS[3] = {"G","R","Y"};

        // frame_back is the full composite; frame_front is scaled version
        float visScaleX = static_cast<float>(frame_front.cols) /
                        static_cast<float>(frame_back.cols);
        float visScaleY = static_cast<float>(frame_front.rows) /
                        static_cast<float>(frame_back.rows);

        auto clampRect = [&](const cv::Rect& r)->cv::Rect {
            return r & cv::Rect(0, 0, frame_front.cols, frame_front.rows);
        };

        {
            std::lock_guard<std::mutex> lock2(frame_copy_mutex);
            for (int i = 0; i < N; ++i) {
                const ActionVis& v = h_vis[i];

                // Skip detections with invalid camera index
                if (v.cam < 0 || v.cam >= num_cameras) {
                    continue;
                }

                // Compute global coordinates in the composite image
                int global_x1 = v.x1 + v.cam * cap_width;
                int global_x2 = v.x2 + v.cam * cap_width;
                int global_y1 = v.y1;
                int global_y2 = v.y2;

                cv::Rect box(
                    cv::Point(
                        static_cast<int>(global_x1 * visScaleX),
                        static_cast<int>(global_y1 * visScaleY)
                    ),
                    cv::Point(
                        static_cast<int>(global_x2 * visScaleX),
                        static_cast<int>(global_y2 * visScaleY)
                    )
                );

                cv::Rect roi = clampRect(box);

                int idx = (v.cls < 0) ? 0 : (v.cls > 2 ? 2 : v.cls);
                const cv::Scalar col = COLORS[idx];
                cv::rectangle(frame_front, roi, col, 2);

                char txt[64];
                std::snprintf(txt, sizeof(txt), "%s (cam %d)",
                            CLABELS[idx], v.cam);
                cv::putText(
                    frame_front,
                    txt,
                    {roi.x, std::max(0, roi.y - 6)},
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv::LINE_AA
                );
            }
        }
    #endif // !DEBUG_ONLY_FRAME
}


// Single-stream version
void debugVis_single_stream(
    int num_cameras, 
    int cap_width, 
    int cap_height, 
    const int N,
    cudaStream_t stream2, 
    unsigned char* d_bgr_undistorted, 
    size_t frame_bytes,
    ActionVis* d_vis, 
    ActionVis* h_vis
) {

            {

                std::lock_guard<std::mutex> lock2(frame_copy_mutex);

                if (frame_back.empty() ||
                    frame_back.cols != cap_width ||
                    frame_back.rows != cap_height ||
                    frame_back.type() != CV_8UC3)
                {
                    frame_back.create(cap_height, cap_width, CV_8UC3);
                }

                cudaMemcpyAsync(
                    frame_back.data,
                    d_bgr_undistorted,
                    frame_bytes,
                    cudaMemcpyDeviceToHost,
                    stream2
                );
                cudaStreamSynchronize(stream2);

                // 2) Downscale frame_back → frame_front for faster display
                cv::resize(frame_back, frame_front,
                           cv::Size(cap_width / 2, cap_height / 2),
                           0, 0, cv::INTER_AREA);

            }

            // Optional: frame-only debug
            #if DEBUG_ONLY_FRAME
                nvtxRangePop(); // DebugVisualization
                // continue;
            #endif

            // 3) Draw overlays on frame_front
            static const cv::Scalar COLORS[3] = 
                {
                    cv::Scalar(0,255,0),   // green
                    cv::Scalar(0,0,255),   // red
                    cv::Scalar(0,255,255)  // yellow
                };
            static const char* CLABELS[3] = {"G","R","Y"};

            float visScaleX = 0.5f;
            float visScaleY = 0.5f;

            auto clampRect = [&](const cv::Rect& r)->cv::Rect { return r & cv::Rect(0, 0, frame_front.cols, frame_front.rows); };

            {
                std::lock_guard<std::mutex> lock2(frame_copy_mutex);

                for (int i = 0; i < N; ++i) {

                    const ActionVis& v = h_vis[i];

                    cv::Rect box(
                        cv::Point(static_cast<int>(v.x1 * visScaleX),
                                  static_cast<int>(v.y1 * visScaleY)),
                        cv::Point(static_cast<int>(v.x2 * visScaleX),
                                  static_cast<int>(v.y2 * visScaleY))
                    );

                    cv::Rect roi = clampRect(box);

                    int idx = (v.cls < 0) ? 0 : (v.cls > 2 ? 2 : v.cls);
                    const cv::Scalar col = COLORS[idx];
                    cv::rectangle(frame_front, box, col, 2);

                    char txt[64];
                    std::snprintf(txt, sizeof(txt), "%s (cam %d)",
                                  CLABELS[idx], v.cam);
                    cv::putText(frame_front, txt,
                                {box.x, std::max(0, box.y - 6)},
                                cv::FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv::LINE_AA);

                }

            }

}



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
// Debug visualization for: 30 fps (frame update) + 15 fps (inference updates)
// - Always updates the displayed frame from GPU (d_bgr_undistorted)
// - Draws the LAST cached detections (so boxes remain stable when inference is skipped)
//
// REQUIRED: call debugVis_update_cached_detections(...) from inference thread ONLY when
//          you actually ran inference and have fresh h_vis + N.
// =====================================================================================

void debugVis_30_stream_15_infer_side_by_side(
    int num_cameras,
    int cap_width,
    int cap_height,
    cudaStream_t stream_vis,          // a stream used ONLY for D2H copies for display (can be stream2)
    cudaEvent_t ev_frame_ready,       // recorded by capture thread after undistort
    unsigned char* d_bgr_undistorted, // batched [cam0][cam1]... BGR
    size_t frame_bytes                // cap_width * cap_height * 3
) {

    // ---- 1) Wait for newest frame, download frames for ALL cameras (NO frame_copy_mutex) ----
    cudaStreamWaitEvent(stream_vis, ev_frame_ready, 0);

    const int vis_width  = cap_width * num_cameras;
    const int vis_height = cap_height;

    static std::vector<cv::Mat> cam_frames;
    if (cam_frames.size() != static_cast<size_t>(num_cameras)) {
        cam_frames.assign(num_cameras, cv::Mat());
    }

    for (int cam = 0; cam < num_cameras; ++cam) {
        if (cam_frames[cam].empty() ||
            cam_frames[cam].cols != cap_width ||
            cam_frames[cam].rows != cap_height ||
            cam_frames[cam].type() != CV_8UC3)
        {
            cam_frames[cam].create(cap_height, cap_width, CV_8UC3);
        }

        const unsigned char* src_cam =
            d_bgr_undistorted + static_cast<size_t>(cam) * frame_bytes;

        cudaMemcpyAsync(
            cam_frames[cam].data,
            src_cam,
            frame_bytes,
            cudaMemcpyDeviceToHost,
            stream_vis
        );
    }

    // Ensure all copies are done before CPU uses cam_frames
    cudaStreamSynchronize(stream_vis);

    // ---- 2) Compose into LOCAL mats (still NO frame_copy_mutex) ----
    cv::Mat local_back(vis_height, vis_width, CV_8UC3);

    for (int cam = 0; cam < num_cameras; ++cam) {
        cv::Rect roi(cam * cap_width, 0, cap_width, cap_height);
        cam_frames[cam].copyTo(local_back(roi));
    }

    // Downscale for display (local)
    const float debug_scale = 0.25f;
    cv::Mat local_front;
    cv::resize(
        local_back,
        local_front,
        cv::Size(
            static_cast<int>(vis_width  * debug_scale),
            static_cast<int>(vis_height * debug_scale)
        ),
        0, 0, cv::INTER_AREA
    );

    #if !DEBUG_ONLY_FRAME
        // ---- 3) Draw overlays onto LOCAL front (still NO frame_copy_mutex) ----
        std::vector<ActionVis> local_det;
        int local_N = 0;
        {
            std::lock_guard<std::mutex> lk(g_det_mutex);
            local_det = g_last_vis;
            local_N   = g_last_N;
        }

        static const cv::Scalar COLORS[3] = {
            cv::Scalar(0,255,0),
            cv::Scalar(0,0,255),
            cv::Scalar(0,255,255)
        };
        static const char* CLABELS[3] = {"G","R","Y"};

        float visScaleX = static_cast<float>(local_front.cols) / static_cast<float>(local_back.cols);
        float visScaleY = static_cast<float>(local_front.rows) / static_cast<float>(local_back.rows);

        auto clampRect = [&](const cv::Rect& r)->cv::Rect {
            return r & cv::Rect(0, 0, local_front.cols, local_front.rows);
        };

        for (int i = 0; i < local_N; ++i) {
            const ActionVis& v = local_det[i];
            if (v.cam < 0 || v.cam >= num_cameras) continue;

            int global_x1 = v.x1 + v.cam * cap_width;
            int global_x2 = v.x2 + v.cam * cap_width;
            int global_y1 = v.y1;
            int global_y2 = v.y2;

            cv::Rect box(
                cv::Point(static_cast<int>(global_x1 * visScaleX),
                        static_cast<int>(global_y1 * visScaleY)),
                cv::Point(static_cast<int>(global_x2 * visScaleX),
                        static_cast<int>(global_y2 * visScaleY))
            );

            cv::Rect roi = clampRect(box);

            int idx = (v.cls < 0) ? 0 : (v.cls > 2 ? 2 : v.cls);
            const cv::Scalar col = COLORS[idx];

            cv::rectangle(local_front, roi, col, 2);

            char txt[64];
            std::snprintf(txt, sizeof(txt), "%s (cam %d)", CLABELS[idx], v.cam);
            cv::putText(
                local_front,
                txt,
                {roi.x, std::max(0, roi.y - 6)},
                cv::FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv::LINE_AA
            );
        }
    #endif

    // ---- 4) Publish to shared mats under ONE lock (short critical section) ----
    {
        std::lock_guard<std::mutex> lock(frame_copy_mutex);

        // If you need frame_back globally elsewhere, update it too:
        frame_back = local_back;     // (note: this makes a ref-counted header copy)
        frame_front = local_front;   // (same)

        // If you prefer deep copy to keep ownership predictable:
        // local_back.copyTo(frame_back);
        // local_front.copyTo(frame_front);
    }


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

