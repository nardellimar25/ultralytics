#include "cuda_structs.cuh"
#include "engine_io.hpp"
#include "threads/cuda_threads.cuh"
#include "threads/debug_snapshot.cuh"
#include "engine_debug_utils.h"
#include "debug_vis.cuh"
#include "udp_streamer_gst.hpp"

#include <nvtx3/nvToolsExt.h>
#include <mutex>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// NEW: to identify correct function for debug
using DebugVisFn = void(*)(int,int,int,cudaStream_t,cudaEvent_t,unsigned char*,size_t);


// -------------------------- DEBUG VISUALIZATION THREAD -------------------------- //

void debug_vis_thread(
    int num_cameras, 
    int cap_width, 
    int cap_height,
    cudaStream_t stream_vis,
    unsigned char* d_bgr_undistorted,
    size_t frame_bytes,
    cudaEvent_t ev_frame_ready
) {

    nvtxRangePush("DebugVisThread");

    // Attach CUDA context for this thread
    int dev = 0;
    cudaSetDevice(dev);

    // Decide to use udp or not and print the chosen mode
    DebugVisFn vis_fn = nullptr;

    #if DEBUG_UDP
        vis_fn = &debugVis_gpu_udp;
        std::printf("\n[DEBUG VIS]\nMode: UDP streaming\n");
    #else
        vis_fn = &debugVis_gpu;
        std::printf("\n[DEBUG VIS]\nMode: OpenCV window\n");
    #endif 

    // // init udp streamer once
    // #if DEBUG_UDP

    //     const float debug_scale = 0.25f;
    //     const int out_w_small   = static_cast<int>(cap_width  * debug_scale);
    //     const int out_h_small   = static_cast<int>(cap_height * debug_scale);

    //     const int sbs_w_small = out_w_small * num_cameras;
    //     const int sbs_h_small = out_h_small;

    //     // start once; if you ever change cameras/scale you can re-start
    //     static bool started = false;
    //     if (!started) {
    //         const int fps = 30;
    //         const int bitrate = 6'000'000;
    //         started = g_stream.start(sbs_w_small, sbs_h_small, fps, "10.42.0.2", 5000, bitrate);
    //         if (!started) {
    //             std::printf("\n[DEBUG VIS]\nERROR: failed to start UDP streamer\n");
    //         }
    //     }

    // #endif

    // latest frame id seen
    uint64_t last_seen_frame = 0;

    while (keep_running) {

        uint64_t cur_frame = 0;

        {
            std::unique_lock<std::mutex> lock(frame_mutex);
            frame_ready.wait(lock, [&] {
                return !keep_running.load(std::memory_order_relaxed) ||
                    g_frame_id.load(std::memory_order_acquire) != last_seen_frame;
            });

            if (!keep_running.load(std::memory_order_relaxed)) {
                break;
            }

            cur_frame = g_frame_id.load(std::memory_order_acquire);
        }

        last_seen_frame = cur_frame;

        // Try to consume latest detections snapshot if ready (non-blocking)
        debug_snapshot_try_update_cache();

        // debugVis function to update the frame display
        nvtxRangePush("DebugVisualization");
        vis_fn(
            num_cameras, 
            cap_width, 
            cap_height, 
            stream_vis, 
            ev_frame_ready, 
            d_bgr_undistorted, 
            frame_bytes
        );
        nvtxRangePop();

    } // while keep_running

    nvtxRangePop(); // DebugVisThread

}
