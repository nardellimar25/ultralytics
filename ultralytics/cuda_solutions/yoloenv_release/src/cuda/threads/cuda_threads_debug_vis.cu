#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "cuda_structs.cuh"
#include "engine_io.hpp"
#include "threads/cuda_threads.cuh"
#include "engine_debug_utils.h"
#include "debug_vis.cuh"
#include <nvtx3/nvToolsExt.h>
#include <mutex>

// NEW
#include "threads/debug_snapshot.cuh"

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
        debugVis_gpu(
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
