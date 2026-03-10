#pragma once

#include <cuda_runtime.h>

#include "cuda_structs.cuh"


// -------------------------------- DEBUG VISUALIZATION -------------------------------- //

// helper to update cached detections used by debug visualization
void debugVis_update_cached_detections(const ActionVis* h_vis, int N);

// gpu version for debug visualization thread
void debugVis_gpu(
    int num_cameras,
    int cap_width,
    int cap_height,
    cudaStream_t stream_vis,
    cudaEvent_t ev_frame_ready,
    unsigned char* d_bgr_undistorted,
    size_t frame_bytes
);

// gpu version that uses udp instead of opencv
void debugVis_gpu_udp(
    int num_cameras,
    int cap_width,
    int cap_height,
    cudaStream_t stream_vis,
    cudaEvent_t ev_frame_ready,
    unsigned char* d_bgr_undistorted,
    size_t frame_bytes
);