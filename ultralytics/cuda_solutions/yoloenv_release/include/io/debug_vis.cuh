#pragma once

#include <cuda_runtime.h>

#include "cuda_structs.cuh"


// -------------------------------- DEBUG VISUALIZATION -------------------------------- //
void debugVis_multistream_side_by_side(
    int num_cameras, 
    int cap_width, 
    int cap_height, 
    const int N,
    cudaStream_t stream2, 
    unsigned char* d_bgr_undistorted, 
    size_t frame_bytes,
    ActionVis* d_vis, 
    ActionVis* h_vis
);

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
);

// helper to update cached detections used by debug visualization
void debugVis_update_cached_detections(const ActionVis* h_vis, int N);

void debugVis_30_stream_15_infer_side_by_side(
    int num_cameras,
    int cap_width,
    int cap_height,
    cudaStream_t stream_vis,          
    cudaEvent_t ev_frame_ready,      
    unsigned char* d_bgr_undistorted, 
    size_t frame_bytes               
);


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

