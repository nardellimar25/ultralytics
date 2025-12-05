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

void debugVis_multistream_four_windows(
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