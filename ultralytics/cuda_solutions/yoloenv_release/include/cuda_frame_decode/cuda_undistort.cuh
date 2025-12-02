#pragma once

#include <cuda_runtime.h>

// Undistort BGR images in batch on GPU
void undistort_bgr_fisheye_gpu_batched(
    const unsigned char* d_src_batch_bgr,
    unsigned char*       d_dst_batch_bgr,
    int                  width,
    int                  height,
    int                  num_cameras,
    float                fish_fov_deg,
    float                out_hfov_deg,
    float                cx_f,
    float                cy_f,
    float                r_f,
    cudaStream_t         stream
);

// OLD: Batched undistort for [num_cameras, H, W, 3] 8UC3 BGR
void undistort_bgr_gpu_batched(
    const unsigned char* d_src_batch_bgr,
    unsigned char*       d_dst_batch_bgr,
    int                  width,
    int                  height,
    int                  num_cameras,
    float                fx, float fy,
    float                cx, float cy,
    float                k1, float k2,
    float                p1, float p2,
    float                k3,
    cudaStream_t         stream
);