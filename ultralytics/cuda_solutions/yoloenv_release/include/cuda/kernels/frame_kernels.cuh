#pragma once

#include <cuda_runtime.h>
#include "cuda_structs.cuh"

// Rectilinear undistortion for BGR (batched)
__global__ void undistort_bgr_kernel_batched(
    const unsigned char* __restrict__ d_src_batch_bgr,
    unsigned char*       __restrict__ d_dst_batch_bgr,
    int                  width,
    int                  height,
    int                  num_cameras,
    float                fx, float fy,
    float                cx, float cy,
    float                k1, float k2,
    float                p1, float p2,
    float                k3
);

// Fisheye → rectilinear BGR (batched)
__global__ void fisheye_rectify_bgr_kernel_batched(
    const unsigned char* __restrict__ src_batch_bgr,
    unsigned char*       __restrict__ dst_batch_bgr,
    int                  width,
    int                  height,
    int                  num_cameras,
    float                cx_f,
    float                cy_f,
    float                r_f,
    float                f_fish,
    float                fx,
    float                cx_rect,
    float                cy_rect
);
