#pragma once
#include <cuda_runtime.h>
#include "cuda_structs.cuh"  // ActionVis

// 1) Build a side-by-side, downscaled BGR image on GPU:
//    input: batched full-res BGR [num_cameras][H][W][3]
//    output: one packed image [H_small][W_small*num_cameras][3]
void launch_sbs_downscale_bgr_u8(
    const unsigned char* d_bgr_undistorted_batched,
    int cap_width, int cap_height,
    int num_cameras,
    unsigned char* d_sbs_small_bgr_u8,
    int out_w_small, int out_h_small,
    cudaStream_t stream
);

// 2) Draw cached detections on GPU on the *small* side-by-side image.
//    Detections are in CPU cache, we copy them to GPU once per update later.
//    For now, we accept a device pointer.
void launch_draw_boxes_sbs_bgr_u8(
    unsigned char* d_sbs_small_bgr_u8,
    int out_w_small, int out_h_small,
    int num_cameras,
    const ActionVis* d_vis, int N,
    float debug_scale,
    int cap_width, int cap_height,
    cudaStream_t stream
);



// NEW: exactly the same but for bgrx
void launch_sbs_downscale_bgrx_u8(
    const unsigned char* d_bgr_undistorted_batched,
    int cap_width, int cap_height,
    int num_cameras,
    unsigned char* d_sbs_small_bgrx_u8,
    int out_w_small, int out_h_small,
    cudaStream_t stream
);

void launch_draw_boxes_sbs_bgrx_u8(
    unsigned char* d_sbs_small_bgrx_u8,
    int out_w_small_sbs, int out_h_small,
    int num_cameras,
    const ActionVis* d_vis, int N,
    float debug_scale,
    int cap_width, int cap_height,
    cudaStream_t stream
);
