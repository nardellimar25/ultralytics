#pragma once

#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <cuda_fp16.h>

#include "cuda_structs.cuh"


// -------------------------------- PRE PROCESS KERNELS ------------------------------ //

// Generic BGR resize (used for YOLO input)
__global__ void resize_bilinear_kernel_batched(
    uchar*       __restrict__ output_batch,
    const uchar* __restrict__ input_batch,
    int in_width,  int in_height,
    int out_width, int out_height,
    float scale_x, float scale_y,
    int num_cameras
);

// YOLO preprocess: BGR (uint8) -> CHW (float/half), norm [0,1]
__global__ void preprocess_kernel_batched(
    float*        __restrict__ d_output_batch, 
    const uchar*  __restrict__ d_input_batch,  
    int img_width, int img_height,
    int num_cameras
);

__global__ void preprocess_kernel_batched_half(
    __half*       __restrict__ d_output_batch_h, 
    const uchar*  __restrict__ d_input_batch,    
    int img_width, int img_height,
    int num_cameras
);


// Fused resize + preprocess: BGR (uint8) -> CHW (float/half), norm [0,1]
__global__ void resize_preprocess_fused_batched_half(
    __half* __restrict__ out_chw,          // [B,3,H,W]
    const unsigned char* __restrict__ in_bgr, // [B,capH,capW,3]
    int in_w, int in_h,
    int out_w, int out_h,
    float scale_x, float scale_y,
    int B
);


// -------------------------------- POST PROCESS KERNELS ------------------------------ //

// YOLO raw output -> Detection + mask
__global__ void extract_detections_kernel_batched(
    const float* output,
    int B,
    int num_anchors,
    float conf_thresh,
    Detection* dets,
    int* mask
);

__global__ void extract_detections_kernel_batched_half(
    const __half* output,
    int B,
    int num_anchors,
    float conf_thresh,
    Detection* dets,
    int* mask
);

// Compact detections based on mask
__global__ void compact_detections_kernel_batched(
    const Detection* __restrict__ d_dets,
    const int*       __restrict__ d_mask,
    Detection*       __restrict__ d_compacted,
    int*             __restrict__ d_count_compact,
    int              num_anchors,
    int              BA
);

// NMS over compacted detections (per-camera via cam_index)
__global__ void nms_kernel_final_output_batched(
    const Detection* __restrict__ d_compacted,
    const int*       __restrict__ d_count_compact,
    float            iou_thresh,
    Detection*       __restrict__ d_final,
    int*             __restrict__ d_count_final,
    int              num_anchors,
    int              max_dets,
    int              B
);
