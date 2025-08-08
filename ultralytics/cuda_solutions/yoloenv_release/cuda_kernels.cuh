#pragma once

#include <cuda_runtime.h>
#include <opencv2/core.hpp>

#include "cuda_detection_struct.h"

// Device utility functions
__device__ float iou(const Detection& a, const Detection& b);
__device__ float lerp(float a, float b, float t);

// Kernels
__global__ void resize_bilinear_kernel(
    uchar* output, const uchar* input,
    int in_width, int in_height,
    int out_width, int out_height,
    float scale_x, float scale_y
);

__global__ void preprocess_kernel(
    float* d_output,
    const uchar* d_input_bgr,
    int img_width,
    int img_height
);

__global__ void extract_detections_kernel(
    const float* output,
    int num_anchors,
    float conf_thresh,
    Detection* dets,
    int* mask
);

__global__ void compact_detections_kernel(
    const Detection* __restrict__ d_dets,
    const int* __restrict__ d_mask,
    Detection* __restrict__ d_compacted,
    int* d_num_valid,
    int num_anchors
);

__global__ void nms_kernel_final_output(
    const Detection* dets_in,
    int num_in,
    float iou_thresh,
    Detection* dets_out,
    int* final_count
);
