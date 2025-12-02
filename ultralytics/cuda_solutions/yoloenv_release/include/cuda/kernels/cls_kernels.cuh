#pragma once

#include <cuda_runtime.h>

#include "cuda_structs.cuh"

// Compute crop params from YOLO bbox
__global__ void cls_compute_params_kernel_batched(
    const Detection* __restrict__ d_boxes,
    int frameW, int frameH,
    float scaleX, float scaleY,
    float pad_ratio,
    int dstW, int dstH,
    int maxCropW, int maxCropH, int maxSquare,
    ClsDevParams* __restrict__ d_params_array,
    int batchN
);

// Frame BGR -> crops
__global__ void crop_img_kernel_batched(
    const unsigned char* __restrict__ d_frame_bgr,
    int frameW, int frameH,
    const ClsDevParams* __restrict__ d_params_array,
    unsigned char* __restrict__ d_out_crop_base,
    int maxCropW, int maxCropH,
    int batchN
);

// Rect (W×H) -> Rect (new_W×new_H) bilinear
__global__ void resize_bilinear_rect_to_rect_batched(
    unsigned char* __restrict__ output_base,
    const unsigned char* __restrict__ input_base,
    const ClsDevParams* __restrict__ params_array,
    int maxCropW,
    int maxCropH,
    int maxOutW,
    int maxOutH,
    int batchN
);

// Center-pad fitted rect into dst×dst square
__global__ void pad_center_to_square_kernel_batched(
    const unsigned char* __restrict__ src_base,
    const ClsDevParams*  __restrict__ params_array,
    unsigned char*       __restrict__ dst_base,
    int maxOutW, int maxOutH,
    int dst,
    int batchN
);

// BGR -> grayscale [0,1]
__global__ void bgr_to_gray_norm_kernel_batched(
    float* __restrict__ out_gray_base,
    const unsigned char* __restrict__ in_bgr_base,
    int width, int height,
    int batchN
);

// BGR -> grayscale [0,1] with strides
__global__ void bgr_to_gray_norm_kernel_batched_strided(
    float* __restrict__ out_gray_base,
    const unsigned char* __restrict__ in_bgr_base,
    int outW, int outH,
    int maxW, int maxH,
    int batchN
);

// Detection + probs -> ActionVis structs
__global__ void final_visual_struct_kernel(
    const Detection* __restrict__ d_dets,
    const float*     __restrict__ d_probs,
    int N,
    float scaleX, float scaleY,
    ActionVis*       __restrict__ d_out
);
