#pragma once

#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <cuda_fp16.h>

#include "cuda_structs.cuh"
#include "cuda_helpers.cuh"
#include "frame_kernels.cuh"
#include "yolo_kernels.cuh"
#include "cls_kernels.cuh"

/*

// -------------------------------- HELPER FUNCTIONS -------------------------------- //

// Computes Intersection over Union (IoU) between two Detection objects
__device__ float iou(const Detection& a, const Detection& b);
// Performs linear interpolation between a and b by factor t
__device__ float lerp(float a, float b, float t);
// Clamps value v to the range [lo, hi]
__device__ inline float clampf(float v, float lo, float hi);
// Clamps float v to the range [0, 255] and converts to unsigned char
__device__ __forceinline__ unsigned char clamp_u8f(float v);


// ---------------------------------- CUDA KERNELS ---------------------------------- //

// Bilinear resize for BGR uchar images (batched).
// Resizes input images to output dimensions using bilinear interpolation.
__global__ void resize_bilinear_kernel_batched(
    uchar* __restrict__ outputs,        
    const uchar* __restrict__ inputs,   
    int in_width, int in_height,
    int out_width, int out_height,
    float scale_x, float scale_y,
    int num_cameras
);
__global__ void resize_bilinear_kernel_from_params_batched(
    uchar* __restrict__ output,
    const uchar* __restrict__ input,
    int out_width, int out_height,
    const ClsDevParams* __restrict__ p,
    int maxSquare,
    int batchN
);
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

// Preprocess BGR uchar images to normalized RGB float tensor (batched, FP32).
__global__ void preprocess_kernel_batched(
    float*       __restrict__ d_outputs,       
    const uchar* __restrict__ d_inputs_bgr,    
    int img_width, int img_height,
    int num_cameras
);
// Preprocess BGR uchar images to normalized RGB half tensor (batched, FP16).
__global__ void preprocess_kernel_batched_half(
    __half*      __restrict__ d_outputs,       
    const uchar* __restrict__ d_inputs_bgr,   
    int img_width, int img_height,
    int num_cameras
);

// Extract YOLO detections above confidence threshold (batched, FP32).
__global__ void extract_detections_kernel_batched(
    const float* output,    
    int B,
    int num_anchors,
    float conf_thresh,
    Detection* dets,
    int* mask
);
// Extract YOLO detections above confidence threshold (batched, FP16).
__global__ void extract_detections_kernel_batched_half(
    const __half* output,
    int B,
    int num_anchors,
    float conf_thresh,
    Detection* dets,
    int* mask
);

// Compact detections based on mask (batched).
// Keeps only valid detections and outputs their count.
__global__ void compact_detections_kernel_batched(
    const Detection* __restrict__ d_dets,
    const int*       __restrict__ d_mask,
    Detection*       __restrict__ d_compacted,
    int*             __restrict__ d_num_valid,
    int              num_anchors,
    int              B
);

// Apply Non-Maximum Suppression (NMS) on detections (batched).
// Removes overlapping boxes above IoU threshold.
__global__ void nms_kernel_final_output_batched(
    const Detection* __restrict__ dets_in,
    const int*       __restrict__ d_count_compact,
    float iou_thresh,
    Detection*       __restrict__ dets_out,
    int*             __restrict__ d_count_final,
    int num_anchors,
    int max_dets_per_img,
    int B
);

// Compute preprocessing parameters for action classification (batched).
// Generates crop coordinates and scaling info from detection boxes.
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

// Crop detection bounding boxes from original frames (batched).
__global__ void crop_img_kernel_batched(
    const unsigned char* __restrict__ d_frame_bgr,
    int frameW, int frameH,
    const ClsDevParams* __restrict__ d_params_array,
    unsigned char* __restrict__ d_out_crop_base,
    int maxCropW, int maxCropH,
    int batchN
);

// Pad cropped regions to square size with black pixels (batched).
__global__ void pad_to_square_kernel_batched(
    const unsigned char* __restrict__ d_crop_base,
    const ClsDevParams*  __restrict__ d_params_array,
    unsigned char*       __restrict__ d_square_base,
    int maxCropW, int maxCropH,
    int maxSquare,
    int batchN
);
__global__ void pad_center_to_square_kernel_batched(
    const unsigned char* __restrict__ src_base,      // fitted output from resize
    const ClsDevParams*  __restrict__ params_array,  // provides new_W/new_H
    unsigned char*       __restrict__ dst_base,      // final square canvas
    int maxOutW, int maxOutH,                        // src slice strides (pixels)
    int dst,                                         // e.g. 96
    int batchN
);

// Convert BGR (U8) image to grayscale float [0,1] (batched, FP32).
__global__ void bgr_to_gray_norm_kernel_batched(
    float* __restrict__ out_gray_base,
    const unsigned char* __restrict__ in_bgr_base,
    int width, int height,
    int batchN
);
__global__ void bgr_to_gray_norm_kernel_batched_strided(
    float* __restrict__ out_gray_base,           // [N, outH, outW]
    const unsigned char* __restrict__ in_bgr_base,// [N, maxH, maxW, 3], valid WxH
    int outW, int outH,                           // typically 96, 96
    int maxW, int maxH,                           // stride basis for input slice
    int batchN
);

// Build final visualization struct from detections and classification results.
// Produces ActionVis with scaled coordinates and probabilities.
__global__ void final_visual_struct_kernel(
    const Detection* __restrict__ d_dets,
    const float*     __restrict__ d_probs,
    int N,
    float scaleX, float scaleY,
    ActionVis*       __restrict__ d_out
);


// Batched Brown–Conrady undistortion on 8UC3 BGR.
// Layout: d_src_batch_bgr / d_dst_batch_bgr are contiguous [num_cameras, H, W, 3].
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


// Batched equidistant fisheye undistortion on 8UC3 BGR.
// Layout: d_src_batch_bgr / d_dst_batch_bgr are contiguous [num_cameras, H, W, 3].
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

// Bilinear sampling of BGR image at floating point coordinates (device function).
__device__ __forceinline__ void sampleBGR_bilinear(
    const unsigned char* src,
    int pitch,
    int w,
    int h,
    float x,
    float y,
    float& B,
    float& G,
    float& R
);

*/