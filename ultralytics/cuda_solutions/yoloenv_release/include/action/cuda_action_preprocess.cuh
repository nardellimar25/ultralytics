#pragma once
#include <cuda_runtime.h>

#include "cuda_detection_struct.cuh"
#include "cuda_visual_struct.cuh"
#include "cuda_action_struct.cuh"


// Preprocess for the action classifier engine
// Takes detections and from them computes crop, padding, resize on the original frame
void action_cls_preprocess_gpu_staged_batched(
    const unsigned char* d_frame_bgr,   int frameW, int frameH,
    const Detection*     d_boxes,         
    float                scaleX, float scaleY,
    float                pad_ratio,
    unsigned char*       d_scratch_crop,        int maxCropW, int maxCropH,
    unsigned char*       d_scratch_square,      int maxSquare,
    unsigned char*       d_scratch_bgr96,
    float*               d_gray96_out_base,
    int                  dstW, int dstH,
    ClsDevParams*        d_params,
    int                  n_dets,        
    cudaStream_t         stream
);


// TODO: refactor name
// Preprocess for the action classifier engine trying to mimic the EI example
void action_cls_preprocess_gpu_staged_batched_EI_copycat(
    const unsigned char* d_frame_bgr,   
    int                  frameW, int frameH,
    const Detection*     d_boxes,
    float                scaleX, float scaleY,
    float                pad_ratio,
    unsigned char*       d_scratch_crop,      
    int                  maxCropW, int maxCropH,
    unsigned char*       d_scratch_fitted,    
    int                  maxSquare,
    unsigned char*       d_scratch_bgr96,
    float*               d_gray96_out_base,   
    int                  dstW, int dstH,
    ClsDevParams*        d_params_array,
    int                  n_dets,
    cudaStream_t         stream
);