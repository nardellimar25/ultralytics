#pragma once
#include <cuda_runtime.h>

#include "cuda_detection_struct.h"

// bbox information on the original frame, compute once, use in every preprocess kernel
struct alignas(16) ClsDevParams {

    // top-left corner of bbox in original frame
    int bx, by;
    // dims of bbox in original frame  
    int W, H;

    // dims of bbox after blac padding, will be a square S = max(W,H)
    int S;
    // offsets of the top-left corner of the bbox inside the new square
    int ox, oy;
    
    // scales from original bbox to padded square
    float s2dst_x, s2dst_y;
};

// Preprocess for the action classifier engine
// Takes detections and from them computes crop, padding, resize on the original frame
void action_cls_preprocess_gpu_staged_batched(

    // inputs
    const unsigned char* d_frame_bgr,   int frameW, int frameH,
    const Detection*     d_boxes,         
    float                scaleX, float scaleY,
    float                pad_ratio,

    // persistent scratch
    unsigned char*       d_scratch_crop,        int maxCropW, int maxCropH,
    unsigned char*       d_scratch_square,      int maxSquare,
    unsigned char*       d_scratch_bgr96,
    float*               d_gray96_out_base,

    // fixed output size based on action classifier requirements
    int                  dstW, int dstH,

    // small device params buffer
    ClsDevParams*        d_params,

    // batching
    int                  n_dets,        
    cudaStream_t         stream
);
