#pragma once

#include <cuda_runtime.h>

// --------------------------- ACTION CLASSIFIER DEV PARAMS --------------------------- //

// bbox information on the original frame, compute once, use in every preprocess kernel
struct alignas(16) ClsDevParams {

    // top-left corner of bbox in original frame
    int bx, by;
    // dims of bbox in original frame  
    int W, H;

    // dims of bbox after rescaling
    int new_W, new_H;
    // offsets of the top-left corner of the bbox inside the new square
    int ox, oy;
    
    // scales from original bbox to rescaled bbox
    float s2dst_x, s2dst_y;
};
