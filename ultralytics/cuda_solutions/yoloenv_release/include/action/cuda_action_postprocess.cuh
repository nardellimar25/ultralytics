// cuda_action_postprocess.cuh
#pragma once
#include "cuda_detection_struct.h"
#include "cuda_visual_struct.cuh"

// Postprocess for the action classifier
// puts together the original detection plus the classification of the detection
void action_cls_postprocess_gpu_batched(
    const Detection* d_dets,
    const float*     d_probs,
    int              N,
    float            scaleX, 
    float            scaleY,
    ActionVis*       d_out,
    cudaStream_t     stream
);
