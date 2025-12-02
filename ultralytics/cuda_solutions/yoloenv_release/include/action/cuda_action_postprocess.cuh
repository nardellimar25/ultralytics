// cuda_action_postprocess.cuh
#pragma once

#include <cuda_runtime.h>

#include "cuda_detection_struct.cuh"
#include "cuda_visual_struct.cuh"
#include "cuda_action_struct.cuh"

// Postprocess for the action classifier
// puts together the original detection plus its classification
// To help handling debug visualization if needed
void action_cls_postprocess_gpu_batched(
    const Detection* d_dets,
    const float*     d_probs,
    int              N,
    float            scaleX, 
    float            scaleY,
    ActionVis*       d_out,
    cudaStream_t     stream
);
