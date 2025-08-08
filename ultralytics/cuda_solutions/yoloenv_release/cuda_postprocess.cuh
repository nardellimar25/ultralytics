#ifndef CUDA_POSTPROCESS_CUH
#define CUDA_POSTPROCESS_CUH

#include <cuda_runtime.h>
#include "cuda_detection_struct.h"

// Runs postprocessing on GPU: filter + NMS
void run_postprocess_gpu(
    const float* d_output,      // TensorRT raw output [1 x 84 x num_anchors]
    int num_anchors,            // Number of anchors
    float conf_thresh,          // Confidence threshold
    float iou_thresh,           // IoU threshold for NMS
    Detection* d_final,         // Output: final filtered detections
    cudaStream_t stream,        // CUDA stream
    Detection* d_dets,          // Temp: raw detections
    Detection* d_compacted,     // Temp: compacted detections
    int* d_mask,                // Temp: detection mask
    int* d_count_final,         // Output: final count on device
    int* h_count_final          // Output: final count on host
);

#endif // CUDA_POSTPROCESS_CUH
