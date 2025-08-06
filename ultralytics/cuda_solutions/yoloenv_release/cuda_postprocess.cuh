#ifndef CUDA_POSTPROCESS_CUH
#define CUDA_POSTPROCESS_CUH

#include <cuda_runtime.h>
#include "cuda_detection_struct.h"

// Runs postprocessing on GPU: filter + NMS
int run_postprocess_gpu(
    const float* d_output,      // TensorRT raw output [1x84xN]
    int num_anchors,            // Number of anchors (N)
    float conf_thresh,          // Confidence threshold
    float iou_thresh,           // IoU threshold for NMS
    int max_dets,               // Max detections (not strictly needed)
    Detection* d_final,         // Output: final filtered detections
    cudaStream_t stream,        // CUDA stream
    Detection* d_dets,          // Temp: raw detections
    Detection* d_compacted,     // Temp: compacted detections
    int* d_mask,                // Temp: detection mask
    int* d_count_final          // Output: final count on device
);

#endif // CUDA_POSTPROCESS_CUH
