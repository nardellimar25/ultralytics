#ifndef CUDA_POSTPROCESS_CUH
#define CUDA_POSTPROCESS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "cuda_detection_struct.h"

// Runs postprocessing on GPU: filter + NMS
void yolo_postprocess_gpu(
    const float* d_output,      // TensorRT raw output [1 x 84 x num_anchors]
    int num_anchors,            // Number of anchors
    float conf_thresh,          // Confidence threshold
    float iou_thresh,           // IoU threshold for NMS
    Detection* d_final,         // Output: final filtered detections
    cudaStream_t stream,        // CUDA stream
    Detection* d_dets,          // Temp: raw detections
    Detection* d_compacted,     // Temp: compacted detections
    int* d_mask,                // Temp: detection mask
    int* d_count_compact,       // Temp: mask detections number
    int* d_count_final,         // Output: final count on device
    int* h_count_final,         // Output: final count on host
    cudaEvent_t ev_count_final,  // Event: final count is ready on device
    int num_cameras
);
// NEW: FP16 overload
void yolo_postprocess_gpu(
    const __half* d_output,
    int num_anchors,
    float conf_thresh,
    float iou_thresh,
    Detection* d_final,
    cudaStream_t stream,
    Detection* d_dets,
    Detection* d_compacted,
    int* d_mask,
    int* d_count_compact,
    int* d_count_final,
    int* h_count_final,
    cudaEvent_t ev_count_final,
    int num_cameras
);

#endif // CUDA_POSTPROCESS_CUH
