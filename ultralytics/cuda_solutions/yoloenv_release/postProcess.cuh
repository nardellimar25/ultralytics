#ifndef POSTPROCESS_CUH
#define POSTPROCESS_CUH

#include <vector>

// === Detection structure ===
struct Detection {
    float x1, y1, x2, y2;  // bbox corners
    float score;           // confidence
    int class_id;          // class index (always 0 for you)
};

// === Host-side postprocess launcher ===
// This is the main function you call from your main.cu
int run_postprocess_gpu(
    const float* d_output,      // TensorRT output on device (CHW format)
    int num_anchors,            // Number of anchors (e.g., 1344 or 8400)
    float conf_thresh,          // Confidence threshold (raw logit)
    float iou_thresh,           // IoU threshold for NMS
    int max_dets,               // Max buffer size for detections
    Detection* d_final          // device output buffer
);


// === Optional: you can move these into .cu instead ===
// __global__ void extract_detections_kernel(...);
// __global__ void nms_kernel_final_output(...);
// __device__ float iou(const Detection& a, const Detection& b);

#endif // POSTPROCESS_CUH
