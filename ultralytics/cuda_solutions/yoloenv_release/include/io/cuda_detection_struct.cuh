#pragma once

#include <cuda_runtime.h>

// -------------------------------- DETECTION STRUCT -------------------------------- //

// Detection structure for YOLO outputs
struct Detection {
    float x1, y1, x2, y2;  // Bounding box corners
    float score;           // Confidence score
    int class_id;          // Class ID
    int cam_index;         // Camera index
};
