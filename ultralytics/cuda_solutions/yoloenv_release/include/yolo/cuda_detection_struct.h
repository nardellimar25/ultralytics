#pragma once


// -------------------------------- DETECTION STRUCT -------------------------------- //

struct Detection {
    float x1, y1, x2, y2;  // Bounding box corners
    float score;           // Confidence score
    int class_id;          // Class ID
    int cam_index;         // Camera index
};
