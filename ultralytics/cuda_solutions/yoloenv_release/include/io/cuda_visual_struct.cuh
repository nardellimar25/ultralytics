#pragma once
#include <cuda_runtime.h>

// Final per-detection info for UI/blur on the ORIGINAL frame
// 32 bytes, 16B aligned, coalesced-friendly
struct alignas(16) ActionVis {
    int x1, y1, x2, y2; // pixel coords on original/camera frame
    int cls;            // 0=G, 1=R, 2=Y
    int cam;            // camera index for multi-stream
    int _pad;           // keep size multiple of 16
};