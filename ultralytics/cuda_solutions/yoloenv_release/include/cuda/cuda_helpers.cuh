#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cuda.h>

#include "cuda_structs.cuh" 


// -------------------------------- HELPER FUNCTIONS -------------------------------- //

// IOU for NMS
__device__ __forceinline__ float iou(const Detection& a, const Detection& b) {
    const float ax = fmaxf(0.f, a.x2 - a.x1);
    const float ay = fmaxf(0.f, a.y2 - a.y1);
    const float bx = fmaxf(0.f, b.x2 - b.x1);
    const float by = fmaxf(0.f, b.y2 - b.y1);

    const float areaA = ax * ay;
    const float areaB = bx * by;
    if (areaA <= 0.f || areaB <= 0.f) return 0.f;

    const float x1 = fmaxf(a.x1, b.x1);
    const float y1 = fmaxf(a.y1, b.y1);
    const float x2 = fminf(a.x2, b.x2);
    const float y2 = fminf(a.y2, b.y2);

    const float iw = fmaxf(0.f, x2 - x1);
    const float ih = fmaxf(0.f, y2 - y1);
    const float inter = iw * ih;

    // tiny epsilon needed to avoid division by zero when boxes touch
    return inter / (areaA + areaB - inter + 1e-6f);
}

// Linear interpolation
__device__ __forceinline__ float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

// Clamp float value between lo and hi
__device__ __forceinline__ float clampf(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

// Clamp float to [0, 255] and convert to unsigned char
__device__ __forceinline__ unsigned char clamp_u8f(float v) {
    v = v < 0.f ? 0.f : (v > 255.f ? 255.f : v);
    return (unsigned char)(v + 0.5f);
}

// Simple CUDA driver error checker
static void checkCu(CUresult r, const char* msg)
{
    if (r != CUDA_SUCCESS)
    {
        const char* errStr = nullptr;
        cuGetErrorString(r, &errStr);
        std::cerr << "[CUDA-EGL] " << msg << " failed: "
                  << (errStr ? errStr : "unknown") << " (" << r << ")\n";
    }
}

