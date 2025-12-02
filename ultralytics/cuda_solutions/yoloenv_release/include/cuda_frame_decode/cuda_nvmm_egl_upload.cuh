#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include "nvbufsurface.h"   // NvBufSurface
#include <stdbool.h>

// Map NvBufSurface -> EGLImage -> CUDA and copy RGBA into tightly-packed BGR
//
// surf   : NvBufSurface* (NVMM buffer from GStreamer)
// index  : usually 0
// d_bgr  : device pointer to [height*width*3] BGR buffer
// width  : frame width
// height : frame height
// stream : CUDA stream for the copy + kernel
//
// returns true on success, false on failure
bool upload_nvmm_rgba_to_d_bgr(
    NvBufSurface* surf,
    int           index,
    unsigned char* d_bgr,
    int           width,
    int           height,
    cudaStream_t  stream
);
