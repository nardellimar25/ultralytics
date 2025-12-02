// cuda_nvmm_egl_upload.cu
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cudaEGL.h>          // CUgraphicsEGLRegisterImage, CUeglFrame, etc.
#include "nvbufsurface.h"     // NvBufSurface

#include "cuda_nvmm_egl_upload.cuh"

// Local helper for CUDA driver errors
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

// Very simple RGBA -> BGR copy (no resize yet, 1:1 copy)
__global__ void rgba_to_bgr_linear_kernel(
    const unsigned char* __restrict__ src,
    int srcPitch,        // in bytes
    int width,
    int height,
    unsigned char* __restrict__ dst   // tightly packed BGR
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const unsigned char* srcRow = src + y * srcPitch + 4 * x;
    unsigned char r = srcRow[0];
    unsigned char g = srcRow[1];
    unsigned char b = srcRow[2];

    int dstIdx = (y * width + x) * 3;
    dst[dstIdx + 0] = b;
    dst[dstIdx + 1] = g;
    dst[dstIdx + 2] = r;
}

// Public entry point used by the capture thread
bool upload_nvmm_rgba_to_d_bgr(
    NvBufSurface* surf,
    int           index,       // plane index (usually 0)
    unsigned char* d_bgr,      // destination on device
    int           width,
    int           height,
    cudaStream_t  stream
)
{
    // Map for device access
    if (NvBufSurfaceMap(surf, index, 0, NVBUF_MAP_READ) != 0)
    {
        std::cerr << "[CUDA-EGL] NvBufSurfaceMap failed\n";
        return false;
    }

    if (NvBufSurfaceSyncForDevice(surf, index, 0) != 0)
    {
        std::cerr << "[CUDA-EGL] NvBufSurfaceSyncForDevice failed\n";
        NvBufSurfaceUnMap(surf, index, 0);
        return false;
    }

    // Map to EGLImage
    if (NvBufSurfaceMapEglImage(surf, index) != 0)
    {
        std::cerr << "[CUDA-EGL] NvBufSurfaceMapEglImage failed\n";
        NvBufSurfaceUnMap(surf, index, 0);
        return false;
    }

    EGLImageKHR eglImage = surf->surfaceList[index].mappedAddr.eglImage;
    if (!eglImage)
    {
        std::cerr << "[CUDA-EGL] eglImage is null\n";
        NvBufSurfaceUnMapEglImage(surf, index);
        NvBufSurfaceUnMap(surf, index, 0);
        return false;
    }

    CUgraphicsResource cuRes = nullptr;
    CUeglFrame eglFrame;

    CUresult r = cuGraphicsEGLRegisterImage(
        &cuRes,
        eglImage,
        CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
    if (r != CUDA_SUCCESS)
    {
        checkCu(r, "cuGraphicsEGLRegisterImage");
        NvBufSurfaceUnMapEglImage(surf, index);
        NvBufSurfaceUnMap(surf, index, 0);
        return false;
    }

    r = cuGraphicsResourceGetMappedEglFrame(
        &eglFrame,
        cuRes,
        0, 0);
    if (r != CUDA_SUCCESS)
    {
        checkCu(r, "cuGraphicsResourceGetMappedEglFrame");
        cuGraphicsUnregisterResource(cuRes);
        NvBufSurfaceUnMapEglImage(surf, index);
        NvBufSurfaceUnMap(surf, index, 0);
        return false;
    }

    if (eglFrame.frameType != CU_EGL_FRAME_TYPE_PITCH)
    {
        std::cerr << "[CUDA-EGL] Unexpected frameType (not PITCH)\n";
        cuGraphicsUnregisterResource(cuRes);
        NvBufSurfaceUnMapEglImage(surf, index);
        NvBufSurfaceUnMap(surf, index, 0);
        return false;
    }

    unsigned char* srcDevPtr = static_cast<unsigned char*>(eglFrame.frame.pPitch[0]);
    int srcPitch             = static_cast<int>(eglFrame.pitch);

    dim3 block(16, 16);
    dim3 grid(
        (width  + block.x - 1) / block.x,
        (height + block.y - 1) / block.y);

    rgba_to_bgr_linear_kernel<<<grid, block, 0, stream>>>(
        srcDevPtr,
        srcPitch,
        width,
        height,
        d_bgr);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "[CUDA-EGL] rgba_to_bgr_linear_kernel error: "
                  << cudaGetErrorString(err) << "\n";
    }

    cuGraphicsUnregisterResource(cuRes);
    NvBufSurfaceUnMapEglImage(surf, index);
    NvBufSurfaceUnMap(surf, index, 0);

    return (err == cudaSuccess);
}
