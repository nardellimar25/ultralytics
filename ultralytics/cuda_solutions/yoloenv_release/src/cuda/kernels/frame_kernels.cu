#include "kernels/frame_kernels.cuh"
#include "cuda_helpers.cuh"
#include "cuda_structs.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>



// ---------------------------------- FRAME KERNELS ---------------------------------- //

// bilinear sample BGR pixel (wrapper)
__device__ __forceinline__ void sampleBGR_bilinear(
    const unsigned char* src,
    int pitch,
    int w,
    int h,
    float x,
    float y,
    float& B,
    float& G,
    float& R
){
    if (x < 0.f || y < 0.f || x > (float)(w - 1) || y > (float)(h - 1)) {
        B = G = R = 0.f;
        return;
    }

    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = (x0 + 1 < w) ? x0 + 1 : w - 1;
    int y1 = (y0 + 1 < h) ? y0 + 1 : h - 1;

    float dx = x - x0;
    float dy = y - y0;

    const unsigned char* p00 = src + y0 * pitch + 3 * x0;
    const unsigned char* p10 = src + y0 * pitch + 3 * x1;
    const unsigned char* p01 = src + y1 * pitch + 3 * x0;
    const unsigned char* p11 = src + y1 * pitch + 3 * x1;

    float B00 = (float)p00[0], G00 = (float)p00[1], R00 = (float)p00[2];
    float B10 = (float)p10[0], G10 = (float)p10[1], R10 = (float)p10[2];
    float B01 = (float)p01[0], G01 = (float)p01[1], R01 = (float)p01[2];
    float B11 = (float)p11[0], G11 = (float)p11[1], R11 = (float)p11[2];

    float B0 = B00 + dx * (B10 - B00);
    float B1 = B01 + dx * (B11 - B01);
    float G0 = G00 + dx * (G10 - G00);
    float G1 = G01 + dx * (G11 - G01);
    float R0 = R00 + dx * (R10 - R00);
    float R1 = R01 + dx * (R11 - R01);

    B = B0 + dy * (B1 - B0);
    G = G0 + dy * (G1 - G0);
    R = R0 + dy * (R1 - R0);
}

// Fisheye → rectilinear, BGR, batched over Z = camera index.
__global__ void fisheye_rectify_bgr_kernel_batched(
    const unsigned char* __restrict__ src_batch_bgr,
    unsigned char*       __restrict__ dst_batch_bgr,
    int                  width,
    int                  height,
    int                  num_cameras,
    float                cx_f,
    float                cy_f,
    float                r_f,
    float                f_fish,
    float                fx,
    float                cx_rect,
    float                cy_rect
){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int cam = blockIdx.z;

    if (x >= width || y >= height || cam >= num_cameras) return;

    const int pitch = width * 3;

    // Offset to this camera slice
    const unsigned char* src = src_batch_bgr + cam * (height * pitch);
    unsigned char*       dst = dst_batch_bgr + cam * (height * pitch);

    // --- Same geometry as your NV12 rectify kernel ----------------------
    // (perspective → equidistant fisheye mapping)

    float xn = (((float)x - cx_rect) / fx);
    float yn = (((float)y - cy_rect) / fx);
    float zn = 1.0f;

    float invn = rsqrtf(xn*xn + yn*yn + zn*zn);
    xn *= invn;
    yn *= invn;
    zn *= invn;

    float theta = acosf(zn);
    float phi   = atan2f(yn, xn);
    float r     = f_fish * theta;

    float sx = cx_f + r * cosf(phi);
    float sy = cy_f + r * sinf(phi);

    float dx = sx - cx_f;
    float dy = sy - cy_f;
    float maxr = r_f + 1.0f;
    bool inside = (dx*dx + dy*dy) <= (maxr * maxr);

    float B = 0.f, G = 0.f, R = 0.f;
    if (inside) {
        sampleBGR_bilinear(src, pitch, width, height, sx, sy, B, G, R);
    }

    int idx = y * pitch + 3 * x;
    dst[idx + 0] = clamp_u8f(B);
    dst[idx + 1] = clamp_u8f(G);
    dst[idx + 2] = clamp_u8f(R);
}
