#include <iostream>
#include <cmath>

#include "cuda_undistort.cuh"
#include "kernels/cuda_kernels.cuh"


// Simple divUp helper
template<typename T>
static inline T divUp(T a, T b) { return (a + b - 1) / b; }

// New: fisheye→rectilinear undistort for BGR, batched
// Uses same conceptual parameters as RectifyConfig:
//   fish_fov_deg, out_hfov_deg, cx_f, cy_f, r_f
//
// width/height are the *distorted and rectified* image resolution
// (we keep them the same here: 1920x1080)

void undistort_bgr_fisheye_gpu_batched(
    const unsigned char* d_src_batch_bgr,
    unsigned char*       d_dst_batch_bgr,
    int                  width,
    int                  height,
    int                  num_cameras,
    float                fish_fov_deg,
    float                out_hfov_deg,
    float                cx_f,
    float                cy_f,
    float                r_f,
    cudaStream_t         stream
)
{
    if (!d_src_batch_bgr || !d_dst_batch_bgr ||
        width <= 0 || height <= 0 || num_cameras <= 0) {
        return;
    }

    // --- Derive fisheye + rectified focal lengths from config -----------

    const float fish_fov_rad = fish_fov_deg * (float)M_PI / 180.0f;
    const float out_hfov_rad = out_hfov_deg * (float)M_PI / 180.0f;

    // Equidistant fisheye: r = f_fish * theta
    // At edge of circle: r_f == f_fish * theta_max
    const float theta_max = 0.5f * fish_fov_rad;
    const float f_fish    = r_f / theta_max;           // fisheye focal

    // Pinhole rectified focal length from desired horizontal FOV:
    //   fx = W / (2 * tan(FOV/2))
    const float fx = (float)width / (2.0f * tanf(0.5f * out_hfov_rad));

    // Center of rectified image (assume full frame)
    const float cx_rect = 0.5f * (float)width;
    const float cy_rect = 0.5f * (float)height;

    // --- Launch kernel over [width x height] x num_cameras --------------

    dim3 block(16, 16, 1);
    dim3 grid(
        (unsigned)divUp(width,  (int)block.x),
        (unsigned)divUp(height, (int)block.y),
        (unsigned)num_cameras
    );

    fisheye_rectify_bgr_kernel_batched<<<grid, block, 0, stream>>>(
        d_src_batch_bgr,
        d_dst_batch_bgr,
        width,
        height,
        num_cameras,
        cx_f,
        cy_f,
        r_f,
        f_fish,
        fx,
        cx_rect,
        cy_rect
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "fisheye_rectify_bgr_kernel_batched launch failed: "
                  << cudaGetErrorString(err) << std::endl;
    }
}

/*
// OLD: simple radial+tangential undistort for BGR, batched
void undistort_bgr_gpu_batched(
    const unsigned char* d_src_batch_bgr,
    unsigned char*       d_dst_batch_bgr,
    int                  width,
    int                  height,
    int                  num_cameras,
    float                fx, float fy,
    float                cx, float cy,
    float                k1, float k2,
    float                p1, float p2,
    float                k3,
    cudaStream_t         stream
)
{
    if (!d_src_batch_bgr || !d_dst_batch_bgr || width <= 0 || height <= 0 || num_cameras <= 0) {
        return;
    }

    dim3 blockXY(16, 16, 1);
    dim3 gridXY(
        (width  + blockXY.x - 1) / blockXY.x,
        (height + blockXY.y - 1) / blockXY.y,
        num_cameras
    );

    undistort_bgr_kernel_batched<<<gridXY, blockXY, 0, stream>>>(
        d_src_batch_bgr,
        d_dst_batch_bgr,
        width,
        height,
        num_cameras,
        fx, fy,
        cx, cy,
        k1, k2,
        p1, p2,
        k3
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "undistort_bgr_kernel_batched launch failed: "
                  << cudaGetErrorString(err) << std::endl;
    }
}
*/