#include "kernels/debug_vis_kernels.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>

static __device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static __device__ __forceinline__ void write_bgr(
    unsigned char* img,
    int W, int H,
    int x, int y,
    unsigned char b, unsigned char g, unsigned char r
) {
    if ((unsigned)x >= (unsigned)W || (unsigned)y >= (unsigned)H) return;
    const int idx = (y * W + x) * 3;
    img[idx + 0] = b;
    img[idx + 1] = g;
    img[idx + 2] = r;
}

// =====================================================================================
// 1) Downscale + pack side-by-side on GPU (nearest neighbor for speed)
// grid: (x,y,cam)
// =====================================================================================
__global__ void k_sbs_downscale_nn_bgr_u8(
    const unsigned char* __restrict__ in_bgr_batched,
    int cap_w, int cap_h,
    int out_w, int out_h,
    int num_cams,
    unsigned char* __restrict__ out_sbs_bgr
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;  // [0, out_w)
    const int y = blockIdx.y * blockDim.y + threadIdx.y;  // [0, out_h)
    const int cam = blockIdx.z;

    if (cam >= num_cams) return;
    if (x >= out_w || y >= out_h) return;

    // Map small -> full (nearest neighbor)
    // inv_scale = cap / out
    const float inv_scale_x = (float)cap_w / (float)out_w;
    const float inv_scale_y = (float)cap_h / (float)out_h;

    int sx = (int)((x + 0.5f) * inv_scale_x);
    int sy = (int)((y + 0.5f) * inv_scale_y);
    sx = (sx < 0) ? 0 : (sx >= cap_w ? cap_w - 1 : sx);
    sy = (sy < 0) ? 0 : (sy >= cap_h ? cap_h - 1 : sy);

    const int cam_stride = cap_w * cap_h * 3;
    const int src_idx = cam * cam_stride + (sy * cap_w + sx) * 3;

    const unsigned char b = in_bgr_batched[src_idx + 0];
    const unsigned char g = in_bgr_batched[src_idx + 1];
    const unsigned char r = in_bgr_batched[src_idx + 2];

    const int out_W = out_w * num_cams;
    const int ox = cam * out_w + x;
    const int dst_idx = (y * out_W + ox) * 3;

    out_sbs_bgr[dst_idx + 0] = b;
    out_sbs_bgr[dst_idx + 1] = g;
    out_sbs_bgr[dst_idx + 2] = r;
}

void launch_sbs_downscale_bgr_u8(
    const unsigned char* d_bgr_undistorted_batched,
    int cap_width, int cap_height,
    int num_cameras,
    unsigned char* d_sbs_small_bgr_u8,
    int out_w_small, int out_h_small,
    cudaStream_t stream
) {
    if (!d_bgr_undistorted_batched || !d_sbs_small_bgr_u8) return;
    if (num_cameras <= 0 || cap_width <= 0 || cap_height <= 0) return;
    if (out_w_small <= 0 || out_h_small <= 0) return;

    dim3 block(16, 16, 1);
    dim3 grid(
        (out_w_small + block.x - 1) / block.x,
        (out_h_small + block.y - 1) / block.y,
        num_cameras
    );

    k_sbs_downscale_nn_bgr_u8<<<grid, block, 0, stream>>>(
        d_bgr_undistorted_batched,
        cap_width, cap_height,
        out_w_small, out_h_small,
        num_cameras,
        d_sbs_small_bgr_u8
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "k_sbs_downscale_nn_bgr_u8 launch failed: %s\n", cudaGetErrorString(err));
    }
}

// =====================================================================================
// 2) Draw boxes on the *small* SBS image on GPU
// One block per detection. Threads stride along edges.
// =====================================================================================
__global__ void k_draw_boxes_sbs_bgr_u8(
    unsigned char* __restrict__ sbs_bgr,
    int sbs_W, int sbs_H,
    int num_cams,
    const ActionVis* __restrict__ dets,
    int N,
    float debug_scale,
    int cap_w, int cap_h
) {
    const int i = blockIdx.x;
    if (i >= N) return;

    const ActionVis v = dets[i];
    if (v.cam < 0 || v.cam >= num_cams) return;

    // Convert full-res box -> SBS small coords
    int x1 = (int)((v.x1 + v.cam * cap_w) * debug_scale);
    int x2 = (int)((v.x2 + v.cam * cap_w) * debug_scale);
    int y1 = (int)(v.y1 * debug_scale);
    int y2 = (int)(v.y2 * debug_scale);

    // Normalize / clamp
    if (x2 < x1) { int t = x1; x1 = x2; x2 = t; }
    if (y2 < y1) { int t = y1; y1 = y2; y2 = t; }

    x1 = clampi(x1, 0, sbs_W - 1);
    x2 = clampi(x2, 0, sbs_W - 1);
    y1 = clampi(y1, 0, sbs_H - 1);
    y2 = clampi(y2, 0, sbs_H - 1);

    if (x2 <= x1 || y2 <= y1) return;

    // Color by class (BGR)
    int cls = v.cls;
    if (cls < 0) cls = 0;
    if (cls > 2) cls = 2;

    unsigned char b = 0, g = 255, r = 0; // green default
    if (cls == 1) { b = 0; g = 0;   r = 255; }      // red
    if (cls == 2) { b = 0; g = 255; r = 255; }      // yellow

    // Thickness (1 px). You can increase to 2 if you want.
    const int t = 1;

    // Draw top and bottom edges
    for (int x = x1 + threadIdx.x; x <= x2; x += blockDim.x) {
        for (int dy = 0; dy < t; ++dy) {
            write_bgr(sbs_bgr, sbs_W, sbs_H, x, y1 + dy, b, g, r);
            write_bgr(sbs_bgr, sbs_W, sbs_H, x, y2 - dy, b, g, r);
        }
    }

    // Draw left and right edges
    for (int y = y1 + threadIdx.x; y <= y2; y += blockDim.x) {
        for (int dx = 0; dx < t; ++dx) {
            write_bgr(sbs_bgr, sbs_W, sbs_H, x1 + dx, y, b, g, r);
            write_bgr(sbs_bgr, sbs_W, sbs_H, x2 - dx, y, b, g, r);
        }
    }
}

void launch_draw_boxes_sbs_bgr_u8(
    unsigned char* d_sbs_small_bgr_u8,
    int out_w_small_sbs, int out_h_small,
    int num_cameras,
    const ActionVis* d_vis, int N,
    float debug_scale,
    int cap_width, int cap_height,
    cudaStream_t stream
) {
    if (!d_sbs_small_bgr_u8) return;
    if (num_cameras <= 0 || out_w_small_sbs <= 0 || out_h_small <= 0) return;
    if (!d_vis || N <= 0) return;

    // One block per det. Threads per block stride along edges.
    const int threads = 256;
    const int blocks = N;

    k_draw_boxes_sbs_bgr_u8<<<blocks, threads, 0, stream>>>(
        d_sbs_small_bgr_u8,
        out_w_small_sbs,
        out_h_small,
        num_cameras,
        d_vis,
        N,
        debug_scale,
        cap_width, cap_height
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "k_draw_boxes_sbs_bgr_u8 launch failed: %s\n", cudaGetErrorString(err));
    }
}
