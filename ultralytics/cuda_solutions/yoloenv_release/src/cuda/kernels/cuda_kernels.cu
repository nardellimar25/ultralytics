
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "kernels/cuda_kernels.cuh"
#include "cuda_helpers.cuh"
#include "cuda_structs.cuh"

// TODO: split into multiple files for better organization


// -------------------------------- HELPER FUNCTIONS -------------------------------- //

// __device__ __forceinline__ float iou(const Detection& a, const Detection& b) {
//     const float ax = fmaxf(0.f, a.x2 - a.x1);
//     const float ay = fmaxf(0.f, a.y2 - a.y1);
//     const float bx = fmaxf(0.f, b.x2 - b.x1);
//     const float by = fmaxf(0.f, b.y2 - b.y1);

//     const float areaA = ax * ay;
//     const float areaB = bx * by;
//     if (areaA <= 0.f || areaB <= 0.f) return 0.f;

//     const float x1 = fmaxf(a.x1, b.x1);
//     const float y1 = fmaxf(a.y1, b.y1);
//     const float x2 = fminf(a.x2, b.x2);
//     const float y2 = fminf(a.y2, b.y2);

//     const float iw = fmaxf(0.f, x2 - x1);
//     const float ih = fmaxf(0.f, y2 - y1);
//     const float inter = iw * ih;

//     // tiny epsilon to avoid division by zero when boxes touch
//     return inter / (areaA + areaB - inter + 1e-6f);
// }

// __device__ float lerp(float a, float b, float t) {
//     return a + t * (b - a);
// }

// __device__ inline float clampf(float v, float lo, float hi) {
//     return fminf(fmaxf(v, lo), hi);
// }


// ---------------------------------- FRAME KERNELS ---------------------------------- //

// NEW: to undistort fisheye BGR images (batched)
// ---------------- Fisheye rectification for BGR (batched) -----------------
//
// Model: equidistant fisheye (like your NV12 rectify kernel)
//   - fisheye FOV: fish_fov_deg
//   - rectified horizontal FOV: out_hfov_deg
//   - fisheye circle center: (cx_f, cy_f)
//   - fisheye circle radius: r_f
//
// This kernel expects tightly packed BGR (pitch = width * 3).
/*
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

__device__ __forceinline__ unsigned char clamp_u8f(float v)
{
    v = v < 0.f ? 0.f : (v > 255.f ? 255.f : v);
    return (unsigned char)(v + 0.5f);
}

// Fisheye → rectilinear, BGR, batched over Z = camera index.
// Geometry is the same as your NV12 'rectifyNV12Kernel', but sampling BGR.

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


*/

// ----------------------------------   YOLO KERNELS ---------------------------------- //

/*
// resize multiple contiguous images from in_width/in_height to out_width/out_height
__global__ void resize_bilinear_kernel_batched(
    uchar*       __restrict__ output_batch,
    const uchar* __restrict__ input_batch,
    int in_width,  int in_height,
    int out_width, int out_height,
    float scale_x, float scale_y,
    int num_cameras
){
    const int cam = blockIdx.z;
    if (cam >= num_cameras) return;

    // Per-camera strides (in elements/bytes since uchar)
    const int in_stride  = in_width  * in_height  * 3;
    const int out_stride = out_width * out_height * 3;

    // Camera bases
    const uchar* __restrict__ input  = input_batch  + cam * in_stride;
    uchar*       __restrict__ output = output_batch + cam * out_stride;

    // XY = pixel coords
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_width || y >= out_height) return;

    const float src_x = x * scale_x;
    const float src_y = y * scale_y;

    const int x0 = static_cast<int>(floorf(src_x));
    const int y0 = static_cast<int>(floorf(src_y));
    const int x1 = min(x0 + 1, in_width  - 1);
    const int y1 = min(y0 + 1, in_height - 1);

    const float dx = src_x - x0;
    const float dy = src_y - y0;

    for (int c = 0; c < 3; ++c) {
        const int idx00 = (y0 * in_width + x0) * 3 + c;
        const int idx01 = (y0 * in_width + x1) * 3 + c;
        const int idx10 = (y1 * in_width + x0) * 3 + c;
        const int idx11 = (y1 * in_width + x1) * 3 + c;

        const float top    = lerp(float(input[idx00]), float(input[idx01]), dx);
        const float bottom = lerp(float(input[idx10]), float(input[idx11]), dx);
        const float value  = lerp(top, bottom, dy);

        const int out_idx = (y * out_width + x) * 3 + c;
        output[out_idx] = uchar(fminf(fmaxf(value, 0.0f), 255.0f));
    }
}


// Execute yolo preprocess on multiple contiguous images
// FP32
__global__ void preprocess_kernel_batched(
    float*        __restrict__ d_output_batch, 
    const uchar*  __restrict__ d_input_batch,  
    int img_width, int img_height,
    int num_cameras
){
    const int cam = blockIdx.z;
    if (cam >= num_cameras) return;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= img_width || y >= img_height) return;

    const int plane     = img_width * img_height;
    const int dst_idx   = y * img_width + x;      // index within a single plane
    const int cam_off_f = cam * 3 * plane;        // float output offset for this camera
    const int cam_off_u = cam * plane * 3;        // U8   input  offset for this camera

    const int src_idx = cam_off_u + dst_idx * 3;  // interleaved BGR
    const float inv255 = 1.f / 255.f;

    const float b = d_input_batch[src_idx + 0] * inv255;
    const float g = d_input_batch[src_idx + 1] * inv255;
    const float r = d_input_batch[src_idx + 2] * inv255;

    d_output_batch[cam_off_f + 0 * plane + dst_idx] = r;
    d_output_batch[cam_off_f + 1 * plane + dst_idx] = g;
    d_output_batch[cam_off_f + 2 * plane + dst_idx] = b;
}
// FP16
__global__ void preprocess_kernel_batched_half(
    __half*       __restrict__ d_output_batch_h, 
    const uchar*  __restrict__ d_input_batch,    
    int img_width, int img_height,
    int num_cameras
){
    const int cam = blockIdx.z;
    if (cam >= num_cameras) return;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= img_width || y >= img_height) return;

    const int plane     = img_width * img_height;
    const int dst_idx   = y * img_width + x;      // index within a single plane
    const int cam_off_h = cam * 3 * plane;        // __half output offset for this camera
    const int cam_off_u = cam * plane * 3;        // U8     input  offset for this camera

    const int src_idx = cam_off_u + dst_idx * 3;  // interleaved BGR
    const float inv255 = 1.f / 255.f;

    const float b = d_input_batch[src_idx + 0] * inv255;
    const float g = d_input_batch[src_idx + 1] * inv255;
    const float r = d_input_batch[src_idx + 2] * inv255;

    d_output_batch_h[cam_off_h + 0 * plane + dst_idx] = __float2half_rn(r);
    d_output_batch_h[cam_off_h + 1 * plane + dst_idx] = __float2half_rn(g);
    d_output_batch_h[cam_off_h + 2 * plane + dst_idx] = __float2half_rn(b);
}


// Extract detections from yolo output
// FP32 
__global__ void extract_detections_kernel_batched(
    const float* output,    
    int B,                  
    int num_anchors,        
    float conf_thresh,
    Detection* dets,        
    int* mask
){
    extern __shared__ int shared_mask[];

    const int A  = num_anchors;
    const int BA = B * A;

    int g = blockIdx.x * blockDim.x + threadIdx.x; // global anchor index in [0..B*A)
    if (g >= BA) return;

    int b = g / A;   // batch (camera) index
    int a = g % A;   // anchor index within that image

    // Per-image base pointer (given [B,84,A])
    constexpr int K = 84; // attrs per anchor (4 box + 80 classes)
    const float* out_b = output + size_t(b) * K * A;

    float score = out_b[4 * A + a];
    if (score >= conf_thresh) {
        float cx = out_b[0 * A + a];
        float cy = out_b[1 * A + a];
        float w  = out_b[2 * A + a];
        float h  = out_b[3 * A + a];

        float x1 = cx - 0.5f * w;
        float y1 = cy - 0.5f * h;
        float x2 = cx + 0.5f * w;
        float y2 = cy + 0.5f * h;

        Detection d;
        d.x1 = x1; d.y1 = y1; d.x2 = x2; d.y2 = y2;
        d.score = score;
        d.class_id = 0;          
        d.cam_index = b;        

        dets[g] = d;
        shared_mask[threadIdx.x] = 1;
    } else {
        shared_mask[threadIdx.x] = 0;
    }

    __syncthreads();

    // One thread writes this block's mask slice back
    if (threadIdx.x == 0) {
        int base = blockIdx.x * blockDim.x;
        for (int j = 0; j < blockDim.x; ++j) {
            int idx = base + j;
            if (idx >= BA) break;
            mask[idx] = shared_mask[j];
        }
    }
}
// FP16
__global__ void extract_detections_kernel_batched_half(
    const __half* output,
    int B,
    int num_anchors,  
    float conf_thresh,
    Detection* dets,  
    int* mask        
){
    extern __shared__ int shared_mask[];

    const int A  = num_anchors;
    const int BA = B * A;

    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= BA) return;

    int b = g / A;
    int a = g % A;

    constexpr int K = 84;
    const __half* out_b = output + (size_t)b * K * A;

    auto h2f = __half2float;

    float score = h2f(out_b[4 * A + a]);  // class 0 prob
    Detection d{};
    int keep = 0;

    if (score >= conf_thresh) {
        float cx = h2f(out_b[0 * A + a]);
        float cy = h2f(out_b[1 * A + a]);
        float w  = h2f(out_b[2 * A + a]);
        float h  = h2f(out_b[3 * A + a]);

        d.x1 = cx - 0.5f * w;
        d.y1 = cy - 0.5f * h;
        d.x2 = cx + 0.5f * w;
        d.y2 = cy + 0.5f * h;
        d.score = score;
        d.class_id = 0;
        d.cam_index = b;

        dets[g] = d;
        keep = 1;
    }
    shared_mask[threadIdx.x] = keep;

    __syncthreads();

    if (threadIdx.x == 0) {
        int base = blockIdx.x * blockDim.x;
        for (int j = 0; j < blockDim.x; ++j) {
            int idx = base + j;
            if (idx >= BA) break;
            mask[idx] = shared_mask[j];
        }
    }
}


// Compact yolo detections based on mask
__global__ void compact_detections_kernel_batched(
    const Detection* __restrict__ d_dets,       
    const int*       __restrict__ d_mask,       
    Detection*       __restrict__ d_compacted, 
    int*             __restrict__ d_count_compact,
    int              num_anchors,             
    int              BA                        
){
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= BA) return;
    if (!d_mask[g]) return;

    int out_idx = atomicAdd(d_count_compact, 1);
    d_compacted[out_idx] = d_dets[g];  // cam_index rides along
}


// Final yolo NMS kernel
// NMS over the single compacted list; guards by cam_index
__global__ void nms_kernel_final_output_batched(
    const Detection* __restrict__ d_compacted, 
    const int*       __restrict__ d_count_compact,
    float iou_thresh,
    Detection*       __restrict__ d_final,    
    int*             __restrict__ d_count_final, 
    int num_anchors,                           
    int max_dets,                       
    int B                                      
){
    const int N = *d_count_compact;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const Detection di = d_compacted[i];
    bool keep = true;
    const float eps = 1e-6f;

    for (int j = 0; j < N; ++j) {
        if (j == i) continue;
        const Detection dj = d_compacted[j];
        if (dj.cam_index != di.cam_index) continue; // per-camera NMS inside global list

        const bool higher_or_tie_before =
            (dj.score > di.score + eps) ||
            (fabsf(dj.score - di.score) <= eps && j < i);

        if (higher_or_tie_before && iou(di, dj) > iou_thresh) {
            keep = false;
            break;
        }
    }

    if (keep) {
        int out = atomicAdd(d_count_final, 1);
        if (out < max_dets) {
            d_final[out] = di;
        }
    }
}
*/

// ---------------------------- ACTION CLASSIFIER KERNELS ---------------------------- //

/*
// Compute cropping parameters from detection bbox and original frame
__global__ void cls_compute_params_kernel_batched(
    const Detection* __restrict__ d_boxes,
    int frameW, int frameH,
    float scaleX, float scaleY,
    float pad_ratio,
    int dstW, int dstH,                 
    int maxCropW, int maxCropH, int maxSquare,
    ClsDevParams* __restrict__ d_params_array,
    int batchN
){
    const int b = blockIdx.z;
    if (b >= batchN) return;

    // Map bbox to CAMERA space
    Detection det = d_boxes[b];
    float x1 = det.x1 * scaleX;
    float y1 = det.y1 * scaleY;
    float x2 = det.x2 * scaleX;
    float y2 = det.y2 * scaleY;

    if (x2 < x1) { float t=x1; x1=x2; x2=t; }
    if (y2 < y1) { float t=y1; y1=y2; y2=t; }

    // Make square with padding (for crop window derivation)
    float bw = fmaxf(1.0f, x2 - x1);
    float bh = fmaxf(1.0f, y2 - y1);
    float side = fmaxf(bw, bh);
    float pad  = side * pad_ratio;

    float cx = 0.5f*(x1 + x2);
    float cy = 0.5f*(y1 + y2);
    float half = 0.5f*side + pad;

    // Square ROI (float)
    float rx1 = cx - half;
    float ry1 = cy - half;
    float rx2 = cx + half;
    float ry2 = cy + half;

    // Integer crop window clipped to frame
    int ix0 = static_cast<int>(floorf(rx1));
    int iy0 = static_cast<int>(floorf(ry1));
    int ix1 = static_cast<int>(ceilf (rx2));
    int iy1 = static_cast<int>(ceilf (ry2));

    ix0 = max(0, min(ix0, frameW));
    iy0 = max(0, min(iy0, frameH));
    ix1 = max(0, min(ix1, frameW));
    iy1 = max(0, min(iy1, frameH));

    int W = max(0, ix1 - ix0);
    int H = max(0, iy1 - iy0);

    // Clamp to scratch capacities
    if (W > maxCropW) W = maxCropW;
    if (H > maxCropH) H = maxCropH;

    // Square side S (clamped)
    int S = static_cast<int>(ceilf(fmaxf(rx2 - rx1, ry2 - ry1)));
    if (S < 1) S = 1;
    if (S > maxSquare) S = maxSquare;

    // Offsets of crop top-left within the SxS square
    float fx_off = (float)ix0 - rx1;  // how far inside the square the crop begins (x)
    float fy_off = (float)iy0 - ry1;  // (y)
    int ox = static_cast<int>(floorf(fx_off + 0.5f)); // round to nearest int
    int oy = static_cast<int>(floorf(fy_off + 0.5f));

    // Clamp offsets so W×H fits inside S×S
    ox = max(0, min(ox, S - W));
    oy = max(0, min(oy, S - H));

    // compute fitted output dims (longest side = dst, keep aspect)
    int new_W = 1, new_H = 1;
    if (W <= 0 || H <= 0) {
        new_W = 1; new_H = 1;   // degenerate fallback
    } else if (W >= H) {
        // width dominates -> width becomes dstW
        new_W = dstW;
        // round to nearest; clamp to [1, dstH]
        float h_f = (static_cast<float>(dstW) * static_cast<float>(H)) / static_cast<float>(W);
        new_H = static_cast<int>(floorf(h_f + 0.5f));
        if (new_H < 1)    new_H = 1;
        if (new_H > dstH) new_H = dstH;
    } else {
        // height dominates -> height becomes dstH
        new_H = dstH;
        float w_f = (static_cast<float>(dstH) * static_cast<float>(W)) / static_cast<float>(H);
        new_W = static_cast<int>(floorf(w_f + 0.5f));
        if (new_W < 1)    new_W = 1;
        if (new_W > dstW) new_W = dstW;
    }

    // Scales for rect->rect resize: map output pixel to input coords
    const float s2dst_x = static_cast<float>(W) / static_cast<float>(new_W);
    const float s2dst_y = static_cast<float>(H) / static_cast<float>(new_H);

    // Write params
    ClsDevParams p;
    p.bx = ix0;  p.by = iy0;
    p.W  = W;    p.H  = H;

    p.new_W = new_W;
    p.new_H = new_H;

    p.ox = ox;   p.oy = oy;
    p.s2dst_x = s2dst_x;
    p.s2dst_y = s2dst_y;

    d_params_array[b] = p;

}


// Create multiple crops in multiple frames
__global__ void crop_img_kernel_batched(
    const unsigned char* __restrict__ d_frame_bgr,
    int frameW, int frameH,
    const ClsDevParams* __restrict__ d_params_array,
    unsigned char* __restrict__ d_out_crop_base,
    int maxCropW, int maxCropH,
    int batchN
){
    const int b = blockIdx.z;                   
    if (b >= batchN) return;

    // Per-detection params
    const ClsDevParams p = d_params_array[b];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Only fill the valid W×H region of this detection
    if (x >= p.W || y >= p.H) return;

    // Source coords in the original frame
    const int sx = p.bx + x;
    const int sy = p.by + y;

    // Destination base offset for this detection's slice
    const size_t sliceStride = static_cast<size_t>(maxCropW) * maxCropH * 3; // bytes per detection
    const size_t dst = static_cast<size_t>(b) * sliceStride
                     + (static_cast<size_t>(y) * maxCropW + x) * 3;

    // Bounds guard on source; write black if out of frame
    if ((unsigned)sx >= (unsigned)frameW || (unsigned)sy >= (unsigned)frameH) {
        d_out_crop_base[dst + 0] = 0;
        d_out_crop_base[dst + 1] = 0;
        d_out_crop_base[dst + 2] = 0;
        return;
    }

    // Interleaved BGR copy
    const int src = (sy * frameW + sx) * 3;
    d_out_crop_base[dst + 0] = d_frame_bgr[src + 0];
    d_out_crop_base[dst + 1] = d_frame_bgr[src + 1];
    d_out_crop_base[dst + 2] = d_frame_bgr[src + 2];
}


// Rect (W×H)  ->  Rect (new_W×new_H) using bilinear, batched.
__global__ void resize_bilinear_rect_to_rect_batched(
    unsigned char* __restrict__ output_base,
    const unsigned char* __restrict__ input_base,
    const ClsDevParams* __restrict__ params_array,
    int maxCropW,   
    int maxCropH,   
    int maxOutW,    
    int maxOutH,    
    int batchN
){
    const int b = blockIdx.z;
    if (b >= batchN) return;

    const ClsDevParams p = params_array[b];

    const int inW  = p.W;
    const int inH  = p.H;
    const int outW = p.new_W;
    const int outH = p.new_H;

    // Early outs
    if (inW <= 0 || inH <= 0) return;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= outW || y >= outH) return;

    // Use precomputed scales from params (input->output)
    const float sx = (x + 0.5f) * p.s2dst_x - 0.5f;  // maps output center to input coord
    const float sy = (y + 0.5f) * p.s2dst_y - 0.5f;

    int x0 = static_cast<int>(floorf(sx));
    int y0 = static_cast<int>(floorf(sy));

    // Clamp neighbors
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    int x1 = (x0 + 1 < inW) ? (x0 + 1) : (inW - 1);
    int y1 = (y0 + 1 < inH) ? (y0 + 1) : (inH - 1);

    const float dx = sx - x0;
    const float dy = sy - y0;

    // Slice bases & row strides (bytes)
    const size_t inSliceStride  = static_cast<size_t>(maxCropW) * maxCropH * 3;
    const size_t outSliceStride = static_cast<size_t>(maxOutW)  * maxOutH  * 3;

    const size_t inBase  = static_cast<size_t>(b) * inSliceStride;
    const size_t outBase = static_cast<size_t>(b) * outSliceStride;

    const size_t inRowStride  = static_cast<size_t>(maxCropW) * 3;
    const size_t outRowStride = static_cast<size_t>(maxOutW)  * 3;

    const size_t row0 = inBase  + static_cast<size_t>(y0) * inRowStride;
    const size_t row1 = inBase  + static_cast<size_t>(y1) * inRowStride;
    const size_t o    = outBase + static_cast<size_t>(y)  * outRowStride + static_cast<size_t>(x) * 3;

    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        const float p00 = static_cast<float>(input_base[row0 + static_cast<size_t>(x0) * 3 + c]);
        const float p01 = static_cast<float>(input_base[row0 + static_cast<size_t>(x1) * 3 + c]);
        const float p10 = static_cast<float>(input_base[row1 + static_cast<size_t>(x0) * 3 + c]);
        const float p11 = static_cast<float>(input_base[row1 + static_cast<size_t>(x1) * 3 + c]);

        const float t0 = p00 + (p01 - p00) * dx;
        const float t1 = p10 + (p11 - p10) * dx;
        float v        = t0  + (t1  - t0)  * dy;

        v = v < 0.f ? 0.f : (v > 255.f ? 255.f : v);
        output_base[o + c] = static_cast<unsigned char>(v);
    }
}


// Center-pad a fitted rect (new_W x new_H) into a 96x96 square (or generic dst)
__global__ void pad_center_to_square_kernel_batched(
    const unsigned char* __restrict__ src_base,      
    const ClsDevParams*  __restrict__ params_array,  
    unsigned char*       __restrict__ dst_base,      
    int maxOutW, int maxOutH,                       
    int dst,                                        
    int batchN
){
    const int b = blockIdx.z;
    if (b >= batchN) return;

    const ClsDevParams p = params_array[b];
    int newW = p.new_W;
    int newH = p.new_H;

    // Clamp (defensive)
    if (newW < 0) newW = 0;
    if (newH < 0) newH = 0;
    if (newW > dst) newW = dst;
    if (newH > dst) newH = dst;

    // symmetric pad: left/top = floor, right/bottom = ceil
    const int padL = (dst - newW) >> 1;                 
    const int padT = (dst - newH) >> 1;                

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst || y >= dst) return;

    // per-slice strides (bytes)
    const size_t srcSliceStride = static_cast<size_t>(maxOutW) * maxOutH * 3;
    const size_t dstSliceStride = static_cast<size_t>(dst)     * dst      * 3;

    const size_t srcRowStride = static_cast<size_t>(maxOutW) * 3;
    const size_t dstRowStride = static_cast<size_t>(dst)     * 3;

    const size_t srcBase = static_cast<size_t>(b) * srcSliceStride;
    const size_t dstBase = static_cast<size_t>(b) * dstSliceStride;

    // map dst (x,y) back into the fitted rect
    const int sx = x - padL;
    const int sy = y - padT;

    const size_t d = dstBase + static_cast<size_t>(y) * dstRowStride + static_cast<size_t>(x) * 3;

    if (sx >= 0 && sx < newW && sy >= 0 && sy < newH) {
        const size_t s = srcBase + static_cast<size_t>(sy) * srcRowStride + static_cast<size_t>(sx) * 3;
        dst_base[d + 0] = src_base[s + 0];
        dst_base[d + 1] = src_base[s + 1];
        dst_base[d + 2] = src_base[s + 2];
    } else {
        // black pad
        dst_base[d + 0] = 0;
        dst_base[d + 1] = 0;
        dst_base[d + 2] = 0;
    }
}



// Convert interleaved BGR U8 (96x96) -> grayscale float [0,1]
__global__ void bgr_to_gray_norm_kernel_batched(
    float* __restrict__ out_gray_base,
    const unsigned char* __restrict__ in_bgr_base,
    int width, int height,
    int batchN
){
    const int b = blockIdx.z;
    if (b >= batchN) return;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const size_t inSliceStride  = (size_t)width * height * 3;  
    const size_t outSliceStride = (size_t)width * height;     

    const size_t idx  = (size_t)y * width + x;
    const size_t in3  = (size_t)b * inSliceStride  + idx * 3;
    const size_t out1 = (size_t)b * outSliceStride + idx;

    // BT.601 luma on BGR order: Y = 0.114B + 0.587G + 0.299R
    float bch = (float)in_bgr_base[in3 + 0];
    float gch = (float)in_bgr_base[in3 + 1];
    float rch = (float)in_bgr_base[in3 + 2];

    float y601 = 0.114f * bch + 0.587f * gch + 0.299f * rch;
    out_gray_base[out1] = y601 * (1.0f / 255.0f);
}
// faster strided version
__global__ void bgr_to_gray_norm_kernel_batched_strided(
    float* __restrict__ out_gray_base,          
    const unsigned char* __restrict__ in_bgr_base,
    int outW, int outH,                           
    int maxW, int maxH,                          
    int batchN
){
    const int b = blockIdx.z; if (b >= batchN) return;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= outW || y >= outH) return;

    const size_t inSliceStride  = (size_t)maxW * maxH * 3;
    const size_t outSliceStride = (size_t)outW * outH;

    const size_t inRowStride  = (size_t)maxW * 3;
    const size_t outRowStride = (size_t)outW;

    const size_t in3  = (size_t)b * inSliceStride  + (size_t)y * inRowStride + (size_t)x * 3;
    const size_t out1 = (size_t)b * outSliceStride + (size_t)y * outRowStride + (size_t)x;

    float bch = (float)in_bgr_base[in3 + 0];
    float gch = (float)in_bgr_base[in3 + 1];
    float rch = (float)in_bgr_base[in3 + 2];
    float y601 = 0.114f * bch + 0.587f * gch + 0.299f * rch;
    out_gray_base[out1] = y601 * (1.0f / 255.0f);
}



// Create final visual struct of each detection
__global__ void final_visual_struct_kernel(
    const Detection* __restrict__ d_dets,  
    const float*     __restrict__ d_probs,
    int N,
    float scaleX, float scaleY,
    ActionVis*       __restrict__ d_out
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Argmax over 3 probs (G,R,Y)
    int off = i * 3;
    float g = d_probs[off + 0];
    float r = d_probs[off + 1];
    float y = d_probs[off + 2];

    int cls = 0;          // 0=G
    float best = g;
    if (r > best) { best = r; cls = 1; }  // 1=R
    if (y > best) {           cls = 2; }  // 2=Y

    // Read det, scale to frame space, ensure ordering, cast to int
    const Detection det = d_dets[i];

    int x1 = __float2int_rz(det.x1 * scaleX);
    int y1 = __float2int_rz(det.y1 * scaleY);
    int x2 = __float2int_rz(det.x2 * scaleX);
    int y2 = __float2int_rz(det.y2 * scaleY);

    if (x2 < x1) { int t = x1; x1 = x2; x2 = t; }
    if (y2 < y1) { int t = y1; y1 = y2; y2 = t; }

    ActionVis v;
    v.x1 = x1; v.y1 = y1; v.x2 = x2; v.y2 = y2;
    v.cls = cls;
    v.cam = det.cam_index;  // comes from the Detection
    v._pad = 0;

    d_out[i] = v;
}
*/

/* ------------------------------- DEPRECATED OLD KERNELS ------------------------------- */

/*

// Bilinear resize kernel for BGR uchar images
__global__ void resize_bilinear_kernel(
    uchar* output, const uchar* input,
    int in_width, int in_height,
    int out_width, int out_height,
    float scale_x, float scale_y
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_width || y >= out_height) return;

    float src_x = x * scale_x;
    float src_y = y * scale_y;

    int x0 = int(floorf(src_x));
    int x1 = min(x0 + 1, in_width - 1);
    int y0 = int(floorf(src_y));
    int y1 = min(y0 + 1, in_height - 1);

    float dx = src_x - x0;
    float dy = src_y - y0;

    for (int c = 0; c < 3; ++c) {
        float top = lerp(
            float(input[(y0 * in_width + x0) * 3 + c]),
            float(input[(y0 * in_width + x1) * 3 + c]),
            dx);

        float bottom = lerp(
            float(input[(y1 * in_width + x0) * 3 + c]),
            float(input[(y1 * in_width + x1) * 3 + c]),
            dx);

        float value = lerp(top, bottom, dy);
        output[(y * out_width + x) * 3 + c] = uchar(min(max(value, 0.0f), 255.0f));
    }
}



__global__ void resize_bilinear_kernel_from_params(
    uchar* __restrict__ output,
    const uchar* __restrict__ input,
    int out_width, int out_height,
    const ClsDevParams* __restrict__ p
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_width || y >= out_height) return;

    const int   S  = p->S;
    const float sx = (x + 0.5f) * p->s2dst_x - 0.5f;  // p->s2dst_x = S / outW
    const float sy = (y + 0.5f) * p->s2dst_y - 0.5f;

    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    int x1 = (x0 + 1 < S) ? (x0 + 1) : (S - 1);
    int y1 = (y0 + 1 < S) ? (y0 + 1) : (S - 1);

    float dx = sx - x0;
    float dy = sy - y0;

    int row0 = y0 * S * 3;
    int row1 = y1 * S * 3;
    int o    = (y * out_width + x) * 3;

    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        float t0 = lerp((float)input[row0 + x0*3 + c],
                          (float)input[row0 + x1*3 + c], dx);
        float t1 = lerp((float)input[row1 + x0*3 + c],
                          (float)input[row1 + x1*3 + c], dx);
        float v  = lerp(t0, t1, dy);
        v = v < 0.f ? 0.f : (v > 255.f ? 255.f : v);
        output[o + c] = (unsigned char)v;
    }
}



// Yolo preprocess BGR uchar image to normalized RGB float tensor of given width and height
__global__ void preprocess_kernel(
    float* d_output, const uchar* d_input_bgr,
    int img_width, int img_height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= img_width || y >= img_height) return;

    const int plane   = img_width * img_height;
    const int dst_idx = y * img_width + x;        
    const int src_idx = dst_idx * 3; 

    const float inv255 = 1.f / 255.f;

    // Read BGR, normalize to [0,1], and write CHW as RGB
    float b = (d_input_bgr[src_idx + 0]) * inv255;
    float g = (d_input_bgr[src_idx + 1]) * inv255;
    float r = (d_input_bgr[src_idx + 2]) * inv255;

    d_output[0 * plane + dst_idx] = r;
    d_output[1 * plane + dst_idx] = g;
    d_output[2 * plane + dst_idx] = b;

}
// FP16 variant
__global__ void preprocess_kernel_half(
    __half* d_output, const uchar* d_input_bgr, 
    int img_width, int img_height
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= img_width || y >= img_height) return;

    const int plane   = img_width * img_height;
    const int dst_idx = y * img_width + x;        
    const int src_idx = dst_idx * 3;              

    const float inv255 = 1.f / 255.f;

    // Read BGR, normalize to [0,1], and write CHW as RGB
    float b = d_input_bgr[src_idx + 0] * inv255;
    float g = d_input_bgr[src_idx + 1] * inv255;
    float r = d_input_bgr[src_idx + 2] * inv255;

    d_output[0 * plane + dst_idx] = __float2half_rn(r);
    d_output[1 * plane + dst_idx] = __float2half_rn(g);
    d_output[2 * plane + dst_idx] = __float2half_rn(b);
}



// Extract detections above confidence threshold and create a mask
__global__ void extract_detections_kernel(
    const float* output, int num_anchors, float conf_thresh,
    Detection* dets, int* mask
) {
    extern __shared__ int shared_mask[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_anchors) return;

    float score = output[4 * num_anchors + i];

    if (score >= conf_thresh) {
        float cx = output[0 * num_anchors + i];
        float cy = output[1 * num_anchors + i];
        float w  = output[2 * num_anchors + i];
        float h  = output[3 * num_anchors + i];

        float x1 = cx - w / 2;
        float y1 = cy - h / 2;
        float x2 = cx + w / 2;
        float y2 = cy + h / 2;

        dets[i] = {x1, y1, x2, y2, score, 0};
        shared_mask[threadIdx.x] = 1;
    } else {
        shared_mask[threadIdx.x] = 0;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int j = 0; j < blockDim.x; ++j) {
            int idx = blockIdx.x * blockDim.x + j;
            if (idx >= num_anchors) break;
            mask[idx] = shared_mask[j];
        }
    }
}
// FP16 variant
__global__ void extract_detections_kernel_half(
    const __half* output, int num_anchors, float conf_thresh,
    Detection* dets, int* mask
) {
    extern __shared__ int shared_mask[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_anchors) return;

    float score = __half2float(output[4 * num_anchors + i]);

    if (score >= conf_thresh) {
        float cx = __half2float(output[0 * num_anchors + i]);
        float cy = __half2float(output[1 * num_anchors + i]);
        float w  = __half2float(output[2 * num_anchors + i]);
        float h  = __half2float(output[3 * num_anchors + i]);

        float x1 = cx - w * 0.5f;
        float y1 = cy - h * 0.5f;
        float x2 = cx + w * 0.5f;
        float y2 = cy + h * 0.5f;

        dets[i] = {x1, y1, x2, y2, score, 0};
        shared_mask[threadIdx.x] = 1;
    } else {
        shared_mask[threadIdx.x] = 0;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int j = 0; j < blockDim.x; ++j) {
            int idx = blockIdx.x * blockDim.x + j;
            if (idx >= num_anchors) break;
            mask[idx] = shared_mask[j];
        }
    }
}







__global__ void compact_detections_kernel(
    Detection* __restrict__ d_dets,
    int* __restrict__ d_mask,
    Detection* __restrict__ d_compacted,
    int* d_num_valid,
    int num_anchors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_anchors) return;

    if (d_mask[idx]) {
        int out_idx = atomicAdd(d_num_valid, 1);
        d_compacted[out_idx] = d_dets[idx];
    }
}






__global__ void nms_kernel_final_output(
    Detection* dets_in, int* d_count_compact, float iou_thresh,
    Detection* dets_out, int* final_count
) {
    __shared__ int proposals;
    if (threadIdx.x == 0) proposals = *d_count_compact;  // single global read
    __syncthreads();

    if (proposals == 0) return; 

    constexpr float eps = 1e-6f;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= proposals) return;

    float score_i = dets_in[i].score;
    bool keep = true;
    
    for (int j = 0; j < proposals; ++j) {

        if (j == i) continue;

        float score_j = dets_in[j].score;
        bool higher_or_tie_before = (score_j > score_i + eps) || (fabsf(score_j - score_i) <= eps && j < i);
        
        if (higher_or_tie_before) {
            if (iou(dets_in[i], dets_in[j]) > iou_thresh) {
                keep = false;
                break;
            }
        }
    }

    if (keep) {
        int idx = atomicAdd(final_count, 1);
        dets_out[idx] = dets_in[i];
    }
}






// ---------------------------- ACTION CLASSIFIER KERNELS ---------------------------- //

__global__ void cls_compute_params_kernel(
    const Detection* boxes, const int detIndex,
    const int frameW, const int frameH,
    const float scaleX, const float scaleY,
    const float pad_ratio,
    const int dstW, const int dstH,
    const int maxCropW, const int maxCropH, const int maxSquare,
    ClsDevParams* outParams
){
    // single-thread kernel; ignore any extra threads if launched with >1
    if (blockIdx.x | blockIdx.y | threadIdx.x | threadIdx.y) return;

    Detection det = boxes[detIndex];

    // Map YOLO/model-space box -> original frame-space using provided scales
    float x0 = fminf(det.x1, det.x2) * scaleX;
    float y0 = fminf(det.y1, det.y2) * scaleY;
    float x1 = fmaxf(det.x1, det.x2) * scaleX;
    float y1 = fmaxf(det.y1, det.y2) * scaleY;

    // Clamp to frame bounds (float domain)
    x0 = fminf(fmaxf(x0, 0.f), (float)(frameW - 1));
    x1 = fminf(fmaxf(x1, 0.f), (float)(frameW - 1));
    y0 = fminf(fmaxf(y0, 0.f), (float)(frameH - 1));
    y1 = fminf(fmaxf(y1, 0.f), (float)(frameH - 1));

    // Size
    float w = fmaxf(1.f, x1 - x0);
    float h = fmaxf(1.f, y1 - y0);

    // Symmetric padding based on longer side
    float maxSide = fmaxf(w, h);
    float pad = fmaxf(0.f, pad_ratio) * maxSide;

    float px0 = x0 - pad;
    float py0 = y0 - pad;
    float px1 = x1 + pad;
    float py1 = y1 + pad;

    // Clamp padded rect to frame bounds
    px0 = fmaxf(0.f, px0);
    py0 = fmaxf(0.f, py0);
    px1 = fminf((float)frameW, px1);   // half-open end
    py1 = fminf((float)frameH, py1);

    int bx = (int)floorf(px0);
    int by = (int)floorf(py0);
    int ex = (int)ceilf(px1);
    int ey = (int)ceilf(py1);

    int W = ex - bx;
    int H = ey - by;
    if (W < 1) { W = 1; bx = min(bx, frameW - 1); }
    if (H < 1) { H = 1; by = min(by, frameH - 1); }

    // Respect scratch crop limits
    if (W > maxCropW) {
        int cx = bx + W / 2;
        W = maxCropW;
        bx = max(0, min(cx - W / 2, frameW - W));
    }
    if (H > maxCropH) {
        int cy = by + H / 2;
        H = maxCropH;
        by = max(0, min(cy - H / 2, frameH - H));
    }

    // Square side
    int S = max(W, H);
    if (S > maxSquare) {
        float s = (float)maxSquare / (float)S;
        W = max(1, (int)floorf(W * s));
        H = max(1, (int)floorf(H * s));
        S = max(W, H);
        // keep crop inside frame after shrink
        if (bx + W > frameW) bx = frameW - W;
        if (by + H > frameH) by = frameH - H;
        bx = max(0, bx);
        by = max(0, by);
    }

    // Where the W×H crop sits inside the S×S square (centered)
    int ox = (S - W) / 2;
    int oy = (S - H) / 2;

    // scales for S×S -> dst (used by the resize kernel)
    float s2dst_x = (float)S / (float)dstW;
    float s2dst_y = (float)S / (float)dstH;

    ClsDevParams p;
    p.bx = bx; p.by = by;
    p.W  = W;  p.H  = H;
    p.S  = S;
    p.ox = ox; p.oy = oy;
    p.s2dst_x = s2dst_x;
    p.s2dst_y = s2dst_y;

    *outParams = p;
}





__global__ void crop_img_kernel(
    const unsigned char* __restrict__ d_frame_bgr,
    int frameW, int frameH,
    const ClsDevParams* __restrict__ d_params,
    unsigned char* __restrict__ d_out_crop
){
    // local copy to registers for efficiency
    const ClsDevParams p = *d_params;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;  
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= p.W || y >= p.H) return;

    // source coords in the original frame
    const int sx = p.bx + x;
    const int sy = p.by + y;

    // safety guard
    if ((unsigned)sx >= (unsigned)frameW || (unsigned)sy >= (unsigned)frameH) {
        // write black just in case
        const int dst = (y * p.W + x) * 3;
        d_out_crop[dst + 0] = 0;
        d_out_crop[dst + 1] = 0;
        d_out_crop[dst + 2] = 0;
        return;
    }

    // interleaved BGR copy
    const int src = (sy * frameW + sx) * 3;
    const int dst = (y * p.W + x) * 3;

    // copy 3 channels
    d_out_crop[dst + 0] = d_frame_bgr[src + 0];
    d_out_crop[dst + 1] = d_frame_bgr[src + 1];
    d_out_crop[dst + 2] = d_frame_bgr[src + 2];
}





__global__ void pad_to_square_kernel(
    const unsigned char* __restrict__ d_crop,
    const ClsDevParams*  __restrict__ d_params,
    unsigned char*       __restrict__ d_square
){
    const ClsDevParams p = *d_params;

    const int x = blockIdx.x * blockDim.x + threadIdx.x; 
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= p.S || y >= p.S) return;

    // destination index in S×S canvas
    const int dst = (y * p.S + x) * 3;

    // check if (x,y) falls inside the placed crop rectangle [ox, ox+W) x [oy, oy+H)
    const int cx = x - p.ox;   
    const int cy = y - p.oy;   
    const bool in_crop = (cx >= 0 && cx < p.W && cy >= 0 && cy < p.H);

    if (!in_crop) {
        // black padding
        d_square[dst + 0] = 0;
        d_square[dst + 1] = 0;
        d_square[dst + 2] = 0;
        return;
    }

    // copy pixel from crop (tightly packed W×H)
    const int src = (cy * p.W + cx) * 3;
    d_square[dst + 0] = d_crop[src + 0];
    d_square[dst + 1] = d_crop[src + 1];
    d_square[dst + 2] = d_crop[src + 2];
}





// FP32
__global__ void bgr_to_gray_norm_kernel(
    float* __restrict__ out_gray,
    const unsigned char* __restrict__ in_bgr,
    int width, int height
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx  = y * width + x;
    int base = idx * 3;

    // BT.601 luma on BGR order: Y = 0.114B + 0.587G + 0.299R
    float b = static_cast<float>(in_bgr[base + 0]);
    float g = static_cast<float>(in_bgr[base + 1]);
    float r = static_cast<float>(in_bgr[base + 2]);

    float y601 = 0.114f * b + 0.587f * g + 0.299f * r;
    out_gray[idx] = y601 * (1.0f / 255.0f);   // normalize to [0,1]
}
// FP16 variant
__global__ void bgr_to_gray_norm_kernel_half(
    __half* __restrict__ out_gray,
    const unsigned char* __restrict__ in_bgr,
    int width, int height
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx  = y * width + x;
    int base = idx * 3;

    // BT.601 luma on BGR order: Y = 0.114B + 0.587G + 0.299R
    float b = static_cast<float>(in_bgr[base + 0]);
    float g = static_cast<float>(in_bgr[base + 1]);
    float r = static_cast<float>(in_bgr[base + 2]);

    float y601 = 0.114f * b + 0.587f * g + 0.299f * r;
    float y_norm = y601 * (1.0f / 255.0f);  // normalize to [0,1]

    out_gray[idx] = __float2half_rn(y_norm);
}





// FP16 Batched variant
__global__ void bgr_to_gray_norm_kernel_batched_half(
    __half* __restrict__ out_gray_base,
    const unsigned char* __restrict__ in_bgr_base,
    int width, int height,
    int batchN
){
    const int b = blockIdx.z;
    if (b >= batchN) return;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const size_t inSliceStride  = (size_t)width * height * 3;  
    const size_t outSliceStride = (size_t)width * height;      

    const size_t idx  = (size_t)y * width + x;
    const size_t in3  = (size_t)b * inSliceStride  + idx * 3;
    const size_t out1 = (size_t)b * outSliceStride + idx;

    float bch = (float)in_bgr_base[in3 + 0];
    float gch = (float)in_bgr_base[in3 + 1];
    float rch = (float)in_bgr_base[in3 + 2];

    float y601 = 0.114f * bch + 0.587f * gch + 0.299f * rch;
    float y_norm = y601 * (1.0f / 255.0f);

    out_gray_base[out1] = __float2half_rn(y_norm);
}










__global__ void cls_compute_params_kernel_batched(
    const Detection* __restrict__ d_boxes,
    int frameW, int frameH,
    float scaleX, float scaleY,
    float pad_ratio,
    int dstW, int dstH,
    int maxCropW, int maxCropH, int maxSquare,
    ClsDevParams* __restrict__ d_params_array,
    int batchN
){
    const int b = blockIdx.z;
    if (b >= batchN) return;

    // Map bbox to CAMERA space
    Detection det = d_boxes[b];
    float x1 = det.x1 * scaleX;
    float y1 = det.y1 * scaleY;
    float x2 = det.x2 * scaleX;
    float y2 = det.y2 * scaleY;

    if (x2 < x1) { float t=x1; x1=x2; x2=t; }
    if (y2 < y1) { float t=y1; y1=y2; y2=t; }

    // Make square with padding
    float bw = fmaxf(1.0f, x2 - x1);
    float bh = fmaxf(1.0f, y2 - y1);
    float side = fmaxf(bw, bh);
    float pad  = side * pad_ratio;

    float cx = 0.5f*(x1 + x2);
    float cy = 0.5f*(y1 + y2);
    float half = 0.5f*side + pad;

    // Square ROI (float)
    float rx1 = cx - half;
    float ry1 = cy - half;
    float rx2 = cx + half;
    float ry2 = cy + half;

    // Integer crop window clipped to frame
    int ix0 = static_cast<int>(floorf(rx1));
    int iy0 = static_cast<int>(floorf(ry1));
    int ix1 = static_cast<int>(ceilf (rx2));
    int iy1 = static_cast<int>(ceilf (ry2));

    ix0 = max(0, min(ix0, frameW));
    iy0 = max(0, min(iy0, frameH));
    ix1 = max(0, min(ix1, frameW));
    iy1 = max(0, min(iy1, frameH));

    int W = max(0, ix1 - ix0);
    int H = max(0, iy1 - iy0);

    // Clamp to scratch capacities
    if (W > maxCropW) W = maxCropW;
    if (H > maxCropH) H = maxCropH;

    // Square side S (clamped)
    int S = static_cast<int>(ceilf(fmaxf(rx2 - rx1, ry2 - ry1)));
    if (S < 1) S = 1;
    if (S > maxSquare) S = maxSquare;

    // Offsets of crop top-left within the SxS square
    float fx_off = (float)ix0 - rx1;  // how far inside the square the crop begins (x)
    float fy_off = (float)iy0 - ry1;  // (y)
    int ox = static_cast<int>(floorf(fx_off + 0.5f)); // round to nearest int
    int oy = static_cast<int>(floorf(fy_off + 0.5f));

    // Clamp offsets so W×H fits inside S×S
    ox = max(0, min(ox, S - W));
    oy = max(0, min(oy, S - H));

    // Scale from SxS to dst
    float s2dst_x = (S > 0) ? (static_cast<float>(S) / static_cast<float>(dstW)) : 0.0f;
    float s2dst_y = (S > 0) ? (static_cast<float>(S) / static_cast<float>(dstH)) : 0.0f;

    // Write params
    ClsDevParams p;
    p.bx = ix0;  p.by = iy0;   
    p.W  = W;    p.H  = H;     
    p.S  = S;                  
    p.ox = ox;   p.oy = oy;    
    p.s2dst_x = s2dst_x;
    p.s2dst_y = s2dst_y;

    d_params_array[b] = p;
}






// Pad the W×H crop to S×S square (letterbox) with black pixels
__global__ void pad_to_square_kernel_batched(
    const unsigned char* __restrict__ d_crop_base,
    const ClsDevParams*  __restrict__ d_params_array,
    unsigned char*        __restrict__ d_square_base,
    int maxCropW, int maxCropH,
    int maxSquare,
    int batchN
){
    const int b = blockIdx.z;  // detection index
    if (b >= batchN) return;

    // Per-detection params
    const ClsDevParams p = d_params_array[b];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Only write within the S×S canvas of this detection
    if (x >= p.S || y >= p.S) return;

    // Slice strides (bytes) for this batched layout
    const size_t cropSliceStride   = static_cast<size_t>(maxCropW)  * maxCropH  * 3;
    const size_t squareSliceStride = static_cast<size_t>(maxSquare) * maxSquare * 3;

    // Destination index in this detection's S×S canvas
    const size_t dst =
        static_cast<size_t>(b) * squareSliceStride +
        (static_cast<size_t>(y) * maxSquare + x) * 3;

    // Where does (x,y) land relative to the placed crop
    const int cx = x - p.ox;
    const int cy = y - p.oy;
    const bool in_crop = (cx >= 0 && cx < p.W && cy >= 0 && cy < p.H);

    if (!in_crop) {
        // black padding
        d_square_base[dst + 0] = 0;
        d_square_base[dst + 1] = 0;
        d_square_base[dst + 2] = 0;
        return;
    }

    // Read from the crop slice (packed W×H inside a maxCropW×maxCropH slice)
    const size_t src =  static_cast<size_t>(b) * cropSliceStride + 
                        (static_cast<size_t>(cy) * maxCropW + cx) * 3;

    d_square_base[dst + 0] = d_crop_base[src + 0];
    d_square_base[dst + 1] = d_crop_base[src + 1];
    d_square_base[dst + 2] = d_crop_base[src + 2];
}






*/