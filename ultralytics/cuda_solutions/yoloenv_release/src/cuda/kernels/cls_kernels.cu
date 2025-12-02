#include "kernels/cls_kernels.cuh"
#include "cuda_helpers.cuh"
#include "cuda_structs.cuh"

#include <cuda_runtime.h>


// ---------------------------- ACTION CLASSIFIER KERNELS ---------------------------- //


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
// TODO: use only this faster strided version
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