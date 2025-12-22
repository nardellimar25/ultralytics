#include "kernels/yolo_kernels.cuh"
#include "cuda_helpers.cuh"
#include "cuda_structs.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>


// ----------------------------   YOLO PREPROCESS KERNELS ---------------------------- //

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


// Resize + preprocess (normalize) fused kernel for half precision
__global__ void resize_preprocess_fused_batched_half(
    __half* __restrict__ out_chw,
    const unsigned char* __restrict__ in_bgr,
    int in_w, int in_h,
    int out_w, int out_h,
    float scale_x, float scale_y,
    int B
){
    const int cam = blockIdx.z;
    if (cam >= B) return;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_w || y >= out_h) return;

    const int in_stride  = in_w  * in_h  * 3;
    const int plane_out  = out_w * out_h;
    const int out_cam_off = cam * 3 * plane_out;

    const unsigned char* in = in_bgr + cam * in_stride;

    // map output pixel -> input coords
    const float src_x = x * scale_x;
    const float src_y = y * scale_y;

    int x0 = (int)floorf(src_x);
    int y0 = (int)floorf(src_y);
    x0 = max(0, min(x0, in_w - 1));
    y0 = max(0, min(y0, in_h - 1));

    const int x1 = min(x0 + 1, in_w - 1);
    const int y1 = min(y0 + 1, in_h - 1);

    const float dx = src_x - (float)x0;
    const float dy = src_y - (float)y0;

    // sample BGR as floats
    auto pix = [&](int xx, int yy, int c)->float {
        return (float)in[(yy * in_w + xx) * 3 + c];
    };

    const float b00 = pix(x0,y0,0), g00 = pix(x0,y0,1), r00 = pix(x0,y0,2);
    const float b01 = pix(x1,y0,0), g01 = pix(x1,y0,1), r01 = pix(x1,y0,2);
    const float b10 = pix(x0,y1,0), g10 = pix(x0,y1,1), r10 = pix(x0,y1,2);
    const float b11 = pix(x1,y1,0), g11 = pix(x1,y1,1), r11 = pix(x1,y1,2);

    // bilinear
    const float b0 = b00 + (b01 - b00) * dx;
    const float b1 = b10 + (b11 - b10) * dx;
    const float b  = b0  + (b1  - b0 ) * dy;

    const float g0 = g00 + (g01 - g00) * dx;
    const float g1 = g10 + (g11 - g10) * dx;
    const float g  = g0  + (g1  - g0 ) * dy;

    const float r0 = r00 + (r01 - r00) * dx;
    const float r1 = r10 + (r11 - r10) * dx;
    const float r  = r0  + (r1  - r0 ) * dy;

    // normalize [0..1] and write CHW in RGB order
    const float inv255 = 1.f / 255.f;
    const int dst = y * out_w + x;

    out_chw[out_cam_off + 0 * plane_out + dst] = __float2half_rn(r * inv255);
    out_chw[out_cam_off + 1 * plane_out + dst] = __float2half_rn(g * inv255);
    out_chw[out_cam_off + 2 * plane_out + dst] = __float2half_rn(b * inv255);
}



// ----------------------------   YOLO POSTPROCESS KERNELS --------------------------- //

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
// OLD
// __global__ void extract_detections_kernel_batched_half(
//     const __half* output,
//     int B,
//     int num_anchors,  
//     float conf_thresh,
//     Detection* dets,  
//     int* mask        
// ){
//     extern __shared__ int shared_mask[];

//     const int A  = num_anchors;
//     const int BA = B * A;

//     int g = blockIdx.x * blockDim.x + threadIdx.x;
//     if (g >= BA) return;

//     int b = g / A;
//     int a = g % A;

//     constexpr int K = 84;
//     const __half* out_b = output + (size_t)b * K * A;

//     auto h2f = __half2float;

//     float score = h2f(out_b[4 * A + a]);  // class 0 prob
//     Detection d{};
//     int keep = 0;

//     if (score >= conf_thresh) {
//         float cx = h2f(out_b[0 * A + a]);
//         float cy = h2f(out_b[1 * A + a]);
//         float w  = h2f(out_b[2 * A + a]);
//         float h  = h2f(out_b[3 * A + a]);

//         d.x1 = cx - 0.5f * w;
//         d.y1 = cy - 0.5f * h;
//         d.x2 = cx + 0.5f * w;
//         d.y2 = cy + 0.5f * h;
//         d.score = score;
//         d.class_id = 0;
//         d.cam_index = b;

//         dets[g] = d;
//         keep = 1;
//     }
//     shared_mask[threadIdx.x] = keep;

//     __syncthreads();

//     if (threadIdx.x == 0) {
//         int base = blockIdx.x * blockDim.x;
//         for (int j = 0; j < blockDim.x; ++j) {
//             int idx = base + j;
//             if (idx >= BA) break;
//             mask[idx] = shared_mask[j];
//         }
//     }
// }
__global__ void extract_detections_kernel_batched_half(
    const __half* output,
    int B,
    int num_anchors,
    float conf_thresh,
    Detection* dets,
    int* mask
){
    const int A  = num_anchors;
    const int BA = B * A;

    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= BA) return;

    const int b = g / A;
    const int a = g - b * A;   // a = g % A

    constexpr int K = 84;

    // output layout assumed: [B, K, A] contiguous in A
    const __half* out_b = output + (size_t)b * K * A;

    // class-0 confidence at channel 4
    const float score = __half2float(out_b[4 * A + a]);

    const int keep = (score >= conf_thresh) ? 1 : 0;
    mask[g] = keep;

    if (!keep) return;

    const float cx = __half2float(out_b[0 * A + a]);
    const float cy = __half2float(out_b[1 * A + a]);
    const float w  = __half2float(out_b[2 * A + a]);
    const float h  = __half2float(out_b[3 * A + a]);

    Detection d{};
    d.x1 = cx - 0.5f * w;
    d.y1 = cy - 0.5f * h;
    d.x2 = cx + 0.5f * w;
    d.y2 = cy + 0.5f * h;
    d.score     = score;
    d.class_id  = 0;
    d.cam_index = b;

    dets[g] = d;
}



// Compact yolo detections based on mask
// better for expectedly low number of detections
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
    // cam_index rides along
    d_compacted[out_idx] = d_dets[g];  

}
// NEW: warp-level compaction better for larger number of detections
// __global__ void compact_detections_kernel_batched(
//     const Detection* __restrict__ d_dets,
//     const int*       __restrict__ d_mask,
//     Detection*       __restrict__ d_compacted,
//     int*             __restrict__ d_count_compact,
//     int              num_anchors,   // not needed anymore
//     int              BA
// ){
//     const int g = blockIdx.x * blockDim.x + threadIdx.x;

//     // predicate (guard BA)
//     const int keep = (g < BA) ? (d_mask[g] != 0) : 0;

//     // ---- warp-local compaction ----
//     const unsigned full = 0xFFFFFFFFu;
//     const unsigned ballot = __ballot_sync(full, keep);

//     const int lane   = threadIdx.x & 31;
//     const int warpId = threadIdx.x >> 5;               // warp index in block
//     const int numWarps = (blockDim.x + 31) >> 5;

//     // count kept in this warp
//     const int warpCount = __popc(ballot);

//     __shared__ int warpCounts[32];   // enough for up to 1024 threads/block (32 warps)
//     __shared__ int warpOffsets[32];
//     __shared__ int blockBase;

//     if (lane == 0) warpCounts[warpId] = warpCount;
//     __syncthreads();

//     // ---- block-level exclusive scan over warps (tiny, done by warp 0) ----
//     if (warpId == 0 && lane == 0) {
//         int sum = 0;
//         for (int w = 0; w < numWarps; ++w) {
//             warpOffsets[w] = sum;
//             sum += warpCounts[w];
//         }
//         // one atomic per block
//         blockBase = atomicAdd(d_count_compact, sum);
//     }
//     __syncthreads();

//     // if not kept, we're done
//     if (!keep) return;

//     // offset within warp: number of kept lanes before me
//     const unsigned lowerMask = (lane == 0) ? 0u : ((1u << lane) - 1u);
//     const int warpLocal = __popc(ballot & lowerMask);

//     const int outIdx = blockBase + warpOffsets[warpId] + warpLocal;
//     d_compacted[outIdx] = d_dets[g];
// }



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