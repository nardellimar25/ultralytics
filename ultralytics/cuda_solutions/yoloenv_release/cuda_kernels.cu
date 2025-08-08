#include "cuda_kernels.cuh"

// === Utility functions ===

__device__ float iou(const Detection& a, const Detection& b) {
    float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);

    float x1 = max(a.x1, b.x1);
    float y1 = max(a.y1, b.y1);
    float x2 = min(a.x2, b.x2);
    float y2 = min(a.y2, b.y2);

    float interW = max(0.f, x2 - x1);
    float interH = max(0.f, y2 - y1);
    float interArea = interW * interH;

    return interArea / (areaA + areaB - interArea + 1e-6f);
}

__device__ float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

// === Kernels ===

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

__global__ void preprocess_kernel(
    float* d_output, const uchar* d_input_bgr,
    int img_width, int img_height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= img_width || y >= img_height) return;

    int src_idx = (y * img_width + x) * 3;
    uchar b = d_input_bgr[src_idx + 0];
    uchar g = d_input_bgr[src_idx + 1];
    uchar r = d_input_bgr[src_idx + 2];

    int dst_idx = y * img_width + x;

    d_output[0 * img_width * img_height + dst_idx] = r / 255.0f;
    d_output[1 * img_width * img_height + dst_idx] = g / 255.0f;
    d_output[2 * img_width * img_height + dst_idx] = b / 255.0f;
}

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

__global__ void compact_detections_kernel(
    const Detection* __restrict__ d_dets,
    const int* __restrict__ d_mask,
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
    const Detection* dets_in, int num_in, float iou_thresh,
    Detection* dets_out, int* final_count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_in) return;

    bool keep = true;
    for (int j = 0; j < num_in; ++j) {
        if (j == i) continue;
        if (dets_in[j].score > dets_in[i].score) {
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
