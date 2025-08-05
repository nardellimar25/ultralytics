#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <time.h>

#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>


#include "postProcess.cuh"

// Function to calculate Intersection over Union (IoU)
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

// Kernel to preprocess the input image
// Converts HWC/BGR image to CHW/float and normalizes it
__global__ void preprocess_kernel(const uchar* input_bgr, float* d_input, int img_width, int img_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= img_width || y >= img_height) return;

    int src_idx = (y * img_width + x) * 3;
    uchar b = input_bgr[src_idx + 0];
    uchar g = input_bgr[src_idx + 1];
    uchar r = input_bgr[src_idx + 2];

    int dst_idx = y * img_width + x;

    d_input[0 * img_width * img_height + dst_idx] = r / 255.0f;
    d_input[1 * img_width * img_height + dst_idx] = g / 255.0f;
    d_input[2 * img_width * img_height + dst_idx] = b / 255.0f;

}

// Kernel to extract detections from the output tensor
// Loops through each anchor and checks if the score exceeds the confidence threshold
// it calculates the bounding box coordinates and stores the detection
__global__ void extract_detections_kernel(const float* output, int num_anchors, float conf_thresh, Detection* dets, int* mask) {

    // Shared memory
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
        shared_mask[threadIdx.x] = 1;  // detection found

    } else {
        shared_mask[threadIdx.x] = 0;
    }

    __syncthreads();

    if(threadIdx.x == 0) {
        for (int j = 0; j < blockDim.x; j++){
            int idx = blockIdx.x * blockDim.x + j;

            if (idx >= num_anchors) break;
           
            mask[idx] = shared_mask[j];  // Copy the mask to global memory
        }
    }

}


// Kernel to perform Non-Maximum Suppression (NMS) [brute force!]
// It iterates through the detections and removes those that have high IoU with a higher scored detection
// The final detections are stored in the output array
__global__ void nms_kernel_final_output(const Detection* dets_in, int num_in, float iou_thresh, Detection* dets_out, int* final_count) {

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
        int idx = atomicAdd(final_count, 1);       //works fine bc we dont have many detections
        dets_out[idx] = dets_in[i];
    }
}





// === Host functions ===

// This function runs the pre-processing on the GPU
// It converts the input image from BGR to float and normalizes it
void run_preprocess_gpu(const cv::Mat& resized_frame, float* d_input, int img_width, int img_height, size_t input_size, uchar* d_bgr, cudaStream_t stream) {

    // Copy the resized frame to GPU
    cudaMemcpyAsync(d_bgr, resized_frame.data, input_size, cudaMemcpyHostToDevice, stream);

    dim3 block(16, 16);
    dim3 grid((img_width + 15) / 16, (img_height + 15) / 16);

    // Preprocess kernel 
    preprocess_kernel<<<grid, block, 0, stream>>>(d_bgr, d_input, img_width, img_height);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "preprocess_kernel failed: " << cudaGetErrorString(err) << std::endl;
    }


}

// This function runs the post-processing on the GPU
// It extracts detections, applies NMS, and returns the final detections
int run_postprocess_gpu(const float* d_output, int num_anchors, float conf_thresh,
                        float iou_thresh, int max_dets, Detection* d_final, cudaStream_t stream,
                        Detection *d_dets, Detection *d_compacted, int *d_mask, int *d_count_final)
{

    cudaMemset(d_count_final, 0, sizeof(int));

    dim3 threads(256);
    dim3 blocks((num_anchors + threads.x - 1) / threads.x);
    size_t shared_mem_size = threads.x * sizeof(int);

    // Filtering kernel
    extract_detections_kernel<<<blocks, threads, shared_mem_size, stream>>>(d_output, num_anchors, conf_thresh, d_dets, d_mask);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "extract_detections_kernel failed: " << cudaGetErrorString(err) << std::endl;
    }


    thrust::device_ptr<Detection> det_ptr(d_dets);
    thrust::device_ptr<int> mask_ptr(d_mask);
    auto compacted_ptr = thrust::device_pointer_cast(d_compacted);

    // Compact the performed filtering (GPU)
    auto end = thrust::copy_if(
        thrust::device,
        det_ptr, det_ptr + num_anchors,                 // input range
        mask_ptr,                                       // filter based on this
        compacted_ptr,                                  // output destination
        thrust::identity<int>()                         // copy where mask == 1
    );

    // Check number of detections found
    int num_dets = end - compacted_ptr;
    //std::cout << "Number of detections after filtering: " << num_dets << std::endl;

    // If no detections early exit
    if (num_dets == 0) {

        return 0;

    }

    dim3 blocks_nms((num_dets + threads.x - 1) / threads.x);

    // NMS kernel
    nms_kernel_final_output<<<blocks_nms, threads, 0, stream>>>(d_compacted, num_dets, iou_thresh, d_final, d_count_final);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "nms_kernel_final_output failed: " << cudaGetErrorString(err) << std::endl;
    }

    // Return to host number of final detections
    int det_count_final = 0;

    cudaMemcpy(&det_count_final, d_count_final, sizeof(int), cudaMemcpyDeviceToHost);

    return det_count_final;

}

