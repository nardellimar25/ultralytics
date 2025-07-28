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
    // Calculate the bboxes area
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

// Kernel to extract detections from the output tensor
// Loops through each anchor and checks if the score exceeds the confidence threshold
// it calculates the bounding box coordinates and stores the detection
__global__ void extract_detections_kernel(const float* output, int num_anchors, float conf_thresh, Detection* dets, int* mask) {


    // Shared memory
    //__shared__ Detection shared_dets[256];  // same as threads per block
    //__shared__ int shared_count;  // shared count for number of detections found
    __shared__ int shared_mask[256];  // mask for detections found, same

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
        for (int i = 0; i < 256; i++){
            mask[i+blockDim.x*blockIdx.x] = shared_mask[i];
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
        int idx = atomicAdd(final_count, 1);
        dets_out[idx] = dets_in[i];
    }
}


// === Host function ===
// This function runs the post-processing on the GPU
// It extracts detections, applies NMS, and returns the final detections
int run_postprocess_gpu(const float* d_output, int num_anchors, float conf_thresh,
                        float iou_thresh, int max_dets, Detection* d_final)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);


    // d_dets for intermediate detections
    Detection *d_dets, *d_compacted;    

    // d_count_in for intermediate count, d_count_final for final count
    int *d_count_final, *d_mask;
    
   

    // Allocate memory on the device
    //float timeStart = static_cast<float>(clock());
    cudaMalloc(&d_dets, num_anchors * sizeof(Detection));
    cudaMalloc(&d_count_final, sizeof(int));
    cudaMalloc(&d_mask, num_anchors * sizeof(int));
    cudaMalloc(&d_compacted, num_anchors * sizeof(Detection));
    cudaMemset(d_count_final, 0, sizeof(int));
    //float timeEnd = static_cast<float>(clock());
    //std::cout << "-Memory allocation: " << (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000 << " ms\n";

    dim3 threads(256);
    dim3 blocks((num_anchors + threads.x - 1) / threads.x);
    //size_t shared_mem_size = threads.x * sizeof(Detection);

    // Filtering kernel
    
    extract_detections_kernel<<<blocks, threads>>>(d_output, num_anchors, conf_thresh, d_dets, d_mask);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "extract_detections_kernel failed: " << cudaGetErrorString(err) << std::endl;
    }
    //float milliseconds = 0;

    //std::cout << " extract det kernel: " << milliseconds << " ms" << std::endl;

    thrust::device_ptr<Detection> det_ptr(d_dets);
    thrust::device_ptr<int> mask_ptr(d_mask);
    auto compacted_ptr = thrust::device_pointer_cast(d_compacted);

    // Perform filtering (GPU-side compaction)
    //timeStart = static_cast<float>(clock());
    auto end = thrust::copy_if(
        thrust::device,
        det_ptr, det_ptr + num_anchors,                 // input range
        mask_ptr,                                       // filter based on this
        compacted_ptr,       // output destination
        thrust::identity<int>()                         // copy where mask == 1
    );
    //timeEnd = static_cast<float>(clock());
    //std::cout << " thrust compaction: " << (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000 << " ms\n";

    // Return to host number of detections found
    int num_dets = end - compacted_ptr;

    //std::cout << " Valid dets: " << num_dets << std::endl;

/*
    // Return to host number of detections found
    int num_dets = 0;
    timeStart = static_cast<float>(clock());
    cudaMemcpy(&num_dets, d_count_in, sizeof(int), cudaMemcpyDeviceToHost);
    timeEnd = static_cast<float>(clock());
    std::cout << " Count extraction memcpy time: " << (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000 << " ms\n";
*/
    // If no detections early exit
    /*if (num_dets == 0) {
        cudaFree(d_dets); 
        cudaFree(d_count_in); 
        cudaFree(d_count_final);
        return 0;
    }*/

    dim3 blocks_nms((num_dets + threads.x - 1) / threads.x);

    // NMS kernel
    //cudaEventRecord(start);
    nms_kernel_final_output<<<blocks_nms, threads>>>(d_compacted, num_dets, iou_thresh, d_final, d_count_final);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "extract_detections_kernel failed: " << cudaGetErrorString(err) << std::endl;
    }
    //cudaEventRecord(stop);
    //cudaEventSynchronize(stop);
    //cudaEventElapsedTime(&milliseconds, start, stop);
    //std::cout << " NMS kernel: " << milliseconds << " ms" << std::endl;


    // Return to host number of final detections
    int det_count_final = 0;

    //timeStart = static_cast<float>(clock());
    cudaMemcpy(&det_count_final, d_count_final, sizeof(int), cudaMemcpyDeviceToHost);
    //timeEnd = static_cast<float>(clock());
    //std::cout << " Final extraction memcpy: " << (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000 << " ms\n";

    //timeStart = static_cast<float>(clock());
    cudaFree(d_dets);
    cudaFree(d_mask);
    cudaFree(d_compacted);
    cudaFree(d_count_final);
    //timeEnd = static_cast<float>(clock());
    //std::cout << " Memory deallocation: " << (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000 << " ms\n";
    float milliseconds = 0;

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << " (inside) postproc time : " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return det_count_final;
}

