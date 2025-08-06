#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <iostream>

#include "cuda_kernels.cuh"     
#include "cuda_detection_struct.h"
#include "cuda_postprocess.cuh"

// This function runs the post-processing on the GPU
// It extracts detections, applies NMS, and returns the final detections
int run_postprocess_gpu(
    const float* d_output,
    int num_anchors,
    float conf_thresh,
    float iou_thresh,
    int max_dets,
    Detection* d_final,
    cudaStream_t stream,
    Detection* d_dets,
    Detection* d_compacted,
    int* d_mask,
    int* d_count_final
) {
    // Reset final detection counter on device
    cudaMemset(d_count_final, 0, sizeof(int));

    dim3 threads(256);
    dim3 blocks((num_anchors + threads.x - 1) / threads.x);
    size_t shared_mem_size = threads.x * sizeof(int);

    // Filtering kernel
    extract_detections_kernel<<<blocks, threads, shared_mem_size, stream>>>(
        d_output, num_anchors, conf_thresh, d_dets, d_mask
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "extract_detections_kernel failed: "
                  << cudaGetErrorString(err) << std::endl;
    }

    // Compact filtered detections using thrust::copy_if
    thrust::device_ptr<Detection> det_ptr(d_dets);
    thrust::device_ptr<int> mask_ptr(d_mask);
    auto compacted_ptr = thrust::device_pointer_cast(d_compacted);

    auto end = thrust::copy_if(
        thrust::device,
        det_ptr, det_ptr + num_anchors,
        mask_ptr,
        compacted_ptr,
        thrust::identity<int>()
    );

    // Number of detections after filtering
    int num_dets = end - compacted_ptr;

    // Early return if no detections
    if (num_dets == 0) {
        return 0;
    }

    dim3 blocks_nms((num_dets + threads.x - 1) / threads.x);

    // Apply NMS kernel
    nms_kernel_final_output<<<blocks_nms, threads, 0, stream>>>(
        d_compacted, num_dets, iou_thresh, d_final, d_count_final
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "nms_kernel_final_output failed: "
                  << cudaGetErrorString(err) << std::endl;
    }

    // Copy back final detection count
    int det_count_final = 0;
    cudaMemcpy(&det_count_final, d_count_final, sizeof(int), cudaMemcpyDeviceToHost);

    return det_count_final;
}
