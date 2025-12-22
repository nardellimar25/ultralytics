#include <cuda_runtime.h>
#include <iostream>
#include <cuda_fp16.h>  

#include "cuda_yolo_postprocess.cuh"
#include "kernels/cuda_kernels.cuh"     
#include "cuda_structs.cuh"


// This function runs the post-processing on the GPU
// It extracts detections, applies NMS, and returns the final detections
void yolo_postprocess_gpu(
    const float* d_output,
    int num_anchors,
    float conf_thresh,
    float iou_thresh,
    Detection* d_final,
    cudaStream_t stream,
    Detection* d_dets,
    Detection* d_compacted,
    int* d_mask,
    int* d_count_compact,
    int* d_count_final,
    int* h_count_final,
    cudaEvent_t ev_count_final,
    int num_cameras
) {

    // TODO : move as parameter
    int max_dets = 100;

    // Reset intermediate detection counter on device
    //cudaMemsetAsync(d_count_compact, 0, sizeof(int));
    cudaMemsetAsync(d_count_compact, 0, sizeof(int), stream);
    // Reset final detection counter on device
    cudaMemsetAsync(d_count_final, 0, sizeof(int), stream);
    // Reset detections on device
    /* TODO : needs update to match max detections */
    cudaMemsetAsync(d_final, 0, max_dets * sizeof(Detection), stream);

    // total anchors across the whole batch
    int tot_anchors = num_cameras * num_anchors;

    dim3 threads(256);
    dim3 blocks((tot_anchors + threads.x - 1) / threads.x);
    size_t shared_mem_size = threads.x * sizeof(int);

    // Filtering kernel
    extract_detections_kernel_batched<<<blocks, threads, shared_mem_size, stream>>>(
        d_output,         // [B,84,A] contiguous in A
        num_cameras,                // batch size
        num_anchors,      // A
        conf_thresh,
        d_dets,           // device array sized [B*A]
        d_mask            // device array sized [B*A]
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "extract_detections_kernel failed: "
                  << cudaGetErrorString(err) << std::endl;
    }

    // Batched compaction
    dim3 blocks2((tot_anchors + threads.x - 1) / threads.x);

    compact_detections_kernel_batched<<<blocks2, threads, 0, stream>>>(
        d_dets,               
        d_mask,               
        d_compacted,          
        d_count_compact,      
        num_anchors,          
        tot_anchors
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "compact_detections_kernel_batched failed: "
                << cudaGetErrorString(err) << std::endl;
    }

    dim3 blocks_nms((num_anchors + threads.x - 1) / threads.x);

    // Apply NMS kernel
    nms_kernel_final_output_batched<<<blocks_nms, threads, 0, stream>>>(
        d_compacted,        
        d_count_compact,     
        iou_thresh,
        d_final,             
        d_count_final,       
        num_anchors,         
        max_dets,
        num_cameras
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "nms_kernel_final_output_batched failed: "
                  << cudaGetErrorString(err) << std::endl;
    }

    // Copy back final detection count
    cudaMemcpyAsync(h_count_final, d_count_final, sizeof(int), cudaMemcpyDeviceToHost, stream);

    // Mark exactly when the host may read *h_count_final
    cudaEventRecord(ev_count_final, stream);

    return;
}
// NEW: FP16 overload
void yolo_postprocess_gpu(
    const __half* d_output,
    int num_anchors,
    float conf_thresh,
    float iou_thresh,
    Detection* d_final,
    cudaStream_t stream,
    Detection* d_dets,
    Detection* d_compacted,
    int* d_mask,
    int* d_count_compact,
    int* d_count_final,
    int* h_count_final,
    cudaEvent_t ev_count_final,
    int num_cameras
) {

    // TODO : pass as parameter
    int max_dets = 64;
    int pre_nms_topk = 256;

    // Reset intermediate detection counter on device
    cudaMemsetAsync(d_count_compact, 0, sizeof(int), stream);
    // Reset final detection counter on device
    cudaMemsetAsync(d_count_final, 0, sizeof(int), stream);

    // TODO : check that downstream we only ready the exact number of detections
    // Reset detections on device
    // cudaMemsetAsync(d_final, 0, max_dets * sizeof(Detection), stream);

    // total anchors across the whole batch
    int tot_anchors = num_cameras * num_anchors;

    dim3 threads(256);
    dim3 blocks((tot_anchors + threads.x - 1) / threads.x);
    // TODO : test without shared memory and see if smoother
    // size_t shared_mem_size = threads.x * sizeof(int);

    // Filtering kernel
    // extract_detections_kernel_batched_half<<<blocks, threads, shared_mem_size, stream>>>(
    extract_detections_kernel_batched_half<<<blocks, threads, 0, stream>>>(
        d_output,        
        num_cameras,              
        num_anchors,     
        conf_thresh,
        d_dets,          
        d_mask           
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "extract_detections_kernel_half failed: "
                  << cudaGetErrorString(err) << std::endl;
    }

    // Compaction kernel
    compact_detections_kernel_batched<<<blocks, threads, 0, stream>>>(
        d_dets,               
        d_mask,               
        d_compacted,          
        d_count_compact,      
        num_anchors,          
        tot_anchors
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "compact_detections_kernel failed: "
                  << cudaGetErrorString(err) << std::endl;
    }

    dim3 blocks_nms((pre_nms_topk + threads.x - 1) / threads.x);

    // Apply NMS kernel
    nms_kernel_final_output_batched<<<blocks_nms, threads, 0, stream>>>(
        d_compacted,        
        d_count_compact,     
        iou_thresh,
        d_final,             
        d_count_final,       
        num_anchors,    // TODO : now is unused inside kernel, remove      
        max_dets,
        num_cameras
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "nms_kernel_final_output failed: "
                  << cudaGetErrorString(err) << std::endl;
    }

    // Copy back final detection count
    cudaMemcpyAsync(h_count_final, d_count_final, sizeof(int), cudaMemcpyDeviceToHost, stream);

    // Mark exactly when the host may read *h_count_final

    cudaEventRecord(ev_count_final, stream);
    
    return;
}