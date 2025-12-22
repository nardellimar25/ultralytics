#include <cuda_runtime.h>
#include <iostream>
#include <cuda_fp16.h>

#include "cuda_yolo_preprocess.cuh"
#include "kernels/cuda_kernels.cuh"     
#include "cuda_structs.cuh"   


// This function runs the yolo pre-processing on the GPU
// It converts the input image from BGR to float and normalizes it
// Assumes host input is also packed contiguously by frame
void yolo_preprocess_gpu_batched(
    float*              d_yolo_out_base,      
    const unsigned char* h_frames_bgr_pinned, 
    unsigned char*       d_cap_batch_u8,     
    unsigned char*       d_resized_batch_u8,  
    int                  in_cap_width,
    int                  in_cap_height,
    int                  out_img_width,
    int                  out_img_height,
    float                scale_x,
    float                scale_y,
    size_t               pinned_size,         
    int                  num_cameras,
    cudaStream_t         stream
){
    if (num_cameras <= 0) return;

    // unused
    (void)h_frames_bgr_pinned;
    (void)pinned_size;

    cudaError_t err;

    // Z-batched over cameras
    dim3 blockXY(16,16,1);
    dim3 gridResize((out_img_width  + blockXY.x - 1)/blockXY.x,
                    (out_img_height + blockXY.y - 1)/blockXY.y,
                    num_cameras);

    // 1) Resize U8 -> U8 into contiguous resized batch
    resize_bilinear_kernel_batched<<<gridResize, blockXY, 0, stream>>>(
        d_resized_batch_u8, d_cap_batch_u8,
        in_cap_width, in_cap_height,
        out_img_width, out_img_height,
        scale_x, scale_y,
        num_cameras
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "resize_bilinear_kernel_batched (contiguous) failed: "
                  << cudaGetErrorString(err) << std::endl;
    }

    // 2) U8 (BGR) -> float (RGB, CHW [0,1]) directly into contiguous [N,3,H,W]
    preprocess_kernel_batched<<<gridResize, blockXY, 0, stream>>>(
        d_yolo_out_base,          
        d_resized_batch_u8,      
        out_img_width, out_img_height,
        num_cameras
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "preprocess_kernel_batched (contiguous) failed: "
                  << cudaGetErrorString(err) << std::endl;
    }
}
// FP16 variant
// void yolo_preprocess_gpu_batched(
//     __half*            d_yolo_out_base_h,    
//     const unsigned char* h_frames_bgr_pinned, 
//     unsigned char*       d_cap_batch_u8,    
//     unsigned char*       d_resized_batch_u8,  
//     int                  in_cap_width,
//     int                  in_cap_height,
//     int                  out_img_width,
//     int                  out_img_height,
//     float                scale_x,
//     float                scale_y,
//     size_t               pinned_size,         
//     int                  num_cameras,
//     cudaStream_t         stream
// ){
//     if (num_cameras <= 0) return;

//     // H2D copy per camera into contiguous device input
//     // for (int i = 0; i < num_cameras; ++i) {
//     //     const size_t off = static_cast<size_t>(i) * pinned_size;
//     //     cudaMemcpyAsync(d_cap_batch_u8 + off,
//     //                     h_frames_bgr_pinned + off,
//     //                     pinned_size, cudaMemcpyHostToDevice, stream);
//     // }

//     // kept in the signature for ABI compatibility, but are no longer used.
//     (void)h_frames_bgr_pinned;
//     (void)pinned_size;

//     cudaError_t err;

//     // Z-batched over cameras
//     dim3 blockXY(16,16,1);
//     dim3 gridResize((out_img_width  + blockXY.x - 1)/blockXY.x,
//                     (out_img_height + blockXY.y - 1)/blockXY.y,
//                     num_cameras);

//     // 1) Resize U8 -> U8 into contiguous resized batch
//     resize_bilinear_kernel_batched<<<gridResize, blockXY, 0, stream>>>(
//         d_resized_batch_u8, d_cap_batch_u8,
//         in_cap_width, in_cap_height,
//         out_img_width, out_img_height,
//         scale_x, scale_y,
//         num_cameras
//     );
//     err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         std::cerr << "resize_bilinear_kernel_batched (contiguous, half path) failed: "
//                   << cudaGetErrorString(err) << std::endl;
//     }

//     // 2) U8 (BGR) -> __half (RGB, CHW [0,1]) directly into contiguous [N,3,H,W]
//     preprocess_kernel_batched_half<<<gridResize, blockXY, 0, stream>>>(
//         d_yolo_out_base_h,       
//         d_resized_batch_u8,      
//         out_img_width, out_img_height,
//         num_cameras
//     );
//     err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         std::cerr << "preprocess_kernel_batched_half (contiguous) failed: "
//                   << cudaGetErrorString(err) << std::endl;
//     }
// }

// Fused resize + preprocess for half precision
void yolo_preprocess_gpu_batched(
    __half*              d_yolo_out_base_h,
    const unsigned char* h_frames_bgr_pinned,
    unsigned char*       d_cap_batch_u8,
    unsigned char*       d_resized_batch_u8,
    int                  in_cap_width,
    int                  in_cap_height,
    int                  out_img_width,
    int                  out_img_height,
    float                scale_x,
    float                scale_y,
    size_t               pinned_size,
    int                  num_cameras,
    cudaStream_t         stream
){

    if (num_cameras <= 0) return;

    // TODO : remove unused parameters
    (void)h_frames_bgr_pinned;
    (void)pinned_size;
    (void)d_resized_batch_u8; 

    dim3 block(16,16,1);
    dim3 grid(  (out_img_width  + block.x - 1)/block.x, 
                (out_img_height + block.y - 1)/block.y,
                num_cameras );

    resize_preprocess_fused_batched_half<<<grid, block, 0, stream>>>(
        d_yolo_out_base_h,
        d_cap_batch_u8,
        in_cap_width, in_cap_height,
        out_img_width, out_img_height,
        scale_x, scale_y,
        num_cameras
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "resize_preprocess_fused_batched_half failed: "
                  << cudaGetErrorString(err) << "\n";
    }
}

