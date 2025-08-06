#include <cuda_runtime.h>
#include <iostream>
#include "cuda_kernels.cuh"     
#include "cuda_detection_struct.h"   
#include "cuda_preprocess.cuh"

// This function runs the pre-processing on the GPU
// It converts the input image from BGR to float and normalizes it
void run_preprocess_gpu(
    float* d_output_img,
    uchar* d_pinned_input,
    uchar* d_cap_frame,
    uchar* d_resized_frame,
    int out_img_width,
    int out_img_height,
    int in_cap_width,
    int in_cap_height,
    float scale_x,
    float scale_y,
    size_t pinned_size,
    cudaStream_t stream
) {
    // Copy the resized frame to GPU
    cudaMemcpyAsync(d_cap_frame, d_pinned_input, pinned_size, cudaMemcpyHostToDevice, stream);

    dim3 block(16, 16);
    dim3 grid((out_img_width + block.x - 1) / block.x, (out_img_height + block.y - 1) / block.y);

    // Resizing kernel
    resize_bilinear_kernel<<<grid, block, 0, stream>>>(
        d_resized_frame, d_cap_frame,
        in_cap_width, in_cap_height,
        out_img_width, out_img_height,
        scale_x, scale_y
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "resize_bilinear_kernel failed: " << cudaGetErrorString(err) << std::endl;
    }

    // Preprocess kernel 
    preprocess_kernel<<<grid, block, 0, stream>>>(
        d_output_img, d_resized_frame,
        out_img_width, out_img_height
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "preprocess_kernel failed: " << cudaGetErrorString(err) << std::endl;
    }
}
