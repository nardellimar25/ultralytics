#ifndef CUDA_PREPROCESS_CUH
#define CUDA_PREPROCESS_CUH

#include <cuda_runtime.h>
#include <cstddef>          // for size_t
#include <opencv2/core.hpp> // only if needed by other includes

// Main detection struct
#include "cuda_detection_struct.h"

// Runs preprocessing on GPU: resize + BGR2RGB + normalize
void run_preprocess_gpu(
    float* d_output_img,        // Output float image for engine input
    uchar* d_pinned_input,      // Host-side pinned memory (BGR input)
    uchar* d_cap_frame,         // Device-side original frame
    uchar* d_resized_frame,     // Device-side resized output
    int out_img_width,          // Output model input width
    int out_img_height,         // Output model input height
    int in_cap_width,           // Input webcam width
    int in_cap_height,          // Input webcam height
    float scale_x,              // Scaling factor for X
    float scale_y,              // Scaling factor for Y
    size_t pinned_size,         // Size of pinned input
    cudaStream_t stream         // CUDA stream to use
);

#endif // CUDA_PREPROCESS_CUH
