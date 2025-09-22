#ifndef CUDA_PREPROCESS_CUH
#define CUDA_PREPROCESS_CUH

#include <cuda_runtime.h>
#include <cstddef>          
#include <cuda_fp16.h>       

#include "cuda_detection_struct.h"

// Runs yolov8s preprocessing on GPU: resize + BGR2RGB + normalize
void yolo_preprocess_gpu(
    float* d_output_img,                        // Output float image for engine input
    const unsigned char* d_pinned_input,        // Host-side pinned memory (BGR input)
    unsigned char* d_cap_frame,                 // Device-side original frame
    unsigned char* d_resized_frame,             // Device-side resized output
    int out_img_width,                          // Output model input width
    int out_img_height,                         // Output model input height
    int in_cap_width,                           // Input webcam width
    int in_cap_height,                          // Input webcam height
    float scale_x,                              // Scaling factor for X
    float scale_y,                              // Scaling factor for Y
    size_t pinned_size,                         // Size of pinned input
    cudaStream_t stream                         // CUDA stream to use
);
// Overloaded function for FP16 output
void yolo_preprocess_gpu(
    __half* d_output_img,      
    const unsigned char* d_pinned_input,
    unsigned char* d_cap_frame,
    unsigned char* d_resized_frame,
    int out_img_width,
    int out_img_height,
    int in_cap_width,
    int in_cap_height,
    float scale_x,
    float scale_y,
    size_t pinned_size,
    cudaStream_t stream
);

// FP32 batched variant
void yolo_preprocess_gpu_batched(
    float*                  d_output_imgs,          // [num_cameras] output pointers (CHW, normalized)
    const unsigned char*    h_frames_bgr_pinned,    // [num_cameras] host pinned BGR frames
    unsigned char*          d_cap_frames,           // [num_cameras] device BGR buffers
    unsigned char*          d_resized_frames,       // [num_cameras] device resized BGR buffers
    int                     out_img_width,
    int                     out_img_height,
    int                     in_cap_width,
    int                     in_cap_height,
    float                   scale_x,
    float                   scale_y,
    size_t                  pinned_size,
    int                     num_cameras,
    cudaStream_t            stream
);
// FP16 batched variant
void yolo_preprocess_gpu_batched(
    __half*                 d_output_imgs,          // [num_cameras] output pointers (CHW, normalized, __half)
    const unsigned char*    h_frames_bgr_pinned,    // [num_cameras] host pinned BGR frames
    unsigned char*          d_cap_frames,           // [num_cameras] device BGR buffers
    unsigned char*          d_resized_frames,       // [num_cameras] device resized BGR buffers
    int                     out_img_width,
    int                     out_img_height,
    int                     in_cap_width,
    int                     in_cap_height,
    float                   scale_x,
    float                   scale_y,
    size_t                  pinned_size,
    int                     num_cameras,
    cudaStream_t            stream
);

#ifndef DEBUG_VIS
#define DEBUG_VIS 1   // set to 0 to compile-out visualization
#endif




#endif // CUDA_PREPROCESS_CUH
