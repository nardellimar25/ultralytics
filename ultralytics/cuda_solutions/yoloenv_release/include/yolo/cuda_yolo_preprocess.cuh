#ifndef CUDA_PREPROCESS_CUH
#define CUDA_PREPROCESS_CUH

#include <cuda_runtime.h>
#include <cstddef>          
#include <cuda_fp16.h>       

#include "cuda_structs.cuh"

// Runs yolov8s preprocessing on GPU: resize + BGR2RGB + normalize
// FP32 batched variant
void yolo_preprocess_gpu_batched(
    float*                  d_output_imgs,          
    const unsigned char*    h_frames_bgr_pinned,    
    unsigned char*          d_cap_frames,          
    unsigned char*          d_resized_frames,      
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
    __half*                 d_output_imgs,         
    const unsigned char*    h_frames_bgr_pinned,   
    unsigned char*          d_cap_frames,          
    unsigned char*          d_resized_frames,       
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

#endif // CUDA_PREPROCESS_CUH
