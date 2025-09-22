#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

#include "cuda_detection_struct.h"
#include "cuda_action_preprocess.cuh"
#include "cuda_kernels.cuh"

// Preprocess for the action classifier engine
void action_cls_preprocess_gpu_staged_batched(
    const unsigned char* d_frame_bgr,   
    int                  frameW, int frameH,
    const Detection*     d_boxes,
    float                scaleX, float scaleY,
    float                pad_ratio,
    unsigned char*       d_scratch_crop,      
    int                  maxCropW, int maxCropH,
    unsigned char*       d_scratch_square,    
    int                  maxSquare,
    unsigned char*       d_scratch_bgr96,
    float*               d_gray96_out_base,   
    int                  dstW, int dstH,
    ClsDevParams*        d_params_array,
    int                  n_dets,
    cudaStream_t         stream
){
    // Just for safety but should never activate
    if (n_dets <= 0) return;

    cudaError_t err;

    // 1) Compute per-detection params (one z-slice per detection)
    dim3 block(1,1,1);
    dim3 grid(1,1,n_dets);
    cls_compute_params_kernel_batched<<<grid, block, 0, stream>>>(
        d_boxes,
        frameW, frameH,
        scaleX, scaleY,
        pad_ratio,
        dstW, dstH,
        maxCropW, maxCropH, maxSquare,
        d_params_array,
        n_dets
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "cls_compute_params_kernel_batched failed: "
                  << cudaGetErrorString(err) << std::endl;
    }

    // 2) Crop per detection
    dim3 blockXY(16,16,1);
    dim3 gridCrop( (maxCropW + blockXY.x - 1)/blockXY.x,
                   (maxCropH + blockXY.y - 1)/blockXY.y,
                    n_dets );
    crop_img_kernel_batched<<<gridCrop, blockXY, 0, stream>>>(
        d_frame_bgr, frameW, frameH,
        d_params_array,
        d_scratch_crop,
        maxCropW,
        maxCropH,
        n_dets
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "crop_img_kernel_batched failed: "
                  << cudaGetErrorString(err) << std::endl;
    }

    // 3) Pad to square per detection
    dim3 gridPad( (maxSquare + blockXY.x - 1)/blockXY.x,
                  (maxSquare + blockXY.y - 1)/blockXY.y,
                   n_dets );
    pad_to_square_kernel_batched<<<gridPad, blockXY, 0, stream>>>(
        d_scratch_crop,
        d_params_array,
        d_scratch_square,
        maxCropW,
        maxCropH,
        maxSquare,
        n_dets
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "pad_to_square_kernel_batched failed: "
                  << cudaGetErrorString(err) << std::endl;
    }

    // 4) Resize to 96x96 per detection (BGR interleaved scratch)
    dim3 gridResize( (dstW + blockXY.x - 1)/blockXY.x,
                     (dstH + blockXY.y - 1)/blockXY.y,
                      n_dets );
    resize_bilinear_kernel_from_params_batched<<<gridResize, blockXY, 0, stream>>>(
        d_scratch_bgr96,
        d_scratch_square,
        dstW, dstH,
        d_params_array,
        maxSquare,
        n_dets
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "resize_bilinear_kernel_from_params_batched failed: "
                  << cudaGetErrorString(err) << std::endl;
    }

    // 5) BGR -> Gray + normalize per detection; write into batched TRT input
    bgr_to_gray_norm_kernel_batched<<<gridResize, blockXY, 0, stream>>>(
        d_gray96_out_base,
        d_scratch_bgr96,
        dstW, dstH,
        n_dets
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "bgr_to_gray_norm_kernel_batched failed: "
                  << cudaGetErrorString(err) << std::endl;
    }
}