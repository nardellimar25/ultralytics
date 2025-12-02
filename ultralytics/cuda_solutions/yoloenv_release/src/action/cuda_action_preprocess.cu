#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

#include "cuda_action_preprocess.cuh"
#include "kernels/cuda_kernels.cuh"
#include "cuda_structs.cuh"

// Preprocess for the action classifier engine trying to mimic the EI example
void action_cls_preprocess_gpu_staged_batched_EI_copycat(
    const unsigned char* d_frame_bgr,           
    int                  frameW, int frameH,
    const Detection*     d_boxes,              
    float                scaleX, float scaleY,
    float                pad_ratio,
    unsigned char*       d_scratch_crop,       
    int                  maxCropW, int maxCropH,
    unsigned char*       d_scratch_fitted,      
    int                  maxSquare,          
    unsigned char*       d_scratch_bgr96,      
    float*               d_gray96_out_base,   
    int                  dstW, int dstH,      
    ClsDevParams*        d_params_array,      
    int                  n_dets,
    cudaStream_t         stream
){
    if (n_dets <= 0) return;

    cudaError_t err;
    dim3 blockXY(16,16,1);

    // 1) Compute per-detection base params (bbox on original frame, scale to crop, etc.)
    {
        dim3 block(1,1,1);
        dim3 grid(1,1,n_dets);
        cls_compute_params_kernel_batched<<<grid, block, 0, stream>>>(
            d_boxes,
            frameW, frameH,
            scaleX, scaleY,
            pad_ratio,
            /*dstW, dstH kept for now*/ dstW, dstH,
            maxCropW, maxCropH, maxSquare,
            d_params_array,
            n_dets
        );
        err = cudaGetLastError();
        if (err != cudaSuccess)
            std::cerr << "cls_compute_params_kernel_batched failed: "
                      << cudaGetErrorString(err) << std::endl;
    }

    // 2) Crop the bbox region from the original frame into d_scratch_crop
    {
        dim3 gridCrop( (maxCropW + blockXY.x - 1)/blockXY.x,
                       (maxCropH + blockXY.y - 1)/blockXY.y,
                       n_dets );
        crop_img_kernel_batched<<<gridCrop, blockXY, 0, stream>>>(
            d_frame_bgr, frameW, frameH,
            d_params_array,
            d_scratch_crop,
            maxCropW, maxCropH,
            n_dets
        );
        err = cudaGetLastError();
        if (err != cudaSuccess)
            std::cerr << "crop_img_kernel_batched failed: "
                      << cudaGetErrorString(err) << std::endl;
    }

    // 3) Resize (keep aspect), fitting the LONGEST side to dst(=96).
    //    Writes the fitted image into d_scratch_square per detection.
    //    Only the rectangle [0..tgtW)×[0..tgtH) is valid (tgtW/tgtH in d_params_array).
    {
        dim3 blockRB(16,16,1);
        dim3 gridRB( (96 + blockRB.x - 1)/blockRB.x,
                     (96 + blockRB.y - 1)/blockRB.y,
                     n_dets );

        resize_bilinear_rect_to_rect_batched<<<gridRB, blockRB, 0, stream>>>(
            d_scratch_fitted,   
            d_scratch_crop,     
            d_params_array,
            maxCropW, maxCropH,
            96, 96,
            n_dets
        );
        err = cudaGetLastError();
        if (err != cudaSuccess)
            std::cerr << "resize_fit_longest_side_kernel_batched failed: "
                      << cudaGetErrorString(err) << std::endl;
    }

    // 4) Pad to centered 96x96 square (symmetric), producing final 96x96 BGR
    //    Reads fitted sub-rect (tgtW,tgtH) from params; centers it with padL/padT.
    {
        dim3 blockC2S(16,16,1);
        dim3 gridC2S( (96 + blockC2S.x - 1)/blockC2S.x,
                        (96 + blockC2S.y - 1)/blockC2S.y,
                        n_dets );

        pad_center_to_square_kernel_batched<<<gridC2S, blockC2S, 0, stream>>>(
            /*src  */ d_scratch_fitted,   // [N, 96, 96, 3] buffer; only new_W×new_H valid
            /*par  */ d_params_array,
            /*dst  */ d_scratch_bgr96,    // [N, 96, 96, 3]
            /*maxW */ 96,                 // maxOutW used for src stride
            /*maxH */ 96,                 // maxOutH used for src stride
            /*dstS */ 96,
            /*N    */ n_dets
        );
        err = cudaGetLastError();
        if (err != cudaSuccess)
            std::cerr << "pad_center_to_square96_kernel_batched failed: "
                      << cudaGetErrorString(err) << std::endl;
    }

    // 5) BGR -> Gray + normalize into batched TRT input (N×1×96×96)
    {
        dim3 gridGray( (dstW + blockXY.x - 1)/blockXY.x,
                       (dstH + blockXY.y - 1)/blockXY.y,
                        n_dets );
        bgr_to_gray_norm_kernel_batched<<<gridGray, blockXY, 0, stream>>>(
            d_gray96_out_base,   // contiguous [n_dets, 96, 96]
            d_scratch_bgr96,     // contiguous [n_dets, 96, 96, 3]
            dstW, dstH,
            n_dets
        );
        err = cudaGetLastError();
        if (err != cudaSuccess)
            std::cerr << "bgr_to_gray_norm_kernel_batched failed: "
                      << cudaGetErrorString(err) << std::endl;
    }
}


/*

// OLD: differs from EI version, but runs faster, with strong engine could work fine
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
*/