#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

#include "cuda_action_postprocess.cuh"
#include "kernels/cuda_kernels.cuh"
#include "cuda_structs.cuh"

// Postprocess function for the action classifier
void action_cls_postprocess_gpu_batched(
    const Detection* d_dets,   // [N]
    const float*     d_probs,  // [N,3] FP32 (G,R,Y)
    int              N,
    float            scaleX,   // model->frame scale factors
    float            scaleY,
    ActionVis*       d_out,    // [N] device buffer for results
    cudaStream_t     stream
){
    if (N <= 0) return; //safety but should never activate

    const int BS = 128;
    const dim3 block(BS, 1, 1);
    const dim3 grid((N + BS - 1) / BS, 1, 1);

    final_visual_struct_kernel<<<grid, block, 0, stream>>>(
        d_dets, d_probs, N, 
        scaleX, scaleY, d_out
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "final_visual_struct_kernel failed: "
                  << cudaGetErrorString(err) << std::endl;
    }

}