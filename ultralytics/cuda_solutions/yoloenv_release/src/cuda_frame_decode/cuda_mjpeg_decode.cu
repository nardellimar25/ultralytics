#include "cuda_mjpeg_decode.cuh"

cudaError_t cuda_mjpeg_decode_to_bgr_device(const uint8_t* /*jpeg_data*/, size_t /*jpeg_len*/,
                                            int /*width*/, int /*height*/,
                                            uint8_t** /*d_bgr*/, size_t* /*d_pitch*/,
                                            cudaStream_t /*stream*/)
{
    // TODO: implement nvJPEG/custom kernel later
    return cudaErrorNotSupported;
}

void cuda_mjpeg_decoder_shutdown() {
    // TODO: free any persistent decoder state if you add it later
}
