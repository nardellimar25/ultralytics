#pragma once
#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

// Decode a JPEG (MJPEG frame) to a device BGR8 image.
// On first call, allocate or resize d_bgr / d_pitch as needed (inside the function).
// Returns cudaSuccess on success, or an error code.
cudaError_t cuda_mjpeg_decode_to_bgr_device(const uint8_t* jpeg_data, size_t jpeg_len,
                                            int width, int height,
                                            uint8_t** d_bgr /*out*/, size_t* d_pitch /*out*/,
                                            cudaStream_t stream);

// Optional: release any decoder-wide static resources (if you create them later).
void cuda_mjpeg_decoder_shutdown();
