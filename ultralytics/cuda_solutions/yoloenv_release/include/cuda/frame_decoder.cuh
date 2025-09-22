#pragma once
// Header-only, host-side API with pluggable CPU/GPU backends.

#include <cstdint>
#include <cstddef>
#include <iostream>
#include <opencv2/core.hpp>

#include "frame_decoder_config.h"

#if defined(__linux__)
  #include <linux/videodev2.h>
#endif

// Heavier OpenCV bits only when enabled for the build
#ifdef HAVE_OPENCV
  #include <opencv2/imgcodecs.hpp> // imdecode
  #include <opencv2/imgproc.hpp>   // cvtColor
#endif

// Cuda decode bits only when enabled for the build
#if ENABLE_CUDA_DECODE
  #include <cuda_runtime.h>
  #include "cuda_frame_decode/cuda_mjpeg_decode.cuh"
#endif


enum class DecodeMode {
    Auto, MJPEG, YUYV, NV12, RGB24, BGR24, GRAY8
};

enum class DecodeBackend {
    CPU_OpenCV, CUDA_NVJPEG, CUDA_CUSTOM
};

struct DecodeParams {
    uint32_t  pixfmt = 0;
    int       width  = 0;
    int       height = 0;
    DecodeMode mode  = DecodeMode::Auto;
    bool      to_bgr = true;
};

// Map FOURCC → DecodeMode when mode==Auto
inline DecodeMode fd_modeFromPixfmt(uint32_t fourcc) {
#if defined(__linux__)
    switch (fourcc) {
    #ifdef V4L2_PIX_FMT_MJPEG
        case V4L2_PIX_FMT_MJPEG:
    #endif
    #ifdef V4L2_PIX_FMT_JPEG
        case V4L2_PIX_FMT_JPEG:
    #endif
            return DecodeMode::MJPEG;
    #ifdef V4L2_PIX_FMT_YUYV
        case V4L2_PIX_FMT_YUYV:  return DecodeMode::YUYV;
    #endif
    #ifdef V4L2_PIX_FMT_NV12
        case V4L2_PIX_FMT_NV12:  return DecodeMode::NV12;
    #endif
    #ifdef V4L2_PIX_FMT_RGB24
        case V4L2_PIX_FMT_RGB24: return DecodeMode::RGB24;
    #endif
    #ifdef V4L2_PIX_FMT_BGR24
        case V4L2_PIX_FMT_BGR24: return DecodeMode::BGR24;
    #endif
    #ifdef V4L2_PIX_FMT_GREY
        case V4L2_PIX_FMT_GREY:  return DecodeMode::GRAY8;
    #endif
        default: return DecodeMode::Auto;
    }
#else
    (void)fourcc;
    return DecodeMode::Auto;
#endif
}

class FrameDecoder {
public:
    FrameDecoder(const DecodeParams& p,
                 DecodeBackend backend = DecodeBackend::CPU_OpenCV)
        : p_(p), backend_(backend) {}

    // To free any GPU resources if used
    ~FrameDecoder();


    void updateParams(const DecodeParams& p) { p_ = p; }
    const DecodeParams& params() const { return p_; }
    void setBackend(DecodeBackend b) { backend_ = b; }
    DecodeBackend backend() const { return backend_; }

    // Decode input buffer → BGR cv::Mat (unless p_.to_bgr=false for RGB/BGR/GRAY)
    cv::Mat decode(const void* data, size_t len);

private:
    DecodeParams  p_;
    DecodeBackend backend_;

#ifdef HAVE_OPENCV
    cv::Mat tmp_in_;
    cv::Mat dst_;
#endif

#if ENABLE_CUDA_DECODE
    cudaStream_t stream_ = nullptr;
    uint8_t*     d_bgr_  = nullptr;
    size_t       d_pitch_ = 0;
#endif

};



// ----------------------------------- DESTRUCTOR ----------------------------------- //

inline FrameDecoder::~FrameDecoder() {
#if ENABLE_CUDA_DECODE
    if (d_bgr_)  { cudaFree(d_bgr_);  d_bgr_ = nullptr; }
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
#endif
}



// ---------------------------------- FRAME DECODER ---------------------------------- //

inline cv::Mat FrameDecoder::decode(const void* data, size_t len) {
    if (!data || len == 0) {
        std::cerr << "FrameDecoder: empty input\n";
        return {};
    }

    DecodeMode mode = (p_.mode == DecodeMode::Auto) ? fd_modeFromPixfmt(p_.pixfmt) : p_.mode;
    if (mode == DecodeMode::Auto) {
        std::cerr << "FrameDecoder: unknown pixfmt with mode=Auto; set mode explicitly.\n";
        return {};
    }

    switch (backend_) {
        case DecodeBackend::CPU_OpenCV: {
        #ifndef HAVE_OPENCV
            std::cerr << "FrameDecoder: CPU_OpenCV backend selected, but HAVE_OPENCV is not defined.\n";
            return {};
        #else
            switch (mode) {
                case DecodeMode::MJPEG: {
                    cv::Mat bitstream(1, static_cast<int>(len), CV_8UC1, const_cast<void*>(data));
                    cv::Mat img = cv::imdecode(bitstream, cv::IMREAD_COLOR); // BGR
                    if (img.empty()) std::cerr << "FrameDecoder: imdecode(MJPEG) failed\n";
                    return img;
                }
                case DecodeMode::YUYV: {
                    if (p_.width <= 0 || p_.height <= 0) { std::cerr << "FrameDecoder: invalid YUYV dims\n"; return {}; }
                    if (len < static_cast<size_t>(p_.width)*p_.height*2) { std::cerr << "FrameDecoder: YUYV buffer too small\n"; return {}; }
                    tmp_in_ = cv::Mat(p_.height, p_.width, CV_8UC2, const_cast<void*>(data));
                    cv::cvtColor(tmp_in_, dst_, cv::COLOR_YUV2BGR_YUY2);
                    return dst_;
                }
                case DecodeMode::NV12: {
                    if (p_.width <= 0 || p_.height <= 0) { std::cerr << "FrameDecoder: invalid NV12 dims\n"; return {}; }
                    size_t need = static_cast<size_t>(p_.width)*p_.height*3/2;
                    if (len < need) { std::cerr << "FrameDecoder: NV12 buffer too small\n"; return {}; }
                    tmp_in_ = cv::Mat(p_.height + p_.height/2, p_.width, CV_8UC1, const_cast<void*>(data));
                    cv::cvtColor(tmp_in_, dst_, cv::COLOR_YUV2BGR_NV12);
                    return dst_;
                }
                case DecodeMode::RGB24: {
                    if (p_.width <= 0 || p_.height <= 0) { std::cerr << "FrameDecoder: invalid RGB24 dims\n"; return {}; }
                    size_t need = static_cast<size_t>(p_.width)*p_.height*3;
                    if (len < need) { std::cerr << "FrameDecoder: RGB24 buffer too small\n"; return {}; }
                    tmp_in_ = cv::Mat(p_.height, p_.width, CV_8UC3, const_cast<void*>(data));
                    if (!p_.to_bgr) return tmp_in_.clone();
                    cv::cvtColor(tmp_in_, dst_, cv::COLOR_RGB2BGR);
                    return dst_;
                }
                case DecodeMode::BGR24: {
                    if (p_.width <= 0 || p_.height <= 0) { std::cerr << "FrameDecoder: invalid BGR24 dims\n"; return {}; }
                    size_t need = static_cast<size_t>(p_.width)*p_.height*3;
                    if (len < need) { std::cerr << "FrameDecoder: BGR24 buffer too small\n"; return {}; }
                    tmp_in_ = cv::Mat(p_.height, p_.width, CV_8UC3, const_cast<void*>(data));
                    if (!p_.to_bgr) return tmp_in_.clone();
                    return tmp_in_.clone(); // already BGR; detach from mmap buffer
                }
                case DecodeMode::GRAY8: {
                    if (p_.width <= 0 || p_.height <= 0) { std::cerr << "FrameDecoder: invalid GRAY8 dims\n"; return {}; }
                    size_t need = static_cast<size_t>(p_.width)*p_.height;
                    if (len < need) { std::cerr << "FrameDecoder: GRAY8 buffer too small\n"; return {}; }
                    tmp_in_ = cv::Mat(p_.height, p_.width, CV_8UC1, const_cast<void*>(data));
                    if (!p_.to_bgr) return tmp_in_.clone();
                    cv::cvtColor(tmp_in_, dst_, cv::COLOR_GRAY2BGR);
                    return dst_;
                }
            }
            std::cerr << "FrameDecoder: unsupported mode (CPU_OpenCV)\n";
            return {};
        #endif
        }

        case DecodeBackend::CUDA_NVJPEG:
        #if !ENABLE_CUDA_DECODE
            std::cerr << "FrameDecoder: CUDA_NVJPEG selected but not built in.\n";
            return {};
        #else
            std::cerr << "FrameDecoder: CUDA_NVJPEG backend not implemented yet.\n";
            return {};
        #endif

        case DecodeBackend::CUDA_CUSTOM:
        #if !ENABLE_CUDA_DECODE
            std::cerr << "FrameDecoder: CUDA_CUSTOM selected but USE_CUDA_DECODE not defined.\n";
            return {};
        #else
            // --- For now, handle MJPEG only; add raw formats later ---
            if (mode != DecodeMode::MJPEG) {
                std::cerr << "FrameDecoder: CUDA_CUSTOM currently supports only MJPEG.\n";
                return {};
            }

            // Lazily create a non-blocking stream
            if (!stream_) {
                cudaError_t e = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
                if (e != cudaSuccess) {
                    std::cerr << "FrameDecoder: cudaStreamCreate failed: " << cudaGetErrorString(e) << "\n";
                    return {};
                }
            }

            // Call the (currently stubbed) GPU decode
            cudaError_t dec = cuda_mjpeg_decode_to_bgr_device(
                reinterpret_cast<const uint8_t*>(data), len,
                p_.width, p_.height,
                &d_bgr_, &d_pitch_,
                stream_
            );

            if (dec != cudaSuccess) {
                std::cerr << "FrameDecoder: cuda_mjpeg_decode_to_bgr_device failed ("
                        << cudaGetErrorString(dec) << "). Falling back to CPU.\n";
            #ifdef HAVE_OPENCV
                // Fallback: CPU imdecode
                cv::Mat bitstream(1, static_cast<int>(len), CV_8UC1, const_cast<void*>(data));
                cv::Mat img = cv::imdecode(bitstream, cv::IMREAD_COLOR); // BGR
                if (img.empty()) std::cerr << "FrameDecoder: CPU imdecode fallback failed\n";
                return img;
            #else
                return {};
            #endif
            }

            // Create the host output Mat and copy device → host
            cv::Mat out(p_.height, p_.width, CV_8UC3);
            cudaError_t m = cudaMemcpy2DAsync(
                out.data, out.step,         // dst ptr & pitch
                d_bgr_, d_pitch_,           // src ptr & pitch
                static_cast<size_t>(p_.width) * 3,  // row bytes
                static_cast<size_t>(p_.height),
                cudaMemcpyDeviceToHost,
                stream_
            );
            if (m != cudaSuccess) {
                std::cerr << "FrameDecoder: cudaMemcpy2DAsync failed: " << cudaGetErrorString(m) << "\n";
                return {};
            }

            // Ensure the pixels are ready before returning the Mat
            cudaStreamSynchronize(stream_);
            return out;
        #endif

    }
    std::cerr << "FrameDecoder: unknown backend\n";
    return {};

}
