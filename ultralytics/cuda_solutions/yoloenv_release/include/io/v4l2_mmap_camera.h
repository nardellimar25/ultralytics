#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <iostream>
#include <sys/time.h>

#if defined(__linux__)
  #include <linux/videodev2.h>
#else
  // If  building on non-Linux
  #error "V4L2 is Linux-only"
#endif

class V4L2MMapCamera {
public:
    V4L2MMapCamera() = default;
    ~V4L2MMapCamera();

    // Open/close
    bool openDevice(const std::string& dev = "/dev/video0");
    void closeDevice();

    // Print info
    bool printCapabilities() const;
    bool printCropCapabilities() const;
    bool listFormats(bool print = true) const;

    // Configure if driver allows it
    bool setFormat(uint32_t width, uint32_t height, uint32_t pixfmt);
    bool setFrameRate(uint32_t fps);

    // Buffers / streaming
    bool initMMap(uint32_t buffer_count = 5);
    bool start();
    bool stop();

    struct Frame {
        void*    data       = nullptr;
        size_t   bytesused  = 0;
        uint32_t index      = UINT32_MAX;
        timeval  timestamp  = {};
    };

    bool dequeue(Frame& out, bool nonblocking = false);
    bool requeue(const Frame& f);

    // Getters
    int      fd()     const noexcept { return fd_; }
    uint32_t width()  const noexcept { return width_; }
    uint32_t height() const noexcept { return height_; }
    uint32_t pixfmt() const noexcept { return pixfmt_; }
    uint32_t bufferCount() const noexcept { return static_cast<uint32_t>(bufs_.size()); }

    static std::string fourccToString(uint32_t fourcc);

private:
    struct Buffer { void* start = nullptr; size_t length = 0; };
    static int xioctl(int fd, unsigned long req, void* arg);

    int fd_ = -1;
    uint32_t width_ = 0, height_ = 0, pixfmt_ = 0;
    std::vector<Buffer> bufs_;
    mutable v4l2_requestbuffers reqbuf_{};
};
