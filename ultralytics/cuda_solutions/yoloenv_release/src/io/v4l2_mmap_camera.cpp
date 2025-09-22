#include "v4l2_mmap_camera.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <cerrno>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <cassert>

int V4L2MMapCamera::xioctl(int fd, unsigned long req, void* arg) {
    int r;
    do {
        r = ::ioctl(fd, req, arg);
    } while (r == -1 && errno == EINTR);
    return r;
}

std::string V4L2MMapCamera::fourccToString(uint32_t f) {
    char s[5];
    s[0] = f & 0xFF;
    s[1] = (f >> 8) & 0xFF;
    s[2] = (f >> 16) & 0xFF;
    s[3] = (f >> 24) & 0xFF;
    s[4] = '\0';
    return std::string(s);
}

V4L2MMapCamera::~V4L2MMapCamera() {
    stop();
    closeDevice();
}

bool V4L2MMapCamera::openDevice(const std::string& dev) {
    if (fd_ != -1) closeDevice();
    fd_ = ::open(dev.c_str(), O_RDWR /* | O_NONBLOCK */);
    if (fd_ < 0) {
        std::perror(("open " + dev).c_str());
        return false;
    }
    return true;
}

void V4L2MMapCamera::closeDevice() {
    // unmap all buffers
    for (auto& b : bufs_) {
        if (b.start && b.length) ::munmap(b.start, b.length);
        b.start = nullptr; b.length = 0;
    }
    bufs_.clear();
    if (fd_ != -1) {
        ::close(fd_);
        fd_ = -1;
    }
}

bool V4L2MMapCamera::printCapabilities() const {
    if (fd_ < 0) return false;
    v4l2_capability caps{};
    if (xioctl(fd_, VIDIOC_QUERYCAP, &caps) == -1) {
        std::perror("VIDIOC_QUERYCAP");
        return false;
    }
    std::printf(
        "Driver Caps:\n"
        "  Driver:   \"%s\"\n"
        "  Card:     \"%s\"\n"
        "  Bus:      \"%s\"\n"
        "  Version:  %u.%u.%u\n"
        "  Caps:     0x%08x\n",
        caps.driver, caps.card, caps.bus_info,
        (caps.version >> 16) & 0xFF, (caps.version >> 8) & 0xFF, (caps.version) & 0xFF,
        caps.capabilities
    );
    return true;
}

bool V4L2MMapCamera::printCropCapabilities() const {
    if (fd_ < 0) return false;
    v4l2_cropcap crop{};
    crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd_, VIDIOC_CROPCAP, &crop) == -1) {
        std::perror("VIDIOC_CROPCAP");
        return false;
    }
    std::printf(
        "Cropping:\n"
        "  Bounds:  %dx%d+%d+%d\n"
        "  Default: %dx%d+%d+%d\n"
        "  Aspect:  %d/%d\n",
        crop.bounds.width, crop.bounds.height, crop.bounds.left, crop.bounds.top,
        crop.defrect.width, crop.defrect.height, crop.defrect.left, crop.defrect.top,
        crop.pixelaspect.numerator, crop.pixelaspect.denominator
    );
    return true;
}

bool V4L2MMapCamera::listFormats(bool print) const {
    if (fd_ < 0) return false;
    v4l2_fmtdesc f{};
    f.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    bool any = false;
    while (xioctl(fd_, VIDIOC_ENUM_FMT, &f) == 0) {
        any = true;
        if (print) {
            bool compressed = f.flags & V4L2_FMT_FLAG_COMPRESSED;
            bool emulated   = f.flags & V4L2_FMT_FLAG_EMULATED;
            std::printf(" %c%c  %s  (%s)\n",
                        compressed ? 'C' : ' ',
                        emulated   ? 'E' : ' ',
                        f.description,
                        fourccToString(f.pixelformat).c_str());
        }
        f.index++;
    }
    if (!any && print) std::puts("No formats found.");
    return any;
}

bool V4L2MMapCamera::setFormat(uint32_t width, uint32_t height, uint32_t pixfmt) {
    if (fd_ < 0) return false;
    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width  = width;
    fmt.fmt.pix.height = height;
    fmt.fmt.pix.pixelformat = pixfmt;         // e.g., V4L2_PIX_FMT_MJPEG or V4L2_PIX_FMT_YUYV
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    if (xioctl(fd_, VIDIOC_S_FMT, &fmt) == -1) {
        std::perror("VIDIOC_S_FMT");
        return false;
    }
    // Store what the driver actually set
    width_  = fmt.fmt.pix.width;
    height_ = fmt.fmt.pix.height;
    pixfmt_ = fmt.fmt.pix.pixelformat;

    std::printf("Set format: %ux%u %s (field=%d)\n",
                width_, height_, fourccToString(pixfmt_).c_str(), fmt.fmt.pix.field);
    return true;
}

bool V4L2MMapCamera::setFrameRate(uint32_t fps) {
    if (fd_ < 0) return false;
    v4l2_streamparm sp{};
    sp.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    sp.parm.capture.timeperframe.numerator = 1;
    sp.parm.capture.timeperframe.denominator = fps;
    if (xioctl(fd_, VIDIOC_S_PARM, &sp) == -1) {
        std::perror("VIDIOC_S_PARM");
        // not fatal; many drivers ignore it
        return false;
    }
    return true;
}

bool V4L2MMapCamera::initMMap(uint32_t buffer_count) {
    if (fd_ < 0) return false;

    // Request buffers
    std::memset(&reqbuf_, 0, sizeof(reqbuf_));
    reqbuf_.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    reqbuf_.memory = V4L2_MEMORY_MMAP;
    reqbuf_.count  = buffer_count;

    if (xioctl(fd_, VIDIOC_REQBUFS, &reqbuf_) == -1) {
        if (errno == EINVAL) std::fprintf(stderr, "mmap streaming not supported.\n");
        else std::perror("VIDIOC_REQBUFS");
        return false;
    }
    if (reqbuf_.count < 2) {
        std::fprintf(stderr, "Not enough buffer memory (got %u).\n", reqbuf_.count);
        return false;
    }

    bufs_.resize(reqbuf_.count);

    // Query + mmap each buffer
    for (uint32_t i = 0; i < reqbuf_.count; ++i) {
        v4l2_buffer b{};
        b.type   = reqbuf_.type;
        b.memory = V4L2_MEMORY_MMAP;
        b.index  = i;

        if (xioctl(fd_, VIDIOC_QUERYBUF, &b) == -1) {
            std::perror("VIDIOC_QUERYBUF");
            return false;
        }

        void* start = ::mmap(nullptr, b.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, b.m.offset);
        if (start == MAP_FAILED) {
            std::perror("mmap");
            return false;
        }

        bufs_[i].start  = start;
        bufs_[i].length = b.length;
    }
    return true;
}

bool V4L2MMapCamera::start() {
    if (fd_ < 0 || bufs_.empty()) return false;

    // Queue all buffers
    for (uint32_t i = 0; i < bufs_.size(); ++i) {
        v4l2_buffer b{};
        b.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        b.memory = V4L2_MEMORY_MMAP;
        b.index  = i;
        if (xioctl(fd_, VIDIOC_QBUF, &b) == -1) {
            std::perror("VIDIOC_QBUF");
            return false;
        }
    }

    // Stream on
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd_, VIDIOC_STREAMON, &type) == -1) {
        std::perror("VIDIOC_STREAMON");
        return false;
    }
    return true;
}

bool V4L2MMapCamera::stop() {
    if (fd_ < 0) return false;
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd_, VIDIOC_STREAMOFF, &type) == -1) {
        if (errno != EINVAL) std::perror("VIDIOC_STREAMOFF");
        // still proceed
    }
    return true;
}

bool V4L2MMapCamera::dequeue(Frame& out, bool nonblocking) {
    if (fd_ < 0) return false;

    v4l2_buffer b{};
    b.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    b.memory = V4L2_MEMORY_MMAP;

    if (nonblocking) {
        // set fd nonblocking temporarily if you want, or just rely on EAGAIN from driver
    }

    if (xioctl(fd_, VIDIOC_DQBUF, &b) == -1) {
        if (errno == EAGAIN) return false;
        std::perror("VIDIOC_DQBUF");
        return false;
    }

    assert(b.index < bufs_.size());
    out.data      = bufs_[b.index].start;
    out.bytesused = b.bytesused;
    out.index     = b.index;
    out.timestamp = b.timestamp;
    return true;
}

bool V4L2MMapCamera::requeue(const Frame& f) {
    if (fd_ < 0 || f.index >= bufs_.size()) return false;
    v4l2_buffer b{};
    b.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    b.memory = V4L2_MEMORY_MMAP;
    b.index  = f.index;
    if (xioctl(fd_, VIDIOC_QBUF, &b) == -1) {
        std::perror("VIDIOC_QBUF");
        return false;
    }
    return true;
}
