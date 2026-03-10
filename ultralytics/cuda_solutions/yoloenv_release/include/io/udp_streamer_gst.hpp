#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

class GstUdpStreamer {
public:
    GstUdpStreamer();
    ~GstUdpStreamer();

    // format: BGR, 8-bit, packed
    bool start(int width, int height, int fps,
               const std::string& host, int port, int bitrate_bps);

    void push_bgr(const uint8_t* data, size_t bytes); // bytes = W*H*3
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};