#include "udp_streamer_gst.hpp"

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>

#include <cstdio>
#include <cstring>
#include <mutex>

static void drain_bus(GstBus* bus) {
    if (!bus) return;
    while (true) {
        GstMessage* msg = gst_bus_pop(bus);
        if (!msg) break;
        gst_message_unref(msg);
    }
}

struct GstUdpStreamer::Impl {
    GstElement* pipeline = nullptr;
    GstElement* appsrc   = nullptr;
    GstBus* bus          = nullptr;

    int width  = 0;
    int height = 0;
    int fps    = 30;
    uint64_t frame_idx = 0;

    std::mutex mtx;
};

GstUdpStreamer::GstUdpStreamer()
    : impl_(new Impl) {}

GstUdpStreamer::~GstUdpStreamer() {
    stop();
}

bool GstUdpStreamer::start(int width, int height, int fps,
                           const std::string& host, int port, int bitrate_bps)
{
    // locks internally
    stop();

    // Initialize GStreamer once per process.
    static std::once_flag gst_once;
    std::call_once(gst_once, []() { gst_init(nullptr, nullptr); });

    std::lock_guard<std::mutex> lk(impl_->mtx);

    impl_->width  = width;
    impl_->height = height;
    impl_->fps    = (fps > 0) ? fps : 30;
    impl_->frame_idx = 0;

    // MPEG-TS over UDP
    char desc[2048];
    std::snprintf(desc, sizeof(desc),
        "appsrc name=src is-live=true block=false format=time do-timestamp=true "
        "caps=video/x-raw,format=BGRx,width=%d,height=%d,framerate=%d/1 "
        "! queue leaky=downstream max-size-buffers=1 max-size-time=0 max-size-bytes=0 "
        "! videorate drop-only=true "
        "! video/x-raw,framerate=%d/1 "
        "! nvvidconv "
        "! video/x-raw(memory:NVMM),format=NV12 "
        "! nvv4l2h264enc insert-sps-pps=true idrinterval=%d iframeinterval=%d "
        "bitrate=%d preset-level=1 maxperf-enable=1 "
        "! h264parse config-interval=1 "
        "! mpegtsmux "
        "! queue leaky=downstream max-size-buffers=1 max-size-time=0 max-size-bytes=0 "
        "! udpsink host=%s port=%d sync=false async=false",
        impl_->width, impl_->height, impl_->fps,
        impl_->fps,
        impl_->fps, impl_->fps,
        bitrate_bps,
        host.c_str(), port
    );


    GError* err = nullptr;
    impl_->pipeline = gst_parse_launch(desc, &err);
    if (!impl_->pipeline) {
        std::fprintf(stderr, "GStreamer parse failed: %s\n", err ? err->message : "unknown");
        if (err) g_error_free(err);
        return false;
    }
    if (err) g_error_free(err);

    impl_->appsrc = gst_bin_get_by_name(GST_BIN(impl_->pipeline), "src");
    if (!impl_->appsrc) {
        std::fprintf(stderr, "Failed to get appsrc element\n");
        stop();
        return false;
    }

    // Force caps on appsrc (helps negotiation)
    GstCaps* caps = gst_caps_new_simple(
        "video/x-raw",
        "format", G_TYPE_STRING, "BGRx",
        "width",  G_TYPE_INT, impl_->width,
        "height", G_TYPE_INT, impl_->height,
        "framerate", GST_TYPE_FRACTION, impl_->fps, 1,
        nullptr
    );
    gst_app_src_set_caps(GST_APP_SRC(impl_->appsrc), caps);
    gst_caps_unref(caps);

    g_object_set(G_OBJECT(impl_->appsrc),
                 "stream-type", 0,  // GST_APP_STREAM_TYPE_STREAM
                 "format", GST_FORMAT_TIME,
                 "is-live", TRUE,
                 "block", TRUE,
                 "do-timestamp", TRUE,
                 nullptr);

    impl_->bus = gst_element_get_bus(impl_->pipeline);

    GstStateChangeReturn ret = gst_element_set_state(impl_->pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        std::fprintf(stderr, "Failed to set pipeline to PLAYING\n");
        stop();
        return false;
    }

    return true;
}

void GstUdpStreamer::push_bgr(const uint8_t* data, size_t bytes) {
    if (!impl_ || !data || bytes == 0) return;

    std::lock_guard<std::mutex> lk(impl_->mtx);
    if (!impl_->appsrc || !impl_->pipeline) return;

    drain_bus(impl_->bus);

    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, bytes, nullptr);
    if (!buffer) return;

    GstMapInfo map;
    if (gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
        std::memcpy(map.data, data, bytes);
        gst_buffer_unmap(buffer, &map);
    }

    // const GstClockTime dur = gst_util_uint64_scale_int(1, GST_SECOND, impl_->fps);
    GST_BUFFER_PTS(buffer) = GST_CLOCK_TIME_NONE;
    GST_BUFFER_DTS(buffer) = GST_CLOCK_TIME_NONE;
    GST_BUFFER_DURATION(buffer) = GST_CLOCK_TIME_NONE;
    // impl_->frame_idx++;

    GstFlowReturn flow_ret;
    g_signal_emit_by_name(impl_->appsrc, "push-buffer", buffer, &flow_ret);
    gst_buffer_unref(buffer);

    (void)flow_ret;
}

void GstUdpStreamer::stop() {
    if (!impl_) return;

    std::lock_guard<std::mutex> lk(impl_->mtx);

    if (impl_->pipeline) {
        gst_element_set_state(impl_->pipeline, GST_STATE_NULL);
    }
    if (impl_->bus) {
        gst_object_unref(impl_->bus);
        impl_->bus = nullptr;
    }
    if (impl_->appsrc) {
        gst_object_unref(impl_->appsrc);
        impl_->appsrc = nullptr;
    }
    if (impl_->pipeline) {
        gst_object_unref(impl_->pipeline);
        impl_->pipeline = nullptr;
    }

    impl_->frame_idx = 0;
}
