// src/cuda/cuda_threads_capture.cu

#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
#include <mutex>
#include <algorithm>
#include <iostream>

#include <nvtx3/nvToolsExt.h>
#include <opencv2/opencv.hpp>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include "nvbufsurface.h"

#include "kernels/cuda_threads.cuh"
#include "cuda_undistort.cuh"
#include "cuda_nvmm_egl_upload.cuh"

// ------------------------------ GLOBALS DEFINITION ------------------------------ //
cv::Mat frame_back;
cv::Mat frame_front;

std::mutex frame_mutex;
std::mutex frame_copy_mutex;

std::condition_variable frame_ready;
std::atomic<bool> new_frame_available(false);
std::atomic<bool> keep_running(true);


// ------------------------------ FRAME CAPTURE THREAD ------------------------------ //

void frame_capture_thread(
    GstElement*   sink,
    int           cap_width,
    int           cap_height,
    int           num_cameras,
    unsigned char* d_bgr_raw,
    unsigned char* d_bgr_undistorted,
    cudaStream_t  gst_stream,
    cudaEvent_t   ev_frame_ready
) {
    
    nvtxRangePush("FrameCaptureThread_GST_NVMM");

    using clock = std::chrono::steady_clock;
    auto last_print   = clock::now();
    auto last_capture = clock::now();

    // Attach CUDA context for this thread
    int dev = 0;
    cudaError_t cerr = cudaSetDevice(dev);
    if (cerr != cudaSuccess) {
        std::cerr << "[GST] cudaSetDevice(" << dev << ") failed in capture thread: "
                  << cudaGetErrorString(cerr) << "\n";
        nvtxRangePop();
        return;
    }

    int    frames        = 0;
    double sum_period_ms = 0.0;   // for source FPS

    // TODO: move these to a config struct / file
    const float fish_fov_deg = 200.0f;   // lens FOV
    const float out_hfov_deg = 90.0f;    // main zoom knob
    const float cx_f = 959.50f;
    const float cy_f = 539.50f;
    const float r_f  = 1100.77f;

    std::cout << "Starting NVMM capture loop (GStreamer + CUDA)...\n";

    while (keep_running) {

        auto t0 = clock::now();

        // --- Pull next sample from appsink (blocking) ---
        GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
        if (!sample) {
            std::cerr << "[GST] gst_app_sink_pull_sample() failed or EOS\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        GstBuffer* buffer = gst_sample_get_buffer(sample);
        if (!buffer) {
            std::cerr << "[GST] sample has no buffer\n";
            gst_sample_unref(sample);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        GstMapInfo map;
        if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            std::cerr << "[GST] Failed to map buffer\n";
            gst_sample_unref(sample);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // map.data is NvBufSurface* because memory:NVMM
        NvBufSurface* surf = reinterpret_cast<NvBufSurface*>(map.data);

        // --- NVMM RGBA -> device BGR (d_bgr_raw) ---
        if (!upload_nvmm_rgba_to_d_bgr(
                surf, 0,
                d_bgr_raw,
                cap_width,
                cap_height,
                gst_stream))
        {
            std::cerr << "[GST] upload_nvmm_rgba_to_d_bgr() failed\n";
            gst_buffer_unmap(buffer, &map);
            gst_sample_unref(sample);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);

        // --- Fisheye rectification on GPU: d_bgr_raw -> d_bgr_undistorted ---
        undistort_bgr_fisheye_gpu_batched(
            d_bgr_raw,            // distorted
            d_bgr_undistorted,    // undistorted
            cap_width,
            cap_height,
            num_cameras,
            fish_fov_deg, out_hfov_deg,
            cx_f, cy_f,
            r_f,
            gst_stream
        );

        // Signal on GPU that d_bgr_undistorted is ready
        cudaEventRecord(ev_frame_ready, gst_stream);

        // Hand off the "frame ready" info to the inference thread
        {
            std::lock_guard<std::mutex> lk(frame_mutex);
            new_frame_available = true;
        }
        frame_ready.notify_one();

        // Stats: source + processed FPS
        double period_ms = std::chrono::duration<double, std::milli>(t0 - last_capture).count();
        last_capture = t0;

        frames++;
        if (frames > 1) {
            sum_period_ms += period_ms;  // skip first period
        }

        auto now = clock::now();
        double elapsed = std::chrono::duration<double>(now - last_print).count();
        if (elapsed >= 5.0 && frames > 1) {
            double src_fps  = 1000.0 / (sum_period_ms / (frames - 1));
            double proc_fps = frames / elapsed;

            std::cout << "\r[GST] Source FPS: " << src_fps
                      << " | Processed FPS: " << proc_fps
                      << "      " << std::flush;

            last_print    = now;
            frames        = 0;
            sum_period_ms = 0.0;
        }
    }

    std::cout << std::endl;
    nvtxRangePop(); // FrameCaptureThread_GST_NVMM
}
