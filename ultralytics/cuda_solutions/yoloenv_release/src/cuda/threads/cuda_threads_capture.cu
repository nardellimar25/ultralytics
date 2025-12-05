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

#include "threads/cuda_threads.cuh"
#include "cuda_undistort.cuh"
#include "cuda_nvmm_egl_upload.cuh"


// ------------------------------ HELPER ------------------------------ //


// TODO: move to helpers
static void log_gst_pipeline_bus(GstElement* pipeline, int cam)
{
    if (!pipeline) return;

    GstBus* bus = gst_element_get_bus(pipeline);
    if (!bus) {
        std::cerr << "[GST] No bus for pipeline of cam " << cam << "\n";
        return;
    }

    // Pop all pending messages (non-blocking)
    while (GstMessage* msg = gst_bus_pop(bus)) {
        switch (GST_MESSAGE_TYPE(msg)) {
            case GST_MESSAGE_ERROR: {
                GError* err = nullptr;
                gchar* dbg = nullptr;
                gst_message_parse_error(msg, &err, &dbg);
                std::cerr << "[GST] ERROR on cam " << cam << ": "
                          << (err ? err->message : "unknown")
                          << "\n       debug: " << (dbg ? dbg : "none") << "\n";
                if (err) g_error_free(err);
                if (dbg) g_free(dbg);
                break;
            }
            case GST_MESSAGE_EOS:
                std::cerr << "[GST] EOS message on cam " << cam << "\n";
                break;

            default:
                // You can print other message types if you want more verbosity
                break;
        }
        gst_message_unref(msg);
    }

    gst_object_unref(bus);
}


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

        // Pull next sample from appsink (blocking)
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

        // NVMM RGBA -> device BGR (d_bgr_raw)
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

        // Fisheye rectification on GPU: d_bgr_raw -> d_bgr_undistorted
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


// -------------------------- MULTISTREAM FRAME CAPTURE THREAD -------------------------- //

// TODO: maybe don't need pipelines here for the error logging?
void multistream_frame_capture_thread(
    GstElement**  sinks,              // array of appsinks, size = num_cameras
    int           num_cameras,
    int           cap_width,
    int           cap_height,
    unsigned char* d_bgr_raw,        
    unsigned char* d_bgr_undistorted, 
    cudaStream_t  gst_stream,
    cudaEvent_t   ev_frame_ready,
    GstElement**  pipelines           // for debugging the error states
) {
    nvtxRangePush("MultiStreamFrameCaptureThread_GST_NVMM");

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

    // to get fps stats
    int    frames        = 0;
    double sum_period_ms = 0.0;   

    // to track consecutive failures per camera
    std::vector<int> cam_fail_count(num_cameras, 0);
    std::vector<bool> cam_pull_enabled(num_cameras, true);
    std::vector<int> cam_probe_cooldown(num_cameras, true);
    // if fails for a whole second, disable pull for 3 seconds
    const int MAX_CONSEC_FAIL = 30;  
    const int PROBE_COOLDOWN_FRAMES = 90;
    const GstClockTime PROBE_TIMEOUT_NS = 200 * GST_MSECOND; // 200 ms

    // TODO: move these to a config struct / file
    const float fish_fov_deg = 200.0f;   // lens FOV
    const float out_hfov_deg = 90.0f;    // main zoom knob
    const float cx_f = 959.50f;
    const float cy_f = 539.50f;
    const float r_f  = 1100.77f;

    const size_t frame_bytes = static_cast<size_t>(cap_width) * cap_height * 3;

    std::cout << "\nStarting MULTISTREAM NVMM capture loop (GStreamer + CUDA)...\n";

    while (keep_running) {

        auto t0 = clock::now();

        // check if any camera is still active
        // bool any_cam_updated = false; 

        // bool all_ok = true;

        // Pull one frame from EACH camera
        for (int cam = 0; cam < num_cameras; ++cam) {

            GstElement* sink = sinks[cam];
            
            // Destination slice for this camera inside the batched buffer
            unsigned char* dst_cam = d_bgr_raw + static_cast<size_t>(cam) * frame_bytes;

            // ---------------------- DISABLED CAMERA: PROBE MODE ---------------------- //
            
            if (!cam_pull_enabled[cam]) {
            
                // Decrement cooldown; when it hits zero, try a single non-blocking probe
                if (cam_probe_cooldown[cam] > 0) {
                    cam_probe_cooldown[cam]--;
                    continue;
                }

                if (!sink) {
                    // Stay disabled
                    cam_probe_cooldown[cam] = PROBE_COOLDOWN_FRAMES;
                    continue;
                }
                
                std::cerr << "[GST] Probing disabled cam " << cam << "...\n";

                GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(sink), PROBE_TIMEOUT_NS);
            
                if (!sample) {
                    // stay disabled, keep frozen frame
                    std::cerr << "[GST] Probe failed for cam " << cam << " (still dead)\n";
                    cam_probe_cooldown[cam] = PROBE_COOLDOWN_FRAMES;
                    continue;
                }

                // Got a frame on a disabled cam → process it and re-enable
                GstBuffer* buffer = gst_sample_get_buffer(sample);
                if (!buffer) {
                    std::cerr << "[GST] Probe sample has no buffer (cam " << cam << ")\n";
                    gst_sample_unref(sample);
                    cam_probe_cooldown[cam] = PROBE_COOLDOWN_FRAMES;
                    continue;
                }

                GstMapInfo map;
                if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
                    std::cerr << "[GST] Failed to map probe buffer (cam " << cam << ")\n";
                    gst_sample_unref(sample);
                    cam_probe_cooldown[cam] = PROBE_COOLDOWN_FRAMES;
                    continue;
                }

                // map.data is NvBufSurface* because memory:NVMM
                NvBufSurface* surf = reinterpret_cast<NvBufSurface*>(map.data);

                if (!upload_nvmm_rgba_to_d_bgr(surf,0,dst_cam,cap_width,cap_height,gst_stream)) {
                    std::cerr << "[GST] upload_nvmm_rgba_to_d_bgr() failed in probe (cam "
                              << cam << ")\n";
                    gst_buffer_unmap(buffer, &map);
                    gst_sample_unref(sample);
                    cam_probe_cooldown[cam] = PROBE_COOLDOWN_FRAMES;
                    continue;
                }

                // Success: new frame → re-enable this camera
                gst_buffer_unmap(buffer, &map);
                gst_sample_unref(sample);

                cam_fail_count[cam]     = 0;
                cam_pull_enabled[cam]   = true;
                cam_probe_cooldown[cam] = 0;

                std::cerr << "[GST] Cam " << cam << " recovered and re-enabled\n";

                continue;
            }
            
            // ---------------------- ACTIVE CAMERA: NORMAL PULL ---------------------- //

            if (!sink) {
                std::cerr << "[GST] sink[" << cam << "] is nullptr\n";
                cam_fail_count[cam]++;
                if (cam_fail_count[cam] >= MAX_CONSEC_FAIL) {
                    cam_pull_enabled[cam]   = false;
                    cam_probe_cooldown[cam] = PROBE_COOLDOWN_FRAMES;
                    std::cerr << "[GST] DISABLING capture for cam " << cam
                              << " after " << cam_fail_count[cam]
                              << " consecutive failures\n";
                }
                continue;
            }

            GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
            
            if (!sample) {
                std::cerr << "[GST] gst_app_sink_pull_sample() failed or EOS for cam " << cam << "\n";
                cam_fail_count[cam]++;
                GstElement* pipeline = pipelines ? pipelines[cam] : nullptr;
                log_gst_pipeline_bus(pipeline, cam);

                if(cam_fail_count[cam] >= MAX_CONSEC_FAIL) {
                    cam_pull_enabled[cam] = false;
                    std::cerr << "[GST] Disabling pull for cam " << cam << " due to repeated failures\n";
                }

                continue;
            }

            GstBuffer* buffer = gst_sample_get_buffer(sample);
            if (!buffer) {
                std::cerr << "[GST] sample has no buffer (cam " << cam << ")\n";
                gst_sample_unref(sample);
                cam_fail_count[cam]++;
                GstElement* pipeline = pipelines ? pipelines[cam] : nullptr;
                log_gst_pipeline_bus(pipeline, cam);

                if(cam_fail_count[cam] >= MAX_CONSEC_FAIL) {
                    cam_pull_enabled[cam] = false;
                    std::cerr << "[GST] Disabling pull for cam " << cam << " due to repeated failures\n";
                }

                continue;
            }

            GstMapInfo map;
            if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
                std::cerr << "[GST] Failed to map buffer (cam " << cam << ")\n";
                gst_sample_unref(sample);
                cam_fail_count[cam]++;
                GstElement* pipeline = pipelines ? pipelines[cam] : nullptr;
                log_gst_pipeline_bus(pipeline, cam);

                if(cam_fail_count[cam] >= MAX_CONSEC_FAIL) {
                    cam_pull_enabled[cam] = false;
                    std::cerr << "[GST] Disabling pull for cam " << cam << " due to repeated failures\n";
                }

                continue;
            }

            // map.data is NvBufSurface* because memory:NVMM
            NvBufSurface* surf = reinterpret_cast<NvBufSurface*>(map.data);


            // NVMM RGBA -> device BGR (d_bgr_raw slice for this camera)
            if (!upload_nvmm_rgba_to_d_bgr(surf, 0, dst_cam, cap_width, cap_height, gst_stream)) {
                std::cerr << "[GST] upload_nvmm_rgba_to_d_bgr() failed (cam " << cam << ")\n";
                gst_buffer_unmap(buffer, &map);
                gst_sample_unref(sample);
                cam_fail_count[cam]++;
                GstElement* pipeline = pipelines ? pipelines[cam] : nullptr;
                log_gst_pipeline_bus(pipeline, cam);

                if(cam_fail_count[cam] >= MAX_CONSEC_FAIL) {
                    cam_pull_enabled[cam] = false;
                    std::cerr << "[GST] Disabling pull for cam " << cam << " due to repeated failures\n";
                }

                continue;
            }

            gst_buffer_unmap(buffer, &map);
            gst_sample_unref(sample);

                        // Reset fail count on success
            cam_fail_count[cam] = 0;

        }

        // fisheye rectification on GPU: batched across cameras
        undistort_bgr_fisheye_gpu_batched(
            d_bgr_raw,            // distorted (batched)
            d_bgr_undistorted,    // undistorted (batched)
            cap_width,
            cap_height,
            num_cameras,          // batch size
            fish_fov_deg, out_hfov_deg,
            cx_f, cy_f,
            r_f,
            gst_stream
        );

        // Signal on GPU that d_bgr_undistorted (for ALL cameras) is ready
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

            std::cout << "\r[GST MULTI] Source FPS: " << src_fps
                      << " | Processed FPS: " << proc_fps
                      << "      " << std::flush;

            last_print    = now;
            frames        = 0;
            sum_period_ms = 0.0;
        }
    }

    std::cout << std::endl;
    nvtxRangePop(); // MultiStreamFrameCaptureThread_GST_NVMM
}
