#include <chrono>
#include <nvtx3/nvToolsExt.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include <vector>
#include <mutex>
#include <algorithm>

#include "cuda_yolo_preprocess.cuh"
#include "cuda_action_preprocess.cuh"
#include "cuda_yolo_postprocess.cuh"
#include "cuda_action_postprocess.cuh"
#include "cuda_threads.cuh"
#include "cuda_detection_struct.h"
#include "cuda_kernels.cuh"
#include "v4l2_mmap_camera.h"
#include "frame_decoder.cuh"
#include "yolodetect.h"
#include "engine_debug_utils.h"
#include "engine_io.hpp"

// ---- define the shared globals declared in the header ----
cv::Mat frame_back;
cv::Mat frame_front;

std::mutex frame_mutex;
std::mutex frame_copy_mutex;

std::condition_variable frame_ready;
std::atomic<bool> new_frame_available(false);
std::atomic<bool> keep_running(true);


// TEMP: global variables and kernels for NVMM EGL -> CUDA BGR copy
// will be moved to kernel files later

// TEMP: redefinition of checkCu
static void checkCu(CUresult r, const char* msg)
{
    if (r != CUDA_SUCCESS)
    {
        const char* errStr = nullptr;
        cuGetErrorString(r, &errStr);
        std::cerr << "[CUDA-EGL] " << msg << " failed: "
                  << (errStr ? errStr : "unknown") << " (" << r << ")\n";
    }
}

// Very simple RGBA -> BGR copy (no resize yet, 1:1 copy)
__global__ void rgba_to_bgr_linear_kernel(
    const unsigned char* __restrict__ src,
    int srcPitch,        // in bytes
    int width,
    int height,
    unsigned char* __restrict__ dst   // tightly packed BGR
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const unsigned char* srcRow = src + y * srcPitch + 4 * x;
    unsigned char r = srcRow[0];
    unsigned char g = srcRow[1];
    unsigned char b = srcRow[2];

    int dstIdx = (y * width + x) * 3;
    dst[dstIdx + 0] = b;
    dst[dstIdx + 1] = g;
    dst[dstIdx + 2] = r;
}

// Map NvBufSurface -> EGLImage -> CUDA and copy RGBA into d_bgr
static bool upload_nvmm_rgba_to_d_bgr(
    NvBufSurface* surf,
    int           index,       // plane index (usually 0)
    unsigned char* d_bgr,      // destination on device
    int           width,
    int           height,
    cudaStream_t  stream
)
{
    // Map for device access
    if (NvBufSurfaceMap(surf, index, 0, NVBUF_MAP_READ) != 0)
    {
        std::cerr << "[CUDA-EGL] NvBufSurfaceMap failed\n";
        return false;
    }

    if (NvBufSurfaceSyncForDevice(surf, index, 0) != 0)
    {
        std::cerr << "[CUDA-EGL] NvBufSurfaceSyncForDevice failed\n";
        NvBufSurfaceUnMap(surf, index, 0);
        return false;
    }

    // Map to EGLImage
    if (NvBufSurfaceMapEglImage(surf, index) != 0)
    {
        std::cerr << "[CUDA-EGL] NvBufSurfaceMapEglImage failed\n";
        NvBufSurfaceUnMap(surf, index, 0);
        return false;
    }

    EGLImageKHR eglImage = surf->surfaceList[index].mappedAddr.eglImage;
    if (!eglImage)
    {
        std::cerr << "[CUDA-EGL] eglImage is null\n";
        NvBufSurfaceUnMapEglImage(surf, index);
        NvBufSurfaceUnMap(surf, index, 0);
        return false;
    }

    CUgraphicsResource cuRes = nullptr;
    CUeglFrame eglFrame;

    CUresult r = cuGraphicsEGLRegisterImage(
        &cuRes,
        eglImage,
        CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
    if (r != CUDA_SUCCESS)
    {
        checkCu(r, "cuGraphicsEGLRegisterImage");
        NvBufSurfaceUnMapEglImage(surf, index);
        NvBufSurfaceUnMap(surf, index, 0);
        return false;
    }

    r = cuGraphicsResourceGetMappedEglFrame(
        &eglFrame,
        cuRes,
        0, 0);
    if (r != CUDA_SUCCESS)
    {
        checkCu(r, "cuGraphicsResourceGetMappedEglFrame");
        cuGraphicsUnregisterResource(cuRes);
        NvBufSurfaceUnMapEglImage(surf, index);
        NvBufSurfaceUnMap(surf, index, 0);
        return false;
    }

    if (eglFrame.frameType != CU_EGL_FRAME_TYPE_PITCH)
    {
        std::cerr << "[CUDA-EGL] Unexpected frameType (not PITCH)\n";
        cuGraphicsUnregisterResource(cuRes);
        NvBufSurfaceUnMapEglImage(surf, index);
        NvBufSurfaceUnMap(surf, index, 0);
        return false;
    }

    // unsigned char* srcDevPtr = static_cast<unsigned char*>(eglFrame.pPitch[0]);
    unsigned char* srcDevPtr = static_cast<unsigned char*>(eglFrame.frame.pPitch[0]);
    int srcPitch             = static_cast<int>(eglFrame.pitch);

    dim3 block(16, 16);
    dim3 grid(
        (width  + block.x - 1) / block.x,
        (height + block.y - 1) / block.y);

    rgba_to_bgr_linear_kernel<<<grid, block, 0, stream>>>(
        srcDevPtr,
        srcPitch,
        width,
        height,
        d_bgr);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "[CUDA-EGL] rgba_to_bgr_linear_kernel error: "
                  << cudaGetErrorString(err) << "\n";
    }

    cuGraphicsUnregisterResource(cuRes);
    NvBufSurfaceUnMapEglImage(surf, index);
    NvBufSurfaceUnMap(surf, index, 0);

    return (err == cudaSuccess);
}




// ------------------------------ FRAME CAPTURE THREAD ------------------------------ //
/*
void frame_capture_thread_v4l2(V4L2MMapCamera& cam) {

    nvtxRangePush("FrameCaptureThread_V4L2");

    // Timing stats to check fps and decode time
    using clock = std::chrono::steady_clock;
    auto last_print         = clock::now();
    auto last_dequeue       = clock::now();
    int frames              = 0;
    double sum_decode_ms    = 0.0;
    double sum_period_ms    = 0.0;

    // Init decoder once with the camera’s negotiated params
    DecodeParams dp;
    dp.pixfmt = cam.pixfmt();
    dp.width  = cam.width();
    dp.height = cam.height();
    dp.mode   = DecodeMode::Auto;  // or force a mode
    dp.to_bgr = true;

    std::cout << "Initializing decoder...\n";
    FrameDecoder decoder(dp, DecodeBackend::CPU_OpenCV);
    std::cout << "->decoder initialized.\n";

    // Show what we’re running with
    std::cout << "\nV4L2 capture: " << cam.width() << "x" << cam.height()
              << " " << V4L2MMapCamera::fourccToString(cam.pixfmt()) << "\n->match output format\n\n";

    // Capture loop
    while (keep_running) {

        // Wait for readiness (avoid busy wait)
        fd_set fds; 
        FD_ZERO(&fds); 
        FD_SET(cam.fd(), &fds);
        timeval tv{}; 
        tv.tv_sec = 2; 
        tv.tv_usec = 0;
        int r = select(cam.fd()+1, &fds, nullptr, nullptr, &tv);
        if (r == -1) { if (errno == EINTR) continue; perror("select"); break; }
        if (r == 0)  { std::cerr << "select timeout\n"; continue; }

        // Dequeue a filled buffer
        V4L2MMapCamera::Frame f;
        if (!cam.dequeue(f)) continue; // EAGAIN, try again

        auto t_deq = clock::now();
        double period_ms = std::chrono::duration<double, std::milli>(t_deq - last_dequeue).count();
        last_dequeue = t_deq;

        // Decode frame
        nvtxRangePush("DecodeFrame");
        cv::Mat decoded_frame = decoder.decode(f.data, f.bytesused);
        nvtxRangePop(); // DecodeFrame

        auto t_after_decode = clock::now();
        double decode_ms = std::chrono::duration<double, std::milli>(t_after_decode - t_deq).count();

        // Re-queue ASAP (copied/decompressed into RAM already)
        cam.requeue(f);

        // Check decode worked
        if (decoded_frame.empty()) {
            std::cerr << "Failed to decode frame\n";
            continue;
        }

        // TEMP: direct view of raw camera frame
        cv::imshow("RG12_RAW_VIEW", decoded_frame);
        int k = cv::waitKey(1);
        if (k == 27) { // ESC to quit
            keep_running = false;
            break;
        }
        

        // Hand off the decoded frame to inference
        {
            std::lock_guard<std::mutex> lk(frame_mutex);
            decoded_frame.copyTo(frame_back);
            new_frame_available = true;
        }

        // Notify inference thread
        frame_ready.notify_one();

        // Print stats
        frames++;
        if (frames > 1) sum_period_ms += period_ms;
        sum_decode_ms += decode_ms;

        auto now = clock::now();
        double elapsed = std::chrono::duration<double>(now - last_print).count();
        if (elapsed >= 5.0) {
            double src_fps   = (frames > 1) ? 1000.0 / (sum_period_ms / (frames - 1)) : 0.0;
            double proc_fps  = frames / elapsed;
            double avg_dec   = sum_decode_ms / frames;
            double avg_per   = (frames > 1) ? sum_period_ms / (frames - 1) : 0.0;
            double duty      = (avg_per > 0.0) ? (avg_dec / avg_per) * 100.0 : 0.0;

            std::cout << "\r[V4L2] Source FPS: " << src_fps
                      << " | Processed FPS: " << proc_fps
                      << " | Avg period: " << avg_per << " ms"
                      << " | Avg imdecode: " << avg_dec << " ms"
                      << " | Duty: " << duty << " %    " << std::flush;

            last_print = now;
            frames = 0;
            sum_decode_ms = 0.0;
            sum_period_ms = 0.0;
        }

    }

    std::cout << std::endl;
    nvtxRangePop(); // FrameCaptureThread_V4L2
}
*/
/*
void frame_capture_thread(cv::VideoCapture& cap, int cap_width, int cap_height) {

    nvtxRangePush("FrameCaptureThread_GStreamer");

    using clock = std::chrono::steady_clock;
    auto last_print   = clock::now();
    auto last_capture = clock::now();

    int    frames        = 0;
    double sum_period_ms = 0.0;
    double sum_grab_ms   = 0.0;

    cv::Mat frame;

    std::cout << "Starting GStreamer capture loop...\n";

    while (keep_running) {

        auto t0 = clock::now();

        // Grab + decode next frame from GStreamer pipeline
        if (!cap.read(frame)) {
            std::cerr << "GStreamer cap.read() failed or EOS\n";
            // Small sleep to avoid tight error loop
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        auto t1 = clock::now();

        double period_ms = std::chrono::duration<double, std::milli>(t0 - last_capture).count();
        last_capture = t0;
        double grab_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Ensure size is exactly what the rest of the pipeline expects
        if (frame.cols != cap_width || frame.rows != cap_height) {
            cv::resize(frame, frame, cv::Size(cap_width, cap_height));
        }

        // Hand off the decoded BGR frame to inference
        {
            std::lock_guard<std::mutex> lk(frame_mutex);
            frame.copyTo(frame_back);
            new_frame_available = true;
        }

        frame_ready.notify_one();

        // Stats
        frames++;
        if (frames > 1) sum_period_ms += period_ms;
        sum_grab_ms += grab_ms;

        auto now = clock::now();
        double elapsed = std::chrono::duration<double>(now - last_print).count();
        if (elapsed >= 5.0) {
            double src_fps  = (frames > 1) ? 1000.0 / (sum_period_ms / (frames - 1)) : 0.0;
            double proc_fps = frames / elapsed;
            double avg_grab = sum_grab_ms / frames;
            double avg_per  = (frames > 1) ? sum_period_ms / (frames - 1) : 0.0;
            double duty     = (avg_per > 0.0) ? (avg_grab / avg_per) * 100.0 : 0.0;

            std::cout << "\r[GST] Source FPS: " << src_fps
                      << " | Processed FPS: " << proc_fps
                      << " | Avg period: " << avg_per << " ms"
                      << " | Avg grab: " << avg_grab << " ms"
                      << " | Duty: " << duty << " %    " << std::flush;

            last_print   = now;
            frames       = 0;
            sum_grab_ms  = 0.0;
            sum_period_ms= 0.0;
        }
    }

    std::cout << std::endl;
    nvtxRangePop(); // FrameCaptureThread_GStreamer
}
*/
void frame_capture_thread(
    GstElement*   sink,
    int           cap_width,
    int           cap_height,
    unsigned char* d_bgr,
    cudaStream_t  gst_stream
) {
    nvtxRangePush("FrameCaptureThread_GST_NVMM");

    using clock = std::chrono::steady_clock;
    auto last_print   = clock::now();
    auto last_capture = clock::now();

    int    frames        = 0;
    double sum_period_ms = 0.0;   // for source FPS

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

        // --- NVMM RGBA -> device BGR (d_bgr) ---
        if (!upload_nvmm_rgba_to_d_bgr(surf, 0, d_bgr, cap_width, cap_height, gst_stream)) {
            std::cerr << "[GST] upload_nvmm_rgba_to_d_bgr() failed\n";
            gst_buffer_unmap(buffer, &map);
            gst_sample_unref(sample);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);

        // Hand off the decoded BGR frame to inference/display
        {
            std::lock_guard<std::mutex> lk(frame_mutex);
            new_frame_available = true;
        }
        frame_ready.notify_one();

        // --- Stats: only source + processed FPS ---
        double period_ms = std::chrono::duration<double, std::milli>(t0 - last_capture).count();
        last_capture = t0;

        frames++;
        if (frames > 1) {
            sum_period_ms += period_ms;  // skip first period (frames>1)
        }

        auto now = clock::now();
        double elapsed = std::chrono::duration<double>(now - last_print).count();
        if (elapsed >= 5.0 && frames > 1) {
            double src_fps  = 1000.0 / (sum_period_ms / (frames - 1)); // based on inter-arrival
            double proc_fps = frames / elapsed;                         // loop throughput

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


// -------------------------------- INFERENCE THREAD -------------------------------- //


void inference_thread(
    unsigned char* h_frame_bgr_pinned, unsigned char* d_bgr, unsigned char* d_resized,
    int yolo_engine_img_width, int yolo_engine_img_height, int cap_width, int cap_height,
    float scaleX, float scaleY, size_t pinned_size,
    cudaStream_t stream1, cudaStream_t stream2,
    int num_anchors, float conf_thresh, float iou_thresh,
    Detection* d_final, Detection* d_dets, Detection* d_compacted, int* d_mask,
    int* d_count_compact, int* d_count_final, int* h_count_final,
    const EngineIO& yoloIO,
    const EngineIO& clsIO,
    /* action cls */
    const int maxSquare, const int cls_engine_img_width, const int cls_engine_img_height,
    unsigned char* d_cls_crop, unsigned char* d_cls_square, 
    unsigned char* d_cls_bgr96, ClsDevParams* d_cls_params, ActionVis* d_vis,
    /* multi stream */
    ActionVis* h_vis, int num_cameras
) {

    nvtxRangePush("InferenceThread");

    // h_frame_bgr_pinned will be unused once preprocess is fully GPU-only
    (void)h_frame_bgr_pinned;

    // 10% padding around box when cropping for cls
    const float pad_ratio = 0.10f; 

    // Event to sync streams
    cudaEvent_t ev_count_final_ready;
    cudaEventCreateWithFlags(&ev_count_final_ready, cudaEventDisableTiming);

    // Event for visualization readiness
    cudaEvent_t ev_vis_ready;
    cudaEventCreateWithFlags(&ev_vis_ready, cudaEventDisableTiming);

    while (keep_running) {

        // Wait for a new frame from the capture thread
        std::unique_lock<std::mutex> lock(frame_mutex);
        frame_ready.wait(lock, [] {
            return new_frame_available.load() || !keep_running.load();
        });

        if (!keep_running && !new_frame_available) {
            // Exit cleanly if shutting down and no new frame is pending
            lock.unlock();
            break;
        }

        // capture thread has written a new frame into d_bgr
        new_frame_available = false;
        // release lock so capture can proceed
        lock.unlock(); 


        // ---------------------- YOLO PREPROCESSING ----------------------
        nvtxRangePush("YOLO_Preprocessing");
        if (yoloIO.inType == nvinfer1::DataType::kFLOAT) {
            yolo_preprocess_gpu_batched(
                static_cast<float*>(yoloIO.dIn),          
                h_frame_bgr_pinned,                          
                d_bgr,                                    
                d_resized,                        
                cap_width, cap_height,
                yolo_engine_img_width, yolo_engine_img_height,
                scaleX,      
                scaleY,
                pinned_size,
                num_cameras,                          
                stream1                
            );
        } else {
            yolo_preprocess_gpu_batched(
                static_cast<__half*>(yoloIO.dIn),          
                h_frame_bgr_pinned,                          
                d_bgr,                                    
                d_resized,                        
                cap_width, cap_height,
                yolo_engine_img_width, yolo_engine_img_height,
                scaleX,      
                scaleY,
                pinned_size,
                num_cameras,                          
                stream1  
            );
        }
        nvtxRangePop(); // YOLO_Preprocessing


        // ---------------------- YOLO INFERENCE ----------------------
        nvtxRangePush("YOLO_Inference");
        yoloIO.ctx->enqueueV2(yoloIO.bindings, stream1, nullptr);
        nvtxRangePop(); // YOLO_Inference


        // ---------------------- YOLO POSTPROCESSING ----------------------
        nvtxRangePush("YOLO_Postprocessing");
        if (yoloIO.outType == nvinfer1::DataType::kFLOAT) {
            yolo_postprocess_gpu(
                static_cast<const float*>(yoloIO.dOut),
                num_anchors, conf_thresh, iou_thresh,
                d_final, stream1, d_dets, d_compacted, d_mask, 
                d_count_compact, d_count_final, h_count_final,
                ev_count_final_ready, num_cameras
            );
        } else { // kHALF
            yolo_postprocess_gpu(
                static_cast<const __half*>(yoloIO.dOut),
                num_anchors, conf_thresh, iou_thresh,
                d_final, stream1, d_dets, d_compacted, d_mask, 
                d_count_compact, d_count_final, h_count_final,
                ev_count_final_ready, num_cameras
            );
        }
        nvtxRangePop(); // YOLO_Postprocessing

        // GPU-side dependency: action stream waits until YOLO dets + count are ready
        cudaStreamWaitEvent(stream2, ev_count_final_ready, 0);

        // for debug safety
        //std::cout << "Detections: " << *h_count_final << std::endl;

        // If we have detections, run the classifier on each one
        if (*h_count_final > 0) {

            const int N = *h_count_final;


            // ---------------------- ACTION CLS PREPROCESSING ----------------------
            nvtxRangePush("ActionCls_Preprocessing_Batched");
            action_cls_preprocess_gpu_staged_batched_EI_copycat(
                d_bgr, cap_width, cap_height,
                d_final,        
                scaleX, scaleY, pad_ratio,
                d_cls_crop, cap_width, cap_height,
                d_cls_square, maxSquare,
                d_cls_bgr96,
                static_cast<float*>(clsIO.dIn),
                cls_engine_img_width, cls_engine_img_height,
                d_cls_params,
                N,
                stream2
            );
            nvtxRangePop(); // ActionCls_Preprocessing_Batched


            // ---------------------- ACTION CLS INFERENCE ----------------------
            nvtxRangePush("ActionCls_Inference");
            // set actual batch N for this frame
            nvinfer1::Dims4 inDims{N, 96, 96, 1};
            if (!clsIO.ctx->setBindingDimensions(0, inDims) ||
                !clsIO.ctx->allInputDimensionsSpecified()) {
                std::cerr << "[ACTION CLS]\n->setBindingDimensions failed\n";
            }
            clsIO.ctx->enqueueV2(clsIO.bindings, stream2, nullptr);
            nvtxRangePop(); // ActionCls_Inference


            // ---------------------- ACTION CLS POSTPROCESSING ----------------------
            nvtxRangePush("ActionCls_Postprocessing_Batched");
            action_cls_postprocess_gpu_batched(
                d_final,
                static_cast<const float*>(clsIO.dOut),
                N,
                scaleX, scaleY,
                d_vis,
                stream2
            );
            nvtxRangePop(); // ActionCls_Postprocessing_Batched

            cudaMemcpyAsync(h_vis, d_vis, N*sizeof(ActionVis), cudaMemcpyDeviceToHost, stream2);
            cudaEventRecord(ev_vis_ready, stream2);

            // Wait until ActionVis packets are ready on host
            cudaStreamWaitEvent(stream2, ev_vis_ready, 0);

        #if DEBUG_VIS

            nvtxRangePush("DebugVisualization");

            // ----------------------------------------------------------
            // 1) Download current GPU frame for visualization
            //    (later maybe move drawing to GPU)
            // ----------------------------------------------------------
            static cv::Mat debug_frame;
            if (debug_frame.empty()) {
                debug_frame.create(cap_height, cap_width, CV_8UC3);
            }

            cudaMemcpyAsync(
                debug_frame.data,
                d_bgr,
                static_cast<size_t>(cap_width) * cap_height * 3,
                cudaMemcpyDeviceToHost,
                stream1
            );
            cudaStreamSynchronize(stream1);

            // ----------------------------------------------------------
            // 2) Downscale debug_frame → frame_front for faster display
            // ----------------------------------------------------------
            {
                std::lock_guard<std::mutex> lock2(frame_copy_mutex);
                cv::resize(debug_frame, frame_front,
                           cv::Size(cap_width / 2, cap_height / 2),
                           0, 0, cv::INTER_AREA);
            }

            // ----------------------------------------------------------
            // 3) DEBUG_ONLY_FRAME → display the frame and skip overlays
            // ----------------------------------------------------------
            #if DEBUG_ONLY_FRAME
                nvtxRangePop(); // DebugVisualization
                continue;
            #endif

            // ----------------------------------------------------------
            // 4) Draw overlays on frame_front
            // ----------------------------------------------------------

            static const cv::Scalar COLORS[3] = {
                cv::Scalar(0,255,0),   // green
                cv::Scalar(0,0,255),   // red
                cv::Scalar(0,255,255)  // yellow
            };
            static const char* CLABELS[3] = {"G","R","Y"};

            float visScaleX = 0.5f;
            float visScaleY = 0.5f;

            auto clampRect = [&](const cv::Rect& r)->cv::Rect {
                return r & cv::Rect(0, 0, frame_front.cols, frame_front.rows);
            };

            for (int i = 0; i < N; ++i) {
                const ActionVis& v = h_vis[i];

                cv::Rect box(
                    cv::Point(static_cast<int>(v.x1 * visScaleX),
                              static_cast<int>(v.y1 * visScaleY)),
                    cv::Point(static_cast<int>(v.x2 * visScaleX),
                              static_cast<int>(v.y2 * visScaleY))
                );

                cv::Rect roi = clampRect(box);

                int idx = (v.cls < 0) ? 0 : (v.cls > 2 ? 2 : v.cls);
                const cv::Scalar col = COLORS[idx];
                cv::rectangle(frame_front, box, col, 2);

                char txt[64];
                std::snprintf(txt, sizeof(txt), "%s (cam %d)", CLABELS[idx], v.cam);
                cv::putText(frame_front, txt,
                            {box.x, std::max(0, box.y - 6)},
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv::LINE_AA);
            }

            nvtxRangePop(); // DebugVisualization

        #endif // DEBUG_VIS



    }

    nvtxRangePop(); // "InferenceThread"
    }

    cudaEventDestroy(ev_count_final_ready);
    cudaEventDestroy(ev_vis_ready);
}