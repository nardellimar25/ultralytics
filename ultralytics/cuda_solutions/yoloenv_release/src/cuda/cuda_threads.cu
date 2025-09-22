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
// std::mutex detection_mutex;

std::condition_variable frame_ready;
std::atomic<bool> new_frame_available(false);
std::atomic<bool> keep_running(true);



// ------------------------------ FRAME CAPTURE THREAD ------------------------------ //

void frame_capture_thread(V4L2MMapCamera& cam) {

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

    // Wrap your pinned host buffer as a cv::Mat (no allocation, just a header)
    cv::Mat pinned_mat(cap_height, cap_width, CV_8UC3, h_frame_bgr_pinned);

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
        frame_ready.wait(lock, [] { return new_frame_available.load(); });
        new_frame_available = false;

        // Latch the decoded frame into pinned memory
        nvtxRangePush("LatchToPinned");
        if (frame_back.size() == cv::Size(cap_width, cap_height) && frame_back.type() == CV_8UC3) {
            frame_back.copyTo(pinned_mat);
        } else {
            // Fallbacks
            cv::Mat tmp;
            if (frame_back.type() != CV_8UC3) {
                // e.g., if you ever feed grayscale/YUYV here (shouldn't for MJPEG)
                cv::cvtColor(frame_back, tmp, cv::COLOR_GRAY2BGR);
            } else {
                tmp = frame_back;
            }
            if (tmp.size() != cv::Size(cap_width, cap_height)) {
                cv::resize(tmp, pinned_mat, cv::Size(cap_width, cap_height));
            } else {
                tmp.copyTo(pinned_mat);
            }
        }
        nvtxRangePop(); // LatchToPinned

        // Release the lock ASAP so capture can proceed
        lock.unlock();


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


        nvtxRangePush("YOLO_Inference");
        yoloIO.ctx->enqueueV2(yoloIO.bindings, stream1, nullptr);
        nvtxRangePop(); // YOLO_Inference


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
        cudaEventSynchronize(ev_count_final_ready);

        //std::cout << "Detections: " << *h_count_final << std::endl;

        // If we have detections, run the classifier on each one
        if (*h_count_final > 0) {

            const int N = *h_count_final;

            nvtxRangePush("ActionCls_Preprocessing_Batched");
            action_cls_preprocess_gpu_staged_batched(
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


            nvtxRangePush("ActionCls_Inference");
            // NHWC (-1,96,96,1) -> set actual batch N for this frame
            nvinfer1::Dims4 inDims{N, 96, 96, 1};
            if (!clsIO.ctx->setBindingDimensions(0, inDims) ||
                !clsIO.ctx->allInputDimensionsSpecified()) {
                std::cerr << "[ACTION CLS] setBindingDimensions failed\n";
            }
            clsIO.ctx->enqueueV2(clsIO.bindings, stream2, nullptr);
            nvtxRangePop(); // ActionCls_Inference


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
            cudaEventSynchronize(ev_vis_ready);

        #if DEBUG_VIS

                nvtxRangePush("DebugVisualization");
                {
                    std::lock_guard<std::mutex> lock2(frame_copy_mutex);
                    frame_back.copyTo(frame_front);
                }

                static const cv::Scalar COLORS[3] = {
                    cv::Scalar(0,255,0),   // 0 = green
                    cv::Scalar(0,0,255),   // 1 = red
                    cv::Scalar(0,255,255)  // 2 = yellow
                };
                static const char* CLABELS[3] = {"G","R","Y"};

                // Helper: clamp ROI to frame bounds
                auto clampRect = [&](const cv::Rect& r)->cv::Rect {
                    return r & cv::Rect(0, 0, frame_front.cols, frame_front.rows);
                };

                // Which class to blur (now: red)
                constexpr int BLUR_CLASS = 1;

                for (int i = 0; i < N; ++i) {
                    const ActionVis& v = h_vis[i];

                    // Build box directly from final pixel coords (already scaled on GPU)
                    cv::Rect box(cv::Point(v.x1, v.y1), cv::Point(v.x2, v.y2));
                    cv::Rect roi = clampRect(box);

                    // Optional Gaussian blur for the chosen class
                    if (v.cls == BLUR_CLASS && roi.area() > 0) {
                        const int maxSide = std::max(roi.width, roi.height);
                        int ksz = std::max(3, (maxSide / 8) | 1);        // odd kernel
                        int kcap = std::max(3, ((std::min(roi.width, roi.height) - 1) | 1));
                        if (ksz > kcap) ksz = kcap;
                        if ((ksz & 1) == 0) ++ksz;
                        cv::GaussianBlur(frame_front(roi), frame_front(roi),
                                        {ksz, ksz}, 0, 0, cv::BORDER_REPLICATE);
                    }

                    // Draw overlay AFTER blur so it stays crisp
                    //const cv::Scalar col = COLORS[std::clamp(v.cls, 0, 2)]; cpp17
                    int idx = (v.cls < 0) ? 0 : (v.cls > 2 ? 2 : v.cls);
                    const cv::Scalar col = COLORS[idx];
                    cv::rectangle(frame_front, box, col, 2);

                    char txt[64];
                    //std::snprintf(txt, sizeof(txt), "%s (cam %d)", CLABELS[std::clamp(v.cls,0,2)], v.cam); cpp17
                    std::snprintf(txt, sizeof(txt), "%s (cam %d)", CLABELS[idx], v.cam);
                    cv::putText(frame_front, txt,
                                {box.x, std::max(0, box.y - 6)},
                                cv::FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv::LINE_AA);
                }

                nvtxRangePop(); // DebugVisualization

        #endif
    }

    nvtxRangePop(); // "InferenceThread"
    }

    cudaEventDestroy(ev_count_final_ready);
    cudaEventDestroy(ev_vis_ready);
}