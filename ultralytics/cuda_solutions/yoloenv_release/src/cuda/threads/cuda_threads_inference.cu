// src/cuda/cuda_threads_inference.cu

#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
#include <mutex>
#include <algorithm>
#include <iostream>

#include <nvtx3/nvToolsExt.h>
#include <opencv2/opencv.hpp>

#include "cuda_yolo_preprocess.cuh"
#include "cuda_action_preprocess.cuh"
#include "cuda_yolo_postprocess.cuh"
#include "cuda_action_postprocess.cuh"
#include "threads/cuda_threads.cuh"
#include "cuda_structs.cuh"
#include "engine_io.hpp"
#include "engine_debug_utils.h"
#include "yolodetect.h"

// -------------------------------- INFERENCE THREAD -------------------------------- //

void inference_thread_full_gpu(
    unsigned char* d_bgr_undistorted, unsigned char* d_resized,
    int yolo_engine_img_width, int yolo_engine_img_height,
    int cap_width, int cap_height,
    float scaleX, float scaleY, size_t frame_bytes,
    cudaStream_t stream1, cudaStream_t stream2,
    int num_anchors, float conf_thresh, float iou_thresh,
    Detection* d_final, Detection* d_dets, Detection* d_compacted,
    int* d_mask, int* d_count_compact, int* d_count_final, int* h_count_final,
    const EngineIO& yoloIO, const EngineIO& clsIO,
    int maxSquare,
    int cls_engine_img_width, int cls_engine_img_height,
    unsigned char* d_cls_crop, unsigned char* d_cls_square, unsigned char* d_cls_bgr96,
    ClsDevParams* d_cls_params,
    ActionVis* d_vis, ActionVis* h_vis,
    int num_cameras,
    cudaEvent_t ev_frame_ready
) {
    nvtxRangePush("InferenceThread");

    // Attach CUDA context
    int dev = 0;
    cudaError_t cerr = cudaSetDevice(dev);
    if (cerr != cudaSuccess) {
        std::cerr << "[INF] cudaSetDevice(" << dev << ") failed in inference thread: "
                  << cudaGetErrorString(cerr) << "\n";
        nvtxRangePop();
        return;
    }

    const float pad_ratio = 0.10f;  // 10% padding for cls crops

    // Events to sync between streams
    cudaEvent_t ev_count_final_ready;
    cudaEventCreateWithFlags(&ev_count_final_ready, cudaEventDisableTiming);
    cudaEvent_t ev_vis_ready;
    cudaEventCreateWithFlags(&ev_vis_ready, cudaEventDisableTiming);

    while (keep_running) {

        // CPU-side sync: wait until capture thread signals a new frame
        std::unique_lock<std::mutex> lock(frame_mutex);
        frame_ready.wait(lock, [] {
            return new_frame_available.load() || !keep_running.load();
        });

        // Check for shutdown
        if (!keep_running) {
            lock.unlock();
            break;
        }

        new_frame_available = false;
        lock.unlock();

        // GPU-side sync: wait until d_bgr_undistorted is ready on the GPU
        cudaStreamWaitEvent(stream1, ev_frame_ready, 0);

        // ---------------------- YOLO PREPROCESS ---------------------- //
        nvtxRangePush("YOLO_Preprocessing");
        if (yoloIO.inType == nvinfer1::DataType::kFLOAT) {
            yolo_preprocess_gpu_batched(
                static_cast<float*>(yoloIO.dIn),
                nullptr,
                d_bgr_undistorted,
                d_resized,
                cap_width, cap_height,
                yolo_engine_img_width,
                yolo_engine_img_height,
                scaleX,
                scaleY,
                0,
                num_cameras,
                stream1
            );
        } else { // kHALF
            yolo_preprocess_gpu_batched(
                static_cast<__half*>(yoloIO.dIn),
                nullptr,
                d_bgr_undistorted,
                d_resized,
                cap_width, cap_height,
                yolo_engine_img_width,
                yolo_engine_img_height,
                scaleX,
                scaleY,
                0,
                num_cameras,
                stream1
            );
        }
        nvtxRangePop(); // YOLO_Preprocessing

        // ---------------------- YOLO INFERENCE ---------------------- //
        nvtxRangePush("YOLO_Inference");
        yoloIO.ctx->enqueueV2(yoloIO.bindings, stream1, nullptr);
        nvtxRangePop(); // YOLO_Inference

        // ---------------------- YOLO POSTPROCESS ---------------------- //
        nvtxRangePush("YOLO_Postprocessing");
        if (yoloIO.outType == nvinfer1::DataType::kFLOAT) {
            yolo_postprocess_gpu(
                static_cast<const float*>(yoloIO.dOut),
                num_anchors, conf_thresh, iou_thresh,
                d_final,
                stream1,
                d_dets, d_compacted, d_mask,
                d_count_compact, d_count_final, h_count_final,
                ev_count_final_ready,
                num_cameras
            );
        } else {
            yolo_postprocess_gpu(
                static_cast<const __half*>(yoloIO.dOut),
                num_anchors, conf_thresh, iou_thresh,
                d_final,
                stream1,
                d_dets, d_compacted, d_mask,
                d_count_compact, d_count_final, h_count_final,
                ev_count_final_ready,
                num_cameras
            );
        }
        nvtxRangePop(); // YOLO_Postprocessing

        // Make action stream wait for YOLO detections
        cudaStreamWaitEvent(stream2, ev_count_final_ready, 0);

        const int N = *h_count_final;
        if (N > 0) {

            // ---------------- ACTION CLS PREPROCESS ---------------- //
            nvtxRangePush("ActionCls_Preprocessing_Batched");
            action_cls_preprocess_gpu_staged_batched_EI_copycat(
                d_bgr_undistorted,
                cap_width, cap_height,
                d_final,
                scaleX, scaleY, pad_ratio,
                d_cls_crop, cap_width, cap_height,
                d_cls_square, maxSquare,
                d_cls_bgr96,
                static_cast<float*>(clsIO.dIn),
                cls_engine_img_width,
                cls_engine_img_height,
                d_cls_params,
                N,
                stream2
            );
            nvtxRangePop(); // ActionCls_Preprocessing_Batched

            // ---------------- ACTION CLS INFERENCE ---------------- //
            nvtxRangePush("ActionCls_Inference");
            {
                nvinfer1::Dims4 inDims{N, 96, 96, 1};
                if (!clsIO.ctx->setBindingDimensions(0, inDims) ||
                    !clsIO.ctx->allInputDimensionsSpecified()) {
                    std::cerr << "[ACTION CLS]\n->setBindingDimensions failed\n";
                }
            }
            clsIO.ctx->enqueueV2(clsIO.bindings, stream2, nullptr);
            nvtxRangePop(); // ActionCls_Inference

            // ---------------- ACTION CLS POSTPROCESS ---------------- //
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

            // D2H ActionVis for overlay/debug
            cudaMemcpyAsync(
                h_vis,
                d_vis,
                N * sizeof(ActionVis),
                cudaMemcpyDeviceToHost,
                stream2
            );
            cudaEventRecord(ev_vis_ready, stream2);
            cudaStreamWaitEvent(stream2, ev_vis_ready, 0);

        #if DEBUG_VIS
            nvtxRangePush("DebugVisualization");

            // 1) Download current frame from GPU just for debug/display
            {
                std::lock_guard<std::mutex> lock2(frame_copy_mutex);

                if (frame_back.empty() ||
                    frame_back.cols != cap_width ||
                    frame_back.rows != cap_height ||
                    frame_back.type() != CV_8UC3)
                {
                    frame_back.create(cap_height, cap_width, CV_8UC3);
                }

                cudaMemcpyAsync(
                    frame_back.data,
                    d_bgr_undistorted,
                    frame_bytes,
                    cudaMemcpyDeviceToHost,
                    stream2
                );
                cudaStreamSynchronize(stream2);

                // 2) Downscale frame_back → frame_front for faster display
                cv::resize(frame_back, frame_front,
                           cv::Size(cap_width / 2, cap_height / 2),
                           0, 0, cv::INTER_AREA);
            }

            // Optional: frame-only debug
            #if DEBUG_ONLY_FRAME
                nvtxRangePop(); // DebugVisualization
                continue;
            #endif

            // 3) Draw overlays on frame_front
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

            {
                std::lock_guard<std::mutex> lock2(frame_copy_mutex);
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
                    std::snprintf(txt, sizeof(txt), "%s (cam %d)",
                                  CLABELS[idx], v.cam);
                    cv::putText(frame_front, txt,
                                {box.x, std::max(0, box.y - 6)},
                                cv::FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv::LINE_AA);
                }
            }

            nvtxRangePop(); // DebugVisualization
        #endif // DEBUG_VIS

        } // if (N > 0)

    } // while(keep_running)

    cudaEventDestroy(ev_count_final_ready);
    cudaEventDestroy(ev_vis_ready);

    nvtxRangePop(); // "InferenceThread"
}
