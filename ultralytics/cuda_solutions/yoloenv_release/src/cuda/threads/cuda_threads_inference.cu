// src/cuda/cuda_threads_inference.cu

#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
#include <mutex>
#include <algorithm>
#include <iostream>

#include <nvtx3/nvToolsExt.h>

#include "cuda_yolo_preprocess.cuh"
#include "cuda_action_preprocess.cuh"
#include "cuda_yolo_postprocess.cuh"
#include "cuda_action_postprocess.cuh"
#include "threads/cuda_threads.cuh"
#include "cuda_structs.cuh"
#include "engine_io.hpp"
#include "engine_debug_utils.h"
#include "yolodetect.h"
#include "debug_vis.cuh"

// new: debug snapshot 
#include "threads/debug_snapshot.cuh"



// -------------------------------- MULTI STREAM INFERENCE THREAD -------------------------------- //


void multi_stream_inference_thread_full_gpu(
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

    // TODO : move as parameter
    // 10% padding for cls crops
    const float pad_ratio = 0.10f;  

    // TODO : move as parameter
    // Events to sync between streams
    cudaEvent_t ev_count_final_ready;
    cudaEventCreateWithFlags(&ev_count_final_ready, cudaEventDisableTiming);
    // cudaEvent_t ev_vis_ready;
    // cudaEventCreateWithFlags(&ev_vis_ready, cudaEventDisableTiming);

    // latest frame id seen
    uint64_t last_seen_frame = 0;
    // latest number of detections
    int last_cls_batch = -1;

    while (keep_running) {

        uint64_t cur_frame = 0;

        // Wait until new frame_id is published
        {
            std::unique_lock<std::mutex> lock(frame_mutex);
            frame_ready.wait(lock, [&] {
                return !keep_running.load(std::memory_order_relaxed) ||
                    g_frame_id.load(std::memory_order_acquire) != last_seen_frame;
            });

            if (!keep_running.load(std::memory_order_relaxed)) {
                break;
            }

            // Grab the newest published frame id
            cur_frame = g_frame_id.load(std::memory_order_acquire);
        }

        // Mark as consumed
        last_seen_frame = cur_frame;


        // GPU-side sync: wait until d_bgr_undistorted is ready on the GPU
        cudaStreamWaitEvent(stream1, ev_frame_ready, 0);

        // ----------------------------- YOLO PREPROCESS ----------------------------- //
        
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


        // ----------------------------- YOLO INFERENCE ------------------------------ //

        nvtxRangePush("YOLO_Inference");
        yoloIO.ctx->enqueueV2(yoloIO.bindings, stream1, nullptr);
        nvtxRangePop(); // YOLO_Inference


        // ----------------------------- YOLO POSTPROCESS ---------------------------- //

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

        // TODO: remove the wait and the memcpy, 
        // make the action cls start and stop if no detections
        // Make action stream wait for YOLO detections
        cudaStreamWaitEvent(stream2, ev_count_final_ready, 0);

        const int N = *h_count_final;

        if (N > 0) {

            // ------------------------ ACTION CLS PREPROCESS ------------------------ //

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


            // ------------------------ ACTION CLS INFERENCE ------------------------- //

            // set bindings dynamic dimensions
            if (N != last_cls_batch) {

                nvinfer1::Dims4 inDims{N, 96, 96, 1};
                if (!clsIO.ctx->setBindingDimensions(0, inDims) ||
                    !clsIO.ctx->allInputDimensionsSpecified()) {
                    std::cerr << "[ACTION CLS]\n->setBindingDimensions failed\n";
                }

                last_cls_batch = N;

            }

            nvtxRangePush("ActionCls_Inference");
            clsIO.ctx->enqueueV2(clsIO.bindings, stream2, nullptr);
            nvtxRangePop(); // ActionCls_Inference


            // ------------------------ ACTION CLS POSTPROCESS ----------------------- //

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
            // cudaMemcpyAsync(h_vis, d_vis, N * sizeof(ActionVis), cudaMemcpyDeviceToHost, stream2);
            // cudaEventRecord(ev_vis_ready, stream2);
            // cudaEventSynchronize(ev_vis_ready);   

            // debugVis_update_cached_detections(h_vis, N);

            // NEW
            debug_snapshot_publish_async(d_vis, N, stream2);

            // Quick debug: count detections per camera in h_vis
            // {
            //     int counts[8] = {0};
            //     for (int i = 0; i < N; ++i) {
            //         const ActionVis& v = h_vis[i];
            //         if (v.cam >= 0 && v.cam < 8) {
            //             counts[v.cam]++;
            //         }
            //     }
            //     std::cout << "\n[INF]\nDetections per cam: ";
            //     for (int cam = 0; cam < num_cameras; ++cam) {
            //         std::cout << "cam" << cam << "=" << counts[cam] << " ";
            //     }
            //     std::cout << std::endl;
            // }

        }
        else {

            // No detections: clear published debug vis
            debug_snapshot_publish_async(nullptr, 0, stream2);
            continue;

        }

    } // while(keep_running)

    cudaEventDestroy(ev_count_final_ready);
    // cudaEventDestroy(ev_vis_ready);

    nvtxRangePop(); // "InferenceThread"
}