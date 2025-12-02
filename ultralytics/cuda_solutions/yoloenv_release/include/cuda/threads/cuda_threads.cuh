#pragma once

#include <opencv2/opencv.hpp>
#include <condition_variable>
#include <atomic>
#include <mutex>

#include <cuda_runtime.h>
#include "cuda_structs.cuh"
#include "engine_io.hpp"


// Forward-declare GstElement to avoid pulling all gst headers
typedef struct _GstElement GstElement;


// // ------------------------------ GLOBALS ------------------------------ //

extern cv::Mat frame_back;
extern cv::Mat frame_front;

extern std::mutex frame_mutex;
extern std::mutex frame_copy_mutex;

extern std::condition_variable frame_ready;
extern std::atomic<bool> new_frame_available;
extern std::atomic<bool> keep_running;


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
);


// ------------------------------ INFERENCE THREAD ------------------------------ //

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
);
