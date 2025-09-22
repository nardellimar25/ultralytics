#pragma once

#include <opencv2/opencv.hpp>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cuda_runtime.h>
#include <NvInfer.h>

#include "cuda_detection_struct.h"
#include "v4l2_mmap_camera.h"
#include "engine_io.hpp"  
#include "cuda_action_preprocess.cuh"
#include "cuda_visual_struct.cuh"

// ---- shared globals ----
extern cv::Mat frame_back;         // Written by capture, read by inference
extern cv::Mat frame_front;        // Written by inference, read by UI

extern std::mutex frame_mutex;
extern std::mutex frame_copy_mutex;
// extern std::mutex detection_mutex;

extern std::condition_variable frame_ready;
extern std::atomic<bool> new_frame_available;
extern std::atomic<bool> keep_running;

// ------------------------------ THREADS ENTRY POINTS ------------------------------ //

// Frame capture thread
void frame_capture_thread(V4L2MMapCamera& cam);

// Inference thread
void inference_thread(
    unsigned char* pinned_frame_data, unsigned char* d_bgr, unsigned char* d_resized,
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
    unsigned char* d_cls_crop, unsigned char* d_cls_square, unsigned char* d_cls_bgr96, 
    ClsDevParams* d_cls_params, ActionVis* d_vis, ActionVis* h_vis, 
    /* multi stream */
    int num_cameras
);