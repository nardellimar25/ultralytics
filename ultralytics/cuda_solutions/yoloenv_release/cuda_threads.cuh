#pragma once

#include <opencv2/opencv.hpp>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cuda_runtime.h>
#include <NvInfer.h>

#include "cuda_detection_struct.h"

// ---- shared globals ----
extern cv::Mat frame_back;         // Written by capture, read by inference
extern cv::Mat frame_front;        // Written by inference, read by UI

extern std::mutex frame_mutex;
extern std::mutex frame_copy_mutex;
// extern std::mutex detection_mutex;

extern std::condition_variable frame_ready;
extern std::atomic<bool> new_frame_available;
extern std::atomic<bool> keep_running;

// ---- thread entry points ----
void frame_capture_thread(cv::VideoCapture& cap);

void inference_thread(
    float* engine_input, unsigned char* pinned_frame_data, unsigned char* d_bgr, unsigned char* d_resized,
    int engine_img_width, int engine_img_height, int cap_width, int cap_height,
    float scaleX, float scaleY, size_t pinned_size, cudaStream_t stream1, 
    cudaStream_t stream2,
    nvinfer1::IExecutionContext* context, void** buffers,
    float* engine_output, int num_anchors, float conf_thresh, float iou_thresh,
    Detection* d_final, Detection* d_dets, Detection* d_compacted, int* d_mask, 
    int* d_count_final, int* h_count_final
);
