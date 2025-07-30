#ifndef YOLODETECT_H
#define YOLODETECT_H

#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>

using namespace nvinfer1;

// Globals (declared as extern)
extern int conf_slider;
extern int iou_slider;
extern float conf_thresh;
extern float iou_thresh;

class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

extern Logger gLogger;

struct Detection {
    cv::Rect box;
    float score;
    int class_id;
};

float sigmoid(float x);
float iou(const cv::Rect& a, const cv::Rect& b);

ICudaEngine* loadEngine(const std::string& engineFile, IRuntime*& runtime);

void preprocessImage(const cv::Mat& img, float* gpuInput, cudaStream_t stream, const int img_width, const int img_height);

std::vector<Detection> postprocessYoloOutput_nmsFalse(const float* output, int num_anchors, int num_classes, float conf_thresh, float iou_thresh);

void on_trackbar(int, void*);

#endif // YOLODETECT_H
