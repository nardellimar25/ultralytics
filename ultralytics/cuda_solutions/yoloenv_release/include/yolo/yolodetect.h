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

// Globals
extern int conf_slider;
extern int iou_slider;
extern float conf_thresh;
extern float iou_thresh;

class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

extern Logger gLogger;

ICudaEngine* loadEngine(const std::string& engineFile, IRuntime*& runtime);

void on_trackbar(int, void*);

#endif // YOLODETECT_H
