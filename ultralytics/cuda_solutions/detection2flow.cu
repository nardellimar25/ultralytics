// assuming yolov8s.engine extracted from yolov8s.onnx generated with nms=False



#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>

#include "yolodetect.h"

using namespace nvinfer1;


int main() {
    std::string engineFile = ENGINE_PATH;

    if (engineFile.empty()) {
        std::cerr << "Engine file path is empty. Please set ENGINE_PATH.\n";
        return -1;
    }
    const int img_width = 256;
    const int img_height = 256;
    const int num_classes = 80;
    const int num_anchors = 1344;

    cv::VideoCapture cap("/dev/video0", cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::cerr << "Impossibile aprire webcam\n";
        return -1;
    }

    IRuntime* runtime = nullptr;
    ICudaEngine* engine = loadEngine(engineFile, runtime);
    IExecutionContext* context = engine->createExecutionContext();

    const int inputIndex = engine->getBindingIndex("images");
    const int outputIndex = engine->getBindingIndex("output0");

    const int inputSize = 1 * 3 * img_height * img_width;
    const int outputSize = 1 * (num_classes + 4) * num_anchors;

    void* buffers[2];
    cudaMalloc(&buffers[inputIndex], inputSize * sizeof(float));
    cudaMalloc(&buffers[outputIndex], outputSize * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cv::Mat frame;

    cv::namedWindow("Detections with bbox", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Conf Threshold", "Detections with bbox", &conf_slider, 100, on_trackbar);
    cv::createTrackbar("IoU Threshold", "Detections with bbox", &iou_slider, 100, on_trackbar);
    on_trackbar(0, 0); // initialize thresholds

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Frame vuoto\n";
            break;
        }

        cv::Mat resizedForModel, resizedForDebug;
        cv::resize(frame, resizedForModel, cv::Size(img_width, img_height));
        resizedForDebug = resizedForModel.clone();
        // Pad resized_frame to match original frame size (centered)
        int top    = (frame.rows - img_height) / 2;
        int bottom = frame.rows - img_height - top;
        int left   = (frame.cols - img_width) / 2;
        int right  = frame.cols - img_width - left;

        cv::Mat padded_resized;
        cv::copyMakeBorder(resizedForDebug, resizedForDebug, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

        preprocessImage(resizedForModel, (float*)buffers[inputIndex], stream, img_width, img_height);

        bool success = context->enqueueV2(buffers, stream, nullptr);
        if (!success) {
            std::cerr << "Inference failed!\n";
            continue;
        }

        std::vector<float> output(outputSize);
        cudaMemcpyAsync(output.data(), buffers[outputIndex], outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        auto detections = postprocessYoloOutput_nmsFalse(output.data(), num_anchors, num_classes, conf_thresh, iou_thresh);

        float scaleX = static_cast<float>(frame.cols) / img_width;
        float scaleY = static_cast<float>(frame.rows) / img_height;

        for (auto& det : detections) {
            det.box.x = static_cast<int>(det.box.x * scaleX);
            det.box.y = static_cast<int>(det.box.y * scaleY);
            det.box.width = static_cast<int>(det.box.width * scaleX);
            det.box.height = static_cast<int>(det.box.height * scaleY);

            cv::rectangle(frame, det.box, {0, 255, 0}, 2);
            cv::putText(frame, "cls " + std::to_string(det.class_id) + " " + std::to_string(det.score),
                        det.box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 255, 0});
        }

        // Combine side by side
        cv::Mat combined_display;
        cv::hconcat(frame, resizedForDebug, combined_display);

        cv::imshow("Detections with bbox", combined_display);
        if (cv::waitKey(1) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}
