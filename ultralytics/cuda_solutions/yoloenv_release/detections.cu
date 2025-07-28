// assuming yolov8s.engine extracted from yolov8s.onnx generated with nms=False



#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <time.h>
#include <chrono>

#include "postProcess.cuh"
#include "yolodetect.h"

using namespace nvinfer1;


int main() {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    std::string engineFile = ENGINE_PATH;

    if (engineFile.empty()) {
        std::cerr << "Engine file path is empty. Please set ENGINE_PATH.\n";
        return -1;
    }

    const int img_width = 256;
    const int img_height = 256;
    const int num_classes = 80;
    const int num_anchors = 1344;
    const int max_detections = 100;

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

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    //cudaStream_t stream2;
    //cudaStreamCreate(&stream2);

    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) {
        std::cerr << "Frame vuoto\n";
        return -1;
    }
    // Calculate scale factors for bounding box coordinates
    float scaleX = static_cast<float>(frame.cols) / img_width;
    float scaleY = static_cast<float>(frame.rows) / img_height;

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

        cv::Mat resizedForModel;
        cv::resize(frame, resizedForModel, cv::Size(img_width, img_height));

        float timeStart = static_cast<float>(clock());
        preprocessImage(resizedForModel, (float*)buffers[inputIndex], stream1, img_width, img_height);
        float timeEnd = static_cast<float>(clock());
        std::cout << "\n-Preprocess time: " << (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000 << " ms\n";

        timeStart = static_cast<float>(clock());
        bool success = context->enqueueV2(buffers, stream1, nullptr);
        timeEnd = static_cast<float>(clock());
        std::cout << "-Inference time: " << (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000 << " ms\n";

        if (!success) {
            std::cerr << "Inference failed!\n";
            continue;
        }

        
        Detection* d_final;
        cudaMalloc(&d_final, max_detections * sizeof(Detection));

        float* engine_output = static_cast<float*>(buffers[outputIndex]);

        //auto start = std::chrono::high_resolution_clock::now();
        cudaEventRecord(start);
        int num_dets = run_postprocess_gpu(engine_output, num_anchors, conf_thresh, iou_thresh, max_detections, d_final);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << " postproc time : " << milliseconds << " ms" << std::endl;
        //cudaDeviceSynchronize();
        //auto end = std::chrono::high_resolution_clock::now();
        //auto duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        //std::cout << "*tot Postprocess time: " << duration_ms << " us*\n";


        // Only do bulk memory transfer if we have final detections
        if (num_dets > 0) {
            std::vector<Detection> detections(num_dets);

            timeStart = static_cast<float>(clock());
            cudaMemcpy(detections.data(), d_final, num_dets * sizeof(Detection), cudaMemcpyDeviceToHost);
            timeEnd = static_cast<float>(clock());
            std::cout << "-Copy detections time: " << (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000 << " ms\n\n";

            for (auto& det : detections) {
            int x = static_cast<int>(det.x1 * scaleX);
            int y = static_cast<int>(det.y1 * scaleY);
            int w = static_cast<int>((det.x2 - det.x1) * scaleX);
            int h = static_cast<int>((det.y2 - det.y1) * scaleY);

            cv::Rect box(cv::Point(x, y), cv::Size(w, h));

            cv::rectangle(frame, box, {0, 255, 0}, 2);
            cv::putText(frame, "cls " + std::to_string(det.class_id) + " " + std::to_string(det.score),
                        box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 255, 0});
            }

        }

        /*timeStart = static_cast<float>(clock());
        std::vector<float> output(outputSize);
        cudaMemcpyAsync(output.data(), buffers[outputIndex], outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream1);
        cudaStreamSynchronize(stream1);
        timeEnd = static_cast<float>(clock());
        std::cout << "Copy raw output time: " << (timeEnd - timeStart) / CLOCKS_PER_SEC * 1000 << " ms\n";


        timeStart = static_cast<float>(clock());
        auto detections = postprocessYoloOutput_nmsFalse(output.data(), num_anchors, num_classes, conf_thresh, iou_thresh);
        timeEnd = static_cast<float>(clock());
        std::cout << "Postprocessing time: " << (timeEnd - timeStart)  / CLOCKS_PER_SEC * 1000 << " ms\n";
        */

        cv::imshow("Detections with bbox", frame);
        if (cv::waitKey(1) == 27) break;

    }

    cap.release();
    cv::destroyAllWindows();
    cudaStreamDestroy(stream1);
    //cudaStreamDestroy(stream2);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}
