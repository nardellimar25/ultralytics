// assuming yolov8s.engine extracted from yolov8s.onnx generated with nms=False



#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <time.h>
#include <thread>
#include <chrono>

#include "postProcess.cuh"
#include "yolodetect.h"

using namespace nvinfer1;


int main() {

    // Start the profiler to measure performance
    cudaProfilerStart();  

    //cudaEvent_t start, stop;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);
    //float milliseconds = 0;

    std::string engineFile = ENGINE_PATH;
    if (engineFile.empty()) {
        std::cerr << "Engine file path is empty. Please set correct ENGINE_PATH.\n";
        return -1;
    }
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = loadEngine(engineFile, runtime);
    IExecutionContext* context = engine->createExecutionContext();
    std::cout << "\n[Engine loaded successfully]\n*engine info:\n\n";
    std::cout << "Engine name: " << engine->getName() << std::endl;
    std::cout << "Number of bindings: " << engine->getNbBindings() << std::endl;

    const int inputIndex = engine->getBindingIndex("images");
    const int outputIndex = engine->getBindingIndex("output0");
    std::cout << "Input index: " << inputIndex << ", Output index: " << outputIndex << std::endl;

    nvinfer1::Dims inputDims = engine->getBindingDimensions(inputIndex);
    nvinfer1::Dims outputDims = engine->getBindingDimensions(outputIndex);
    std::cout << "Input dims: ";
    for (int i = 0; i < inputDims.nbDims; ++i) {
        std::cout << inputDims.d[i];
        if (i < inputDims.nbDims - 1) std::cout << "x";
    }
    std::cout << std::endl;
    std::cout << "Output dims: ";
    for (int i = 0; i < outputDims.nbDims; ++i) {
        std::cout << outputDims.d[i];
        if (i < outputDims.nbDims - 1) std::cout << "x";
    }
    std::cout << std::endl << std::endl;


    // Initialize based on engine dimensions
    const int img_width = inputDims.d[3];
    const int img_height = inputDims.d[2];
    const int num_classes = 80;
    const int num_anchors = outputDims.d[2];
    const int max_detections = 100;
    const int inputSize = inputDims.d[0] * inputDims.d[1] * img_height * img_width;
    const int outputSize = outputDims.d[0] * (num_classes + 4) * num_anchors;
    // get needed resized frame dimensions
    size_t input_size = img_height * img_width * 3;
    // Initialize number of detections
    int num_dets = 0;


    // Initialize CUDA buffers
    void* buffers[2];
    cudaMalloc(&buffers[inputIndex], inputSize * sizeof(float));
    cudaMalloc(&buffers[outputIndex], outputSize * sizeof(float));
    
    float* engine_output = static_cast<float*>(buffers[outputIndex]);

    // Initialize intermediate detections buffer
    Detection *d_dets;
    cudaMalloc(&d_dets, num_anchors * sizeof(Detection));

    // Initialize compacted detections buffer
    Detection* d_compacted;
    cudaMalloc(&d_compacted, max_detections * sizeof(Detection));

    // Initialize final detections results buffer
    Detection* d_final;
    cudaMalloc(&d_final, max_detections * sizeof(Detection));

    // Initialize BGR image preprocessing buffer
    uchar* d_bgr = nullptr;
    cudaMalloc(&d_bgr, input_size);
    
    // Initialize final count buffer
    int *d_count_final;
    cudaMalloc(&d_count_final, sizeof(int));

    // Initialize mask buffer
    int *d_mask;
    cudaMalloc(&d_mask, num_anchors * sizeof(int));

    // Initialize CUDA stream
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);




    // Initialize OpenCV VideoCapture
    cv::VideoCapture cap("/dev/video0", cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::cerr << "Impossibile aprire webcam\n";
        return -1;
    }
    // get original cap size
    const int cap_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const int cap_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    const int cap_channels = 3; // assume BGR
    // Allocate pinned memory for the frame data
    uchar* pinned_frame_data = nullptr;
    size_t pinned_size = cap_width * cap_height * cap_channels * sizeof(uchar);
    cudaHostAlloc((void**)&pinned_frame_data, pinned_size, cudaHostAllocDefault);

    // Create Mat to hold the frame data using pinned memory
    cv::Mat frame(cap_height, cap_width, CV_8UC3, pinned_frame_data);
    cv::Mat resizedForModel;
    cap.read(frame);
    if (frame.empty()) {
        std::cerr << "Frame vuoto\n";
        return -1;
    }

    // Calculate scale factors for bounding box coordinates
    float scaleX = static_cast<float>(frame.cols) / img_width;
    float scaleY = static_cast<float>(frame.rows) / img_height;


    // Create OpenCV windows and trackbars
    cv::namedWindow("Detections with bbox", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Conf Threshold", "Detections with bbox", &conf_slider, 100, on_trackbar);
    cv::createTrackbar("IoU Threshold", "Detections with bbox", &iou_slider, 100, on_trackbar);
    on_trackbar(0, 0);


    while (true) {

        cap.read(frame);
        if (frame.empty()) {
            std::cerr << "Frame vuoto\n";
            break;
        }

        
        // Resize the frame to the model input size
        cv::resize(frame, resizedForModel, cv::Size(img_width, img_height));

        // Event: preprocess
        //cudaEventRecord(start);
        //preprocessImage(resizedForModel, (float*)buffers[inputIndex], stream1, img_width, img_height);
        run_preprocess_gpu(resizedForModel, (float*)buffers[inputIndex], img_width, img_height, input_size, d_bgr, stream1);
        //cudaEventRecord(stop);
        //cudaEventSynchronize(stop);
        //cudaEventElapsedTime(&milliseconds, start, stop);
        //std::cout << "[Preprocess] " << milliseconds << " ms\n";
        
        // Event: inference
        //cudaEventRecord(start);
        context->enqueueV2(buffers, stream1, nullptr);
        //cudaEventRecord(stop);
        //cudaEventSynchronize(stop);
        //cudaEventElapsedTime(&milliseconds, start, stop);
        //cudaStreamSynchronize(stream1);
        //std::cout << "[Inference] " << milliseconds << " ms\n";

        // Event: post-process
        //cudaEventRecord(start, stream1);
        num_dets = run_postprocess_gpu(engine_output, num_anchors, conf_thresh, iou_thresh, max_detections, d_final, stream1, d_dets, d_compacted, d_mask, d_count_final);
        //cudaEventRecord(stop, stream1);
        //cudaEventSynchronize(stop);
        //cudaEventElapsedTime(&milliseconds, start, stop);
        //std::cout << "[Post-process] " << milliseconds << " ms\n";


        // Only do memory transfer if we have final detections
        if (num_dets > 0) {

            std::vector<Detection> detections(num_dets);

            cudaMemcpy(detections.data(), d_final, num_dets * sizeof(Detection), cudaMemcpyDeviceToHost);
         
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

        cv::imshow("Detections with bbox", frame);

        //uncomment to exit using ESC key instead of Ctrl+C
        if (cv::waitKey(1) == 27) break;


    }

    // Ensure all operations are complete before cleanup
    cudaDeviceSynchronize();  

    // Ensure profiler is stopped
    cudaProfilerStop();

    // Cleanup
    cap.release();
    cv::destroyAllWindows();
    cudaStreamDestroy(stream1);
    cudaFreeHost(pinned_frame_data);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
    cudaFree(d_bgr);
    cudaFree(d_final);
    cudaFree(d_dets);
    cudaFree(d_compacted);
    cudaFree(d_count_final);
    cudaFree(d_mask);
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}
