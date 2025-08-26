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
#include <nvtx3/nvToolsExt.h>

#include "cuda_preprocess.cuh"
#include "cuda_postprocess.cuh"
#include "cuda_kernels.cuh"
#include "cuda_detection_struct.h"
#include "cuda_threads.cuh"
#include "yolodetect.h"

#include <mutex>
#include <condition_variable>
#include <atomic>

using namespace nvinfer1;

int main() {

    // Start the profiler to measure performance
    nvtxMark("start");
    cudaProfilerStart();  

    // Get engine file path
    std::string engineFile = ENGINE_PATH;
    if (engineFile.empty()) {
        std::cerr << "Engine file path is empty. Please set correct ENGINE_PATH.\n";
        return -1;
    }
    // Load the TensorRT engine
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = loadEngine(engineFile, runtime);
    IExecutionContext* context = engine->createExecutionContext();
    const int inputIndex = engine->getBindingIndex("images");
    const int outputIndex = engine->getBindingIndex("output0");
    nvinfer1::Dims inputDims = engine->getBindingDimensions(inputIndex);
    nvinfer1::Dims outputDims = engine->getBindingDimensions(outputIndex);
    // Print engine info
    std::cout << "\n[Engine loaded successfully]\n";
    std::cout << "Engine name: " << engine->getName() << std::endl;
    std::cout << "Number of bindings: " << engine->getNbBindings() << std::endl;
    std::cout << "Input index: " << inputIndex << ", Output index: " << outputIndex << std::endl;
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
    const int engine_img_width = inputDims.d[3];
    const int engine_img_height = inputDims.d[2];
    const int num_classes = 80;
    const int num_anchors = outputDims.d[2];
    const int max_detections = 100;
    const int inputSize = inputDims.d[0] * inputDims.d[1] * engine_img_height * engine_img_width;
    const int outputSize = outputDims.d[0] * (num_classes + 4) * num_anchors;
    // Get needed resized frame dimensions
    size_t engine_input_size = engine_img_height * engine_img_width * 3;
    // Initialize number of detections
    //int num_dets = 0;

    // Initialize engine buffers
    void* buffers[2];
    cudaMalloc(&buffers[inputIndex], inputSize * sizeof(float));
    cudaMalloc(&buffers[outputIndex], outputSize * sizeof(float));
    
    // Create alias for input and output engine buffers
    float* engine_input = static_cast<float*>(buffers[inputIndex]);
    float* engine_output = static_cast<float*>(buffers[outputIndex]);

    // Initialize device intermediate detections buffer
    Detection *d_dets;
    cudaMalloc(&d_dets, num_anchors * sizeof(Detection));

    // Initialize device compacted detections buffer
    Detection* d_compacted;
    cudaMalloc(&d_compacted, max_detections * sizeof(Detection));

    // Initialize device final detections results buffer
    Detection* d_final;
    cudaMalloc(&d_final, max_detections * sizeof(Detection));
    
    // Initialize device resized image preprocessing buffer
    uchar* d_resized = nullptr;
    cudaMalloc(&d_resized, engine_input_size);

    // Initialize device final count buffer
    int *d_count_final;
    cudaMalloc(&d_count_final, sizeof(int));

    // Initialize device mask buffer
    int *d_mask;
    cudaMalloc(&d_mask, num_anchors * sizeof(int));

    // Initialize host final count buffer
    int *h_count_final = nullptr;
    cudaHostAlloc(&h_count_final, sizeof(int), cudaHostAllocDefault);

    // Initialize CUDA stream
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);


    // Initialize OpenCV VideoCapture
    cv::VideoCapture cap("/dev/video0", cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::cerr << "Impossibile aprire webcam\n";
        return -1;
    }
    // Get original cap size
    const int cap_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const int cap_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    const int cap_channels = 3; // assume BGR
    // Allocate pinned memory for the frame data
    uchar* pinned_frame_data = nullptr;
    size_t pinned_size = cap_width * cap_height * cap_channels * sizeof(uchar);
    cudaHostAlloc((void**)&pinned_frame_data, pinned_size, cudaHostAllocDefault);

    // Create Mat to hold the frame data using pinned memory
    frame_back = cv::Mat(cap_height, cap_width, CV_8UC3, pinned_frame_data);
    frame_front = frame_back.clone();
    cap.read(frame_back);
    if (frame_back.empty()) {
        std::cerr << "Frame vuoto\n";
        return -1;
    }

    // Calculate scale factors for bounding box coordinates
    float scaleX = static_cast<float>(cap_width) / engine_img_width;
    float scaleY = static_cast<float>(cap_height) / engine_img_height;

    // Initialize BGR image preprocessing buffer
    uchar* d_bgr = nullptr;
    cudaMalloc(&d_bgr, pinned_size);


    // Create OpenCV windows and trackbars
    cv::namedWindow("Detections with bbox", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Conf Threshold", "Detections with bbox", &conf_slider, 100, on_trackbar);
    cv::createTrackbar("IoU Threshold", "Detections with bbox", &iou_slider, 100, on_trackbar);
    on_trackbar(0, 0);



    std::thread t_frame_capture(frame_capture_thread, std::ref(cap));
    std::thread t_inference(
        inference_thread, 
        engine_input, pinned_frame_data, d_bgr, d_resized, 
        engine_img_width, engine_img_height, cap_width, cap_height, 
        scaleX, scaleY, pinned_size, stream1, stream2,

        context, buffers,

        engine_output, num_anchors, conf_thresh, iou_thresh,                        
        d_final, d_dets, d_compacted, d_mask, 
        d_count_final, h_count_final
    );
    

    while (keep_running) {


        {
        std::lock_guard<std::mutex> lock(frame_copy_mutex);
        if (!frame_front.empty())
            cv::imshow("Detections with bbox", frame_front);
        }

        // stop threads on ESC key
        if (cv::waitKey(1) == 27) {
            keep_running = false;       //  Tell both threads to exit
            frame_ready.notify_all();   //  Wake inference thread if waiting
            break;                      //  Exit loop
        }
        nvtxRangePop();
        
    }

    // Wait for threads to finish
    t_frame_capture.join();
    t_inference.join();     
   
    // Ensure all operations are complete before cleanup
    cudaDeviceSynchronize();

    // Mark the end of the profiling range
    nvtxMark("stop");

    // Ensure profiler is stopped
    cudaProfilerStop();

    // Cleanup
    cap.release();
    cv::destroyAllWindows();
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFreeHost(pinned_frame_data);
    cudaFreeHost(h_count_final);
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
