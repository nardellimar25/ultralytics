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
#include <atomic>
#include <chrono>
#include <nvtx3/nvToolsExt.h>
#include <cuda_fp16.h>


#include "cuda_yolo_preprocess.cuh"
#include "cuda_yolo_postprocess.cuh"
#include "cuda_kernels.cuh"
#include "cuda_detection_struct.h"
#include "cuda_threads.cuh"
#include "v4l2_mmap_camera.h"
#include "yolodetect.h"
#include "engine_debug_utils.h"
#include "engine_io.hpp"
#include "cuda_action_preprocess.cuh"
#include "cuda_visual_struct.cuh"

#include <mutex>
#include <condition_variable>
#include <atomic>

using namespace nvinfer1;


int main() {

    // Start the profiler
    nvtxMark("start");
    cudaProfilerStart(); 

    std::cout << "\n[REAL-TIME DETECTION AND ACTION CLASSIFICATION]\n\n";
    std::cout << "\nTensorRT version:   " << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH << "\n";
    std::cout << "Permission warning fix:   sudo chmod 700 /run/user/1000" << "\n\n\n";

    // tmp
    int num_cameras = 1;


    // --------------------------------- YOLO ENGINE --------------------------------- //

    #ifndef YOLO_ENGINE_PATH
    #define YOLO_ENGINE_PATH "engines/yolo.engine"
    #endif

    std::string yolo_engineFile = YOLO_ENGINE_PATH;
    if (yolo_engineFile.empty()) {
        std::cerr << "Engine file path is empty. Please set correct ENGINE_PATH.\n";
        return -1;
    }
    std::cout << "Loading yolo engine from: " << yolo_engineFile << "...\n";
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = loadEngine(yolo_engineFile, runtime);
    if (!engine) {
        std::cerr << "Failed to load yolo engine at: " << yolo_engineFile << "\n";
        return -1;
    }
    std::cout << "->yolo engine loaded successfully\n";

    IExecutionContext* context = engine->createExecutionContext();
    const int inputIndex = engine->getBindingIndex("yolo_input");
    const int outputIndex = engine->getBindingIndex("yolo_output");
    nvinfer1::Dims inputDims = engine->getBindingDimensions(inputIndex);
    nvinfer1::Dims outputDims = engine->getBindingDimensions(outputIndex);
    auto inType  = engine->getTensorDataType("yolo_input");
    auto outType = engine->getTensorDataType("yolo_output");

    // Print engine info for yolo
    std::cout << "\n[YOLO engine]\n";
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
    std::cout << "\nInput dtype: "  << dtypeName(inType)  << "\n";
    std::cout << "Output dtype: " << dtypeName(outType) << "\n";
    std::cout << std::endl << std::endl;

    // Initialize based on yolo engine dimensions
    const int yolo_engine_img_width     = inputDims.d[3];
    const int yolo_engine_img_height    = inputDims.d[2];
    const int yolo_num_anchors          = outputDims.d[2];
    const int yolo_num_classes          = 80;
    const int max_final_detections      = 100;
    const int yolo_inputSize            = inputDims.d[0] * inputDims.d[1] * yolo_engine_img_height * yolo_engine_img_width;
    const int yolo_outputSize           = outputDims.d[0] * (yolo_num_classes + 4) * yolo_num_anchors;
    // Get needed resized frame dimensions
    size_t yolo_engine_input_size       = yolo_engine_img_height * yolo_engine_img_width * 3;

    // Initialize yolo engine buffers
    void* yoloBuffers[2] = {nullptr, nullptr};
    cudaMalloc(&yoloBuffers[inputIndex],  num_cameras * static_cast<size_t>(yolo_inputSize)  * elemSize(inType));
    cudaMalloc(&yoloBuffers[outputIndex], num_cameras * static_cast<size_t>(yolo_outputSize) * elemSize(outType));

    // Build YOLO EngineIO struct
    EngineIO yoloIO;
    yoloIO.ctx      = context;
    yoloIO.bindings = yoloBuffers;
    yoloIO.inType   = inType;                 
    yoloIO.outType  = outType;
    yoloIO.dIn      = yoloBuffers[inputIndex];
    yoloIO.dOut     = yoloBuffers[outputIndex];   
    yoloIO.inElems  = static_cast<size_t>(yolo_inputSize);  
    yoloIO.outElems = static_cast<size_t>(yolo_outputSize);  

    nvinfer1::Dims4 inDims{num_cameras, 3, 256, 256};
    if (!yoloIO.ctx->setBindingDimensions(0, inDims) || !yoloIO.ctx->allInputDimensionsSpecified()) {
        std::cerr << "\n[YOLO]\n->setBindingDimensions failed\n";
    }


    // -------------------------- ACTION CLASSIFIER ENGINE -------------------------- //

    // ---- constants for CLS ----
    constexpr int CLS_MAX_BATCH = 20;
    constexpr int CLS_OUT = 3; // classes
    const size_t CLS_MAX_ELEMS = static_cast<size_t>(CLS_MAX_BATCH) * CLS_OUT;

    #ifndef ACTION_ENGINE_PATH
    #define ACTION_ENGINE_PATH "engines/action_cls_fp16.engine"
    #endif

    // Get action classifier engine file path
    std::string action_cls_engineFile = ACTION_ENGINE_PATH;
    if (action_cls_engineFile.empty()) {
        std::cerr << "Engine file path is empty. Please set correct ACTION_ENGINE_PATH.\n";
        return -1;
    }
    std::cout << "Loading action classifier engine from: " << action_cls_engineFile << "...\n";
    ICudaEngine* clsEngine = loadEngine(action_cls_engineFile, runtime);
    if (!clsEngine) {
        std::cerr << "Failed to load classifier engine at: " << action_cls_engineFile << "\n";
        return -1;
    }
    std::cout << "->action classifier engine loaded successfully\n";

    IExecutionContext* clsContext = clsEngine->createExecutionContext();

    // === Print what TRT actually has ===
    //dumpBindings(clsEngine, "ACTION CLS engine");

    int clsInputIndex  = clsEngine->getBindingIndex("gray_images_input");
    int clsOutputIndex = clsEngine->getBindingIndex("action_cls_output");
    nvinfer1::Dims clsInputDims  = clsEngine->getBindingDimensions(clsInputIndex);
    nvinfer1::Dims clsOutputDims = clsEngine->getBindingDimensions(clsOutputIndex);
    auto clsInType  = clsEngine->getTensorDataType("gray_images_input");
    auto clsOutType = clsEngine->getTensorDataType("action_cls_output");

    // Print engine info for action classifier
    std::cout << "\n[ACTION CLS engine]\n";
    std::cout << "Engine name: " << clsEngine->getName() << std::endl;
    std::cout << "Number of bindings: " << clsEngine->getNbBindings() << std::endl;
    std::cout << "Input index: " << clsInputIndex << std::endl;
    std::cout << "Output index: " << clsOutputIndex << std::endl;
    std::cout << "Input dims: ";
    for (int i = 0; i < clsInputDims.nbDims; ++i) {
        std::cout << clsInputDims.d[i] << (i < clsInputDims.nbDims - 1 ? "x" : "");
    }
    std::cout << std::endl;
    std::cout << "Output dims: ";
    for (int i = 0; i < clsOutputDims.nbDims; ++i) {
        std::cout << clsOutputDims.d[i] << (i < clsOutputDims.nbDims - 1 ? "x" : "");
    }
    std::cout << "\nInput dtype: "  << dtypeName(clsInType)  << "\n";
    std::cout << "Output dtype: " << dtypeName(clsOutType) << "\n";
    std::cout << std::endl << std::endl;

    // Initialize based on action classifier engine dimensions
    const int cls_batch       = CLS_MAX_BATCH;  
    const int cls_img_height  = clsInputDims.d[1];   
    const int cls_img_width   = clsInputDims.d[2];   
    const int cls_channels    = clsInputDims.d[3];   
    const int cls_inputSize   = cls_channels * cls_img_height * cls_img_width;
    const int cls_out_batch   = CLS_MAX_BATCH;
    const int cls_num_classes = CLS_OUT; 
    const int cls_outputSize  = cls_out_batch * cls_num_classes;
    // Get needed resized frame dimensions
    size_t cls_engine_input_size = cls_img_height * cls_img_width * 1;

    // Initialize classifier engine buffers
    void* clsBuffers[2] = {nullptr, nullptr};
    cudaMalloc(&clsBuffers[clsInputIndex], CLS_MAX_BATCH * static_cast<size_t>(cls_inputSize) * elemSize(clsInType));
    cudaMalloc(&clsBuffers[clsOutputIndex], CLS_MAX_BATCH * static_cast<size_t>(cls_outputSize) * elemSize(clsOutType));

    // Build CLS EngineIO struct
    EngineIO clsIO;
    clsIO.ctx      = clsContext;
    clsIO.bindings = clsBuffers;
    clsIO.inType   = clsInType;
    clsIO.outType  = clsOutType;
    clsIO.dIn      = clsBuffers[clsInputIndex];
    clsIO.dOut     = clsBuffers[clsOutputIndex];
    clsIO.inElems  = static_cast<size_t>(cls_inputSize);
    clsIO.outElems = static_cast<size_t>(cls_outputSize);


    // ----------------------------- VIDEO CAPTURE SETUP ----------------------------- //

    V4L2MMapCamera cam;
    std::cout << "\nOpening camera...\n";
    if (!cam.openDevice("/dev/video0")) {
        std::cerr << "Error opening camera device\n";
        return -1;
    }
    std::cout << "->camera opened successfully\n";

    std::cout << "\n[CAMERA SETTINGS]\n";
    if (!cam.printCapabilities()) {
        std::cerr << "Error getting camera capabilities\n";
        return -1;
    }
    if (!cam.printCropCapabilities()) {
        std::cerr << "Error getting camera crop capabilities\n";
        return -1;
    }

    std::cout << "\nInitializing camera...\n";
    v4l2_cropcap crop{};
    if (!cam.setFormat(crop.bounds.width, crop.bounds.height, V4L2_PIX_FMT_MJPEG)) {
        std::cerr << "Error setting camera format\n";
        return -1;
    }
    if (!cam.setFrameRate(30)) {
        std::cerr << "Error setting frame rate\n";
    }
    std::cout << "->camera initialized successfully\n";

    int cap_width  = cam.width();
    int cap_height  = cam.height();
    int cap_channels = 3; // will be decoded to BGR
    size_t pinned_size = cap_width * cap_height * cap_channels * sizeof(uchar);
    uint32_t f = cam.pixfmt();

    // Allocate pinned memory for the frame data
    std::cout << "\nAllocating camera pinned memory buffers...\n";
    if (!cam.initMMap(5)) {
        std::cerr << "Error initializing memory map\n";
        return -1;
    }
    std::cout << "->camera pinned memory buffers allocated successfully\n";

    // Start the camera
    std::cout << "\nStarting camera...\n";
    if (!cam.start()) {
        std::cerr << "Error starting camera\n";
        return -1;
    }
    std::cout << "->camera started successfully\n\n";

    
    // ----------------------------- MEMORY ALLOCATIONS ----------------------------- //

    // Allocate pinned memory for the frame data
    uchar* pinned_frame_data = nullptr;
    cudaHostAlloc((void**)&pinned_frame_data, num_cameras * pinned_size, cudaHostAllocDefault);

    // Initialize yolo BGR image preprocessing buffer
    uchar* d_bgr = nullptr;
    cudaMalloc(&d_bgr, num_cameras * pinned_size);

    // Device yolo resized image preprocessing buffer
    uchar* d_resized = nullptr;
    cudaMalloc(&d_resized, num_cameras * yolo_engine_input_size);

    // Device yolo intermediate detections buffer
    Detection *d_dets;
    cudaMalloc(&d_dets, num_cameras * yolo_num_anchors * sizeof(Detection));

    // Device yolo mask buffer
    int *d_mask;
    cudaMalloc(&d_mask, num_cameras * yolo_num_anchors * sizeof(int));

    // Device yolo compacted detections buffer
    Detection* d_compacted;
    cudaMalloc(&d_compacted, num_cameras * yolo_num_anchors * sizeof(Detection));

    // Device yolo intermediate count buffer
    int* d_count_compact;
    cudaMalloc(&d_count_compact, sizeof(int));

    // Device yolo final detections results buffer
    Detection* d_final;
    cudaMalloc(&d_final, num_cameras * max_final_detections * sizeof(Detection));

    // Device yolo final count buffer
    int *d_count_final;
    cudaMalloc(&d_count_final, sizeof(int));

    // Host yolo final count buffer
    int *h_count_final = nullptr;
    cudaHostAlloc(&h_count_final, sizeof(int), cudaHostAllocDefault);

    // Initialize CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Device classifier params from yolo bbox buffer
    ClsDevParams* d_cls_params = nullptr;
    cudaMalloc(&d_cls_params, CLS_MAX_BATCH * sizeof(ClsDevParams));    

    // Device classifier bbox crop buffer, one crop per detection
    unsigned char* d_cls_crop   = nullptr;
    cudaMalloc(&d_cls_crop, pinned_size * CLS_MAX_BATCH);

    // Device classifier squared bbox buffer, one square per detection
    const int maxSquare  = std::max(cap_width, cap_height);
    unsigned char* d_cls_square = nullptr;
    cudaMalloc(&d_cls_square, CLS_MAX_BATCH* size_t(maxSquare) * maxSquare * 3);

    // Device classifier 96x96 BGR buffer, one per detection
    unsigned char* d_cls_bgr96  = nullptr;
    cudaMalloc(&d_cls_bgr96,  CLS_MAX_BATCH * cls_engine_input_size);

    // Device visualization output buffer 
    ActionVis* d_vis = nullptr;
    cudaMalloc(&d_vis, CLS_MAX_BATCH * sizeof(ActionVis));

    // Host-pinned memory for UI
    ActionVis* h_vis = nullptr;
    cudaMallocHost(&h_vis, CLS_MAX_BATCH * sizeof(ActionVis));

    //cudaEvent_t ev_cls_probs_ready; // to know when the D2H finished
    //cudaEventCreateWithFlags(&ev_cls_probs_ready, cudaEventDisableTiming);

    
    // ---------------------------- DEBUG INITIALIZATIONS ---------------------------- //

    // Create OpenCV windows and trackbars for debugging
    cv::namedWindow("Detections with bbox", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Conf Threshold", "Detections with bbox", nullptr, 100, on_trackbar);
    cv::createTrackbar("IoU Threshold", "Detections with bbox", nullptr, 100, on_trackbar);
    cv::setTrackbarPos("Conf Threshold", "Detections with bbox", conf_slider);
    cv::setTrackbarPos("IoU Threshold",  "Detections with bbox", iou_slider);
    on_trackbar(0, 0);

    // Calculate scale factors for bounding box coordinates
    float scaleX = static_cast<float>(cap_width) / yolo_engine_img_width;
    float scaleY = static_cast<float>(cap_height) / yolo_engine_img_height;


    // ------------------------------ SHARED VAR INITs ------------------------------ //

    {
    std::lock_guard<std::mutex> lk(frame_mutex);
    frame_back.create(cap_height, cap_width, CV_8UC3);
    frame_front.create(cap_height, cap_width, CV_8UC3);
    }


    // -------------------------------- START THREADS -------------------------------- //

    std::thread t_frame_capture(frame_capture_thread, std::ref(cam));
    std::thread t_inference;

    if (clsIO.outType == nvinfer1::DataType::kFLOAT) {
        t_inference = std::thread(inference_thread,
            pinned_frame_data, d_bgr, d_resized,
            yolo_engine_img_width, yolo_engine_img_height, cap_width, cap_height,
            scaleX, scaleY, pinned_size,
            stream1, stream2,
            yolo_num_anchors, conf_thresh, iou_thresh,
            d_final, d_dets, d_compacted, d_mask,
            d_count_compact, d_count_final, h_count_final,
            yoloIO, clsIO,
            /* act cls */
            maxSquare, cls_img_width, cls_img_height,
            d_cls_crop, d_cls_square, d_cls_bgr96, d_cls_params,
            d_vis, h_vis, 
            /* multi stream */
            num_cameras
        );
    } else {
        std::cout<<"\n[ERROR]\nUnexpected exception!\n->the output of the action classifier is fp16";
    }

    
    // ---------------------------- MAIN DISPLAY LOOP ---------------------------- //
    
    while (keep_running) {

        // Display the frame with bounding boxes
        {
        std::lock_guard<std::mutex> lock(frame_copy_mutex);
        if (!frame_front.empty())
            cv::imshow("Detections with bbox", frame_front);
        }

        // stop threads on ESC key
        if (cv::waitKey(1) == 27) {
            keep_running = false;       
            frame_ready.notify_all();   
            break;                      
        }

    }

    // Wait for threads to finish
    t_frame_capture.join();
    t_inference.join();
   
    // Ensure all operations are complete before cleanup
    cudaDeviceSynchronize();

    // Mark the end of the profiling range
    nvtxMark("stop");
    cudaProfilerStop();


    // ---------------------------------- CLEANUP ----------------------------------- //
    
    std::cout << "\nStopping and starting cleanup...\n";
    cam.stop();
    cam.closeDevice();
    cv::destroyAllWindows();
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFreeHost(pinned_frame_data);
    cudaFreeHost(h_count_final);
    cudaFree(yoloBuffers[inputIndex]); 
    cudaFree(yoloBuffers[outputIndex]);
    cudaFree(d_bgr);
    cudaFree(d_final);
    cudaFree(d_dets);
    cudaFree(d_compacted);
    cudaFree(d_count_final);
    cudaFree(d_mask);
    context->destroy();
    clsContext->destroy();
    engine->destroy();
    clsEngine->destroy();
    runtime->destroy();
    std::cout << "->cleanup complete\n\n";


    return 0;

}
