// ================================================================================ //
// engine files must be extracted on the same architecture as the target deployment //
// ================================================================================ //
#if !defined(__aarch64__)
#error "This build of yolodetector is Jetson-only (aarch64). Use the other branch from git."
#endif

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
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <cuda.h>
#include <cudaEGL.h>
#include "nvbufsurface.h"


// --- for the test ---

// #include <sys/stat.h>
// #include <sys/types.h>
// #include <unistd.h>
// #include <errno.h>
// #include <iomanip>
// #include <fstream>
// #include <sstream>

// --- for the test ---        

#include "threads/cuda_threads.cuh"
#include "engine_io.hpp"
#include "engine_debug_utils.h"
#include "cuda_structs.cuh"
#include "yolodetect.h"


using namespace nvinfer1;


// Simple CUDA driver error checker
static void checkCu(CUresult r, const char* msg)
{
    if (r != CUDA_SUCCESS)
    {
        const char* errStr = nullptr;
        cuGetErrorString(r, &errStr);
        std::cerr << "[CUDA-EGL] " << msg << " failed: "
                  << (errStr ? errStr : "unknown") << " (" << r << ")\n";
    }
}


// ------------------------------ MAIN FUNCTION ------------------------------ //

int main() {

    // Start the profiler
    nvtxMark("start");
    cudaProfilerStart(); 

    std::cout << "\n[REAL-TIME DETECTION AND ACTION CLASSIFICATION]\n\n";
    std::cout << "\nTensorRT version:   " << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH << "\n";
    std::cout << "Permission warning fix:   sudo chmod 700 /run/user/1000" << "\n\n\n";

    // TEMP: number of cameras hard coded
    int num_cameras = 3;

    #if defined(__aarch64__)
    // Initialize CUDA Driver API for CUDA-EGL interop
    CUresult cuRes = cuInit(0);
    if (cuRes != CUDA_SUCCESS) {
        checkCu(cuRes, "cuInit");
        return -1;
    }
    #endif

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

    // Set concrete input shape on the CONTEXT
    nvinfer1::Dims4 inDims{num_cameras, 3, 256, 256};
    if (!context->setBindingDimensions(inputIndex, inDims)) {
        std::cerr << "\n->yolo setBindingDimensions failed\n";
        return -1;
    }
    if (!context->allInputDimensionsSpecified()) {
        std::cerr << "\n->yolo input dimensions not fully specified\n";
        return -1;
    }
    // Read CONCRETE dims from CONTEXT
    nvinfer1::Dims inputDimsCtx  = context->getBindingDimensions(inputIndex);
    nvinfer1::Dims outputDimsCtx = context->getBindingDimensions(outputIndex);

    // Initialize based on yolo engine dimensions
    const int yolo_engine_img_width     = inputDims.d[3];
    const int yolo_engine_img_height    = inputDims.d[2];
    const int yolo_num_anchors          = outputDims.d[2];
    const int yolo_num_classes          = 80;
    const int max_final_detections      = 100;
    const int yolo_inputSize            = inputDimsCtx.d[0] * inputDimsCtx.d[1] * yolo_engine_img_height * yolo_engine_img_width;
    const int yolo_outputSize           = outputDimsCtx.d[0] * outputDimsCtx.d[1] * yolo_num_anchors;
    // Get needed resized frame dimensions
    size_t yolo_engine_input_size       = yolo_engine_img_height * yolo_engine_img_width * 3;

    // Initialize yolo engine buffers
    void* yoloBuffers[2] = {nullptr, nullptr};
    cudaMalloc(&yoloBuffers[inputIndex],  static_cast<size_t>(yolo_inputSize)  * elemSize(inType));
    cudaMalloc(&yoloBuffers[outputIndex], static_cast<size_t>(yolo_outputSize) * elemSize(outType));

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


    // -------------------------- ACTION CLASSIFIER ENGINE -------------------------- //

    constexpr int CLS_MAX_BATCH = 64; // should match the number of the max batch size used during the engine extraction
    constexpr int CLS_OUT = 3;
    const size_t CLS_MAX_ELEMS = static_cast<size_t>(CLS_MAX_BATCH) * CLS_OUT;

    #ifndef ACTION_ENGINE_PATH
    #define ACTION_ENGINE_PATH "engines/cls.engine"
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

    // Print what TRT actually has
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


    // ----------------------------- MULTI CAMERA VIDEO CAPTURE SETUP (Gstreamer)----------------------------- //

    // Capture size
    int cap_width    = 1920;
    int cap_height   = 1080;
    int cap_channels = 3;
    size_t frame_bytes = static_cast<size_t>(cap_width) * cap_height * cap_channels;

    std::cout << "\n[CAMERA SETTINGS]\n";
    std::cout << "Resolution: " << cap_width << "x" << cap_height << "\n";
    std::cout << "Channels:   " << cap_channels << " (BGR)\n";

    std::cout << "\n[GST NVMM]\nInitializing GStreamer...\n";
    // Init GStreamer once
    gst_init(nullptr, nullptr);
    std::cout << "->GStreamer initialized successfully\n";

    // One pipeline + appsink per camera
    std::vector<GstElement*> pipelines(num_cameras, nullptr);
    std::vector<GstElement*> sinks(num_cameras, nullptr);

    std::cout << "\nCreating and starting pipelines for " << num_cameras << " cameras...\n";

    for (int cam = 0; cam < num_cameras; ++cam) {

        // Build a pipeline string with the right sensor-id and a unique sink name
        std::ostringstream oss;
        oss  << "nvarguscamerasrc sensor-id=" << cam << " ! "
            << "video/x-raw(memory:NVMM), width=(int)" << cap_width
            << ", height=(int)" << cap_height
            << ", framerate=30/1, format=NV12 ! "
            << "nvvidconv flip-method=0 ! "
            << "video/x-raw(memory:NVMM), format=RGBA ! "
            << "appsink name=sink" << cam
            << " drop=true max-buffers=1 sync=false";

        std::string pipeline_desc = oss.str();
        std::cout << "\nCreating pipeline for camera " << cam
                << " with desc:\n  " << pipeline_desc << "\n";

        GError* error = nullptr;
        pipelines[cam] = gst_parse_launch(pipeline_desc.c_str(), &error);
        if (!pipelines[cam]) {
            std::cerr << "->failed to create pipeline for camera " << cam
                    << ": " << (error ? error->message : "unknown error") << "\n";
            if (error) g_error_free(error);

            // Cleanup already-created pipelines before returning
            for (int j = 0; j < cam; ++j) {
                if (pipelines[j]) {
                    gst_element_set_state(pipelines[j], GST_STATE_NULL);
                    gst_object_unref(pipelines[j]);
                }
            }
            return -1;
        }

        // Get the appsink by its unique name
        std::string sink_name = "sink" + std::to_string(cam);
        sinks[cam] = gst_bin_get_by_name(GST_BIN(pipelines[cam]), sink_name.c_str());
        if (!sinks[cam]) {
            std::cerr << "[GST NVMM TEST] Failed to get appsink '" << sink_name
                    << "' for camera " << cam << "\n";
            gst_object_unref(pipelines[cam]);
            pipelines[cam] = nullptr;

            for (int j = 0; j < cam; ++j) {
                if (pipelines[j]) {
                    gst_element_set_state(pipelines[j], GST_STATE_NULL);
                    gst_object_unref(pipelines[j]);
                }
            }
            return -1;
        }

        gst_app_sink_set_emit_signals(GST_APP_SINK(sinks[cam]), FALSE);
        gst_app_sink_set_drop(GST_APP_SINK(sinks[cam]), TRUE);
        gst_app_sink_set_max_buffers(GST_APP_SINK(sinks[cam]), 1);

        GstStateChangeReturn ret = gst_element_set_state(pipelines[cam], GST_STATE_PLAYING);
        if (ret == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "->failed to set pipeline to PLAYING for camera " << cam << "\n";
            gst_object_unref(sinks[cam]);
            gst_object_unref(pipelines[cam]);
            sinks[cam]     = nullptr;
            pipelines[cam] = nullptr;

            for (int j = 0; j < cam; ++j) {
                if (pipelines[j]) {
                    gst_element_set_state(pipelines[j], GST_STATE_NULL);
                    gst_object_unref(pipelines[j]);
                }
            }
            return -1;
        }

        std::cout << "->pipeline running for camera " << cam << "\n";

    }

    std::cout << "\nAll pipelines running, grabbing frames...\n";



    // ----------------------------- MEMORY ALLOCATIONS ----------------------------- //


    // GPU buffer for RGBA(NVMM) -> BGR distorted frames
    unsigned char* d_bgr_raw = nullptr;
    cudaMalloc(&d_bgr_raw, frame_bytes * num_cameras);
    cudaMemset(d_bgr_raw, 0, frame_bytes * num_cameras);

    // Device buffer for undistorted BGR frames
    unsigned char* d_bgr_undistorted = nullptr;
    cudaMalloc(&d_bgr_undistorted, frame_bytes * num_cameras);

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
    cudaStream_t stream1, gst_stream, stream2;
    cudaStreamCreate(&stream1);
    // cudaStreamCreate(&stream2);
    cudaStreamCreate(&gst_stream);

    // Device classifier params from yolo bbox buffer
    ClsDevParams* d_cls_params = nullptr;
    cudaMalloc(&d_cls_params, CLS_MAX_BATCH * sizeof(ClsDevParams));    

    // Device classifier bbox crop buffer, one crop per detection
    unsigned char* d_cls_crop   = nullptr;
    cudaMalloc(&d_cls_crop, frame_bytes * CLS_MAX_BATCH);

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

    // event to signal frame readyness
    cudaEvent_t ev_frame_ready; 
    cudaEventCreateWithFlags(&ev_frame_ready, cudaEventDisableTiming);


    // --------------------------------- TESTING SPACE ------------------------------- //



    // ----------------------------- END OF TESTING SPACE ---------------------------- //

    
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
        int vis_width  = cap_width * num_cameras;
        int vis_height = cap_height;

        frame_back.create(vis_height, vis_width, CV_8UC3);
        frame_front.create(vis_height, vis_width, CV_8UC3);
    }


    // -------------------------------- START THREADS -------------------------------- //

    // Start frame capture thread
    std::thread t_frame_capture(
        multistream_frame_capture_thread,
        sinks.data(),       
        num_cameras,
        cap_width,
        cap_height,
        d_bgr_raw,
        d_bgr_undistorted,
        gst_stream,
        ev_frame_ready,
        pipelines.data() // for debugging the error states
    );

    // Start inference thread
    std::thread t_inference;
    if (clsIO.outType == nvinfer1::DataType::kFLOAT) {
        t_inference = std::thread(
            multi_stream_inference_thread_full_gpu,
            d_bgr_undistorted, d_resized,
            yolo_engine_img_width, yolo_engine_img_height,
            cap_width, cap_height,
            scaleX, scaleY, frame_bytes,
            stream1, stream2,
            yolo_num_anchors, conf_thresh, iou_thresh,
            d_final, d_dets, d_compacted, d_mask, 
            d_count_compact, d_count_final, h_count_final,
            yoloIO, clsIO,
            maxSquare,
            cls_img_width, cls_img_height,
            d_cls_crop, d_cls_square, d_cls_bgr96, d_cls_params,
            d_vis, h_vis,
            num_cameras,
            ev_frame_ready
        );
    } else {
        std::cout<<"\n[ERROR]\nUnexpected exception!\n->the output of the action classifier is fp16";
        keep_running = false;
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

    // opencv
    cv::destroyAllWindows();
    // gst
    for (int cam = 0; cam < num_cameras; ++cam) {
        if (pipelines[cam]) {
            gst_element_send_event(pipelines[cam], gst_event_new_eos());
        }
    }
    for (int cam = 0; cam < num_cameras; ++cam) {
        if (pipelines[cam]) {
            gst_element_set_state(pipelines[cam], GST_STATE_NULL);
        }
    }
    for (int cam = 0; cam < num_cameras; ++cam) {
        if (sinks[cam]) {
            gst_object_unref(sinks[cam]);
        }
        if (pipelines[cam]) {
            gst_object_unref(pipelines[cam]);
        }
    }
    // streams 
    cudaStreamDestroy(stream1);
    // cudaStreamDestroy(stream2);
    cudaStreamDestroy(gst_stream);
    // events
    cudaEventDestroy(ev_frame_ready);
    // memory
    // cudaFreeHost(pinned_frame_data);
    cudaFreeHost(h_count_final);
    cudaFree(yoloBuffers[inputIndex]); 
    cudaFree(yoloBuffers[outputIndex]);
    cudaFree(clsBuffers[clsInputIndex]);
    cudaFree(clsBuffers[clsOutputIndex]);
    cudaFree(d_bgr_raw);
    cudaFree(d_bgr_undistorted);
    cudaFree(d_final);
    cudaFree(d_dets);
    cudaFree(d_compacted);
    cudaFree(d_count_final);
    cudaFree(d_mask);
    cudaFree(d_count_compact);
    cudaFree(d_resized);
    cudaFree(d_cls_crop);
    cudaFree(d_cls_square);
    cudaFree(d_cls_bgr96);
    cudaFree(d_cls_params);
    cudaFree(d_vis);
    cudaFreeHost(h_vis);
    // engines and contexts
    context->destroy();
    clsContext->destroy();
    engine->destroy();
    clsEngine->destroy();
    runtime->destroy();
    std::cout << "->cleanup complete\n\n";



    return 0;

}
