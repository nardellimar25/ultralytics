// assuming yolov8s.engine extracted from yolov8s.onnx generated with nms=False
#if !defined(__aarch64__)
#error "This build of yolodetector is Jetson-only (aarch64). Use the other branch for desktop."
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

// --- for the test ---
// #include <sys/stat.h>
// #include <sys/types.h>
// #include <unistd.h>
// #include <errno.h>
// #include <iomanip>
// #include <fstream>
// #include <sstream>
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
// Jetson NVMM + CUDA EGL interop
#include <cuda.h>
#include <cudaEGL.h>
#include "nvbufsurface.h"

// --- for the test ---        

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


// TEMP: global variables and kernels for NVMM EGL -> CUDA BGR copy
// will be moved to kernel files later
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

// Very simple RGBA -> BGR copy (no resize yet, 1:1 copy)
__global__ void rgba_to_bgr_linear_kernel(
    const unsigned char* __restrict__ src,
    int srcPitch,        // in bytes
    int width,
    int height,
    unsigned char* __restrict__ dst   // tightly packed BGR
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const unsigned char* srcRow = src + y * srcPitch + 4 * x;
    unsigned char r = srcRow[0];
    unsigned char g = srcRow[1];
    unsigned char b = srcRow[2];

    int dstIdx = (y * width + x) * 3;
    dst[dstIdx + 0] = b;
    dst[dstIdx + 1] = g;
    dst[dstIdx + 2] = r;
}

// Map NvBufSurface -> EGLImage -> CUDA and copy RGBA into d_bgr
static bool upload_nvmm_rgba_to_d_bgr(
    NvBufSurface* surf,
    int           index,       // plane index (usually 0)
    unsigned char* d_bgr,      // destination on device
    int           width,
    int           height,
    cudaStream_t  stream
)
{
    // Map for device access
    if (NvBufSurfaceMap(surf, index, 0, NVBUF_MAP_READ) != 0)
    {
        std::cerr << "[CUDA-EGL] NvBufSurfaceMap failed\n";
        return false;
    }

    if (NvBufSurfaceSyncForDevice(surf, index, 0) != 0)
    {
        std::cerr << "[CUDA-EGL] NvBufSurfaceSyncForDevice failed\n";
        NvBufSurfaceUnMap(surf, index, 0);
        return false;
    }

    // Map to EGLImage
    if (NvBufSurfaceMapEglImage(surf, index) != 0)
    {
        std::cerr << "[CUDA-EGL] NvBufSurfaceMapEglImage failed\n";
        NvBufSurfaceUnMap(surf, index, 0);
        return false;
    }

    EGLImageKHR eglImage = surf->surfaceList[index].mappedAddr.eglImage;
    if (!eglImage)
    {
        std::cerr << "[CUDA-EGL] eglImage is null\n";
        NvBufSurfaceUnMapEglImage(surf, index);
        NvBufSurfaceUnMap(surf, index, 0);
        return false;
    }

    CUgraphicsResource cuRes = nullptr;
    CUeglFrame eglFrame;

    CUresult r = cuGraphicsEGLRegisterImage(
        &cuRes,
        eglImage,
        CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
    if (r != CUDA_SUCCESS)
    {
        checkCu(r, "cuGraphicsEGLRegisterImage");
        NvBufSurfaceUnMapEglImage(surf, index);
        NvBufSurfaceUnMap(surf, index, 0);
        return false;
    }

    r = cuGraphicsResourceGetMappedEglFrame(
        &eglFrame,
        cuRes,
        0, 0);
    if (r != CUDA_SUCCESS)
    {
        checkCu(r, "cuGraphicsResourceGetMappedEglFrame");
        cuGraphicsUnregisterResource(cuRes);
        NvBufSurfaceUnMapEglImage(surf, index);
        NvBufSurfaceUnMap(surf, index, 0);
        return false;
    }

    if (eglFrame.frameType != CU_EGL_FRAME_TYPE_PITCH)
    {
        std::cerr << "[CUDA-EGL] Unexpected frameType (not PITCH)\n";
        cuGraphicsUnregisterResource(cuRes);
        NvBufSurfaceUnMapEglImage(surf, index);
        NvBufSurfaceUnMap(surf, index, 0);
        return false;
    }

    // unsigned char* srcDevPtr = static_cast<unsigned char*>(eglFrame.pPitch[0]);
    unsigned char* srcDevPtr = static_cast<unsigned char*>(eglFrame.frame.pPitch[0]);
    int srcPitch             = static_cast<int>(eglFrame.pitch);

    dim3 block(16, 16);
    dim3 grid(
        (width  + block.x - 1) / block.x,
        (height + block.y - 1) / block.y);

    rgba_to_bgr_linear_kernel<<<grid, block, 0, stream>>>(
        srcDevPtr,
        srcPitch,
        width,
        height,
        d_bgr);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "[CUDA-EGL] rgba_to_bgr_linear_kernel error: "
                  << cudaGetErrorString(err) << "\n";
    }

    cuGraphicsUnregisterResource(cuRes);
    NvBufSurfaceUnMapEglImage(surf, index);
    NvBufSurfaceUnMap(surf, index, 0);

    return (err == cudaSuccess);
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
    int num_cameras = 1;

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

    // check dimensions set inside context
    /*
    std::cout << "\n[YOLO engine after setBindingDimensions]\n";
    std::cout << "Input dims (ctx): ";
    for (int i = 0; i < inputDimsCtx.nbDims; ++i) {
        std::cout << inputDimsCtx.d[i];
        if (i < inputDimsCtx.nbDims - 1) std::cout << "x";
    }
    std::cout << std::endl;
    std::cout << "Output dims (ctx): ";
    for (int i = 0; i < outputDimsCtx.nbDims; ++i) {
        std::cout << outputDimsCtx.d[i];
        if (i < outputDimsCtx.nbDims - 1) std::cout << "x";
    }
    std::cout << std::endl;

    //return -1;
    */

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


    // ----------------------------- VIDEO CAPTURE SETUP (v4l2)----------------------------- //
    {
    // V4L2MMapCamera cam;
    // std::cout << "\nOpening camera via v4l2...\n";
    // if (!cam.openDevice("/dev/video0")) {
    //     std::cerr << "Error opening camera device\n";
    //     return -1;
    // }
    // std::cout << "->camera opened successfully\n";

    // std::cout << "\n[CAMERA SETTINGS]\n";
    // if (!cam.printCapabilities()) {
    //     std::cerr << "Error getting camera capabilities\n";
    //     return -1;
    // }
    // if (!cam.printCropCapabilities()) {
    //     // std::cerr << "Error getting camera crop capabilities\n";
    //     std::cerr << "Warning: camera does not support CROPCAP (VIDIOC_CROPCAP), continuing...\n";
    //     // return -1;
    // }

    // std::cout << "\nAvailable pixel formats:\n";
    // cam.listFormats(true);   // just to see what the driver supports

    // // For Jetson CSI IMX390 a typical mode is 1920x1080 NV12 or YUYV.
    // uint32_t req_width  = 1920;
    // uint32_t req_height = 1080;
    // uint32_t req_pixfmt = V4L2_PIX_FMT_RG12; // V4L2_PIX_FMT_NV12; //V4L2_PIX_FMT_YUYV; //V4L2_PIX_FMT_MJPEG;


    // std::cout << "\nInitializing camera...\n";
    // v4l2_cropcap crop{};
    // if (!cam.setFormat(req_width, req_height, req_pixfmt)) { //(1280, 720, V4L2_PIX_FMT_MJPEG)
    //     //std::cerr << "Error setting camera format\n";
    //     std::cerr << "Error setting camera format "
    //           << req_width << "x" << req_height
    //           << " " << V4L2MMapCamera::fourccToString(req_pixfmt) << "\n";
    //     return -1;
    // }
    // if (!cam.setFrameRate(30)) {
    //     std::cerr << "Error setting frame rate\n";
    // }
    // std::cout << "->camera initialized successfully\n";

    // int cap_width  = cam.width();
    // int cap_height  = cam.height();
    // int cap_channels = 3; // will be decoded to BGR
    // size_t pinned_size = cap_width * cap_height * cap_channels * sizeof(uchar);
    // uint32_t f = cam.pixfmt();

    // // Allocate pinned memory for the frame data
    // std::cout << "\nAllocating camera pinned memory buffers...\n";
    // if (!cam.initMMap(5)) {
    //     std::cerr << "Error initializing memory map\n";
    //     return -1;
    // }
    // std::cout << "->camera pinned memory buffers allocated successfully\n";

    // // Start the camera
    // std::cout << "\nStarting camera...\n";
    // if (!cam.start()) {
    //     std::cerr << "Error starting camera\n";
    //     return -1;
    // }
    // std::cout << "->camera started successfully\n\n";
    }



    // ----------------------------- VIDEO CAPTURE SETUP (Gstreamer)----------------------------- //

    std::cout << "\n[GST NVMM TEST] Opening camera via raw GStreamer (nvarguscamerasrc, NVMM RGBA)...\n";

    gst_init(nullptr, nullptr);

    // Pure NVMM pipeline: NV12 -> nvvidconv -> RGBA (NVMM) -> appsink
    const char* pipeline_desc =
        "nvarguscamerasrc sensor-id=0 bufapi-version=true ! "
        "video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, framerate=30/1, format=NV12 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw(memory:NVMM), format=RGBA ! "
        "appsink name=sink drop=true max-buffers=1 sync=false";

    GError* error = nullptr;
    GstElement* pipeline = gst_parse_launch(pipeline_desc, &error);
    if (!pipeline) {
        std::cerr << "[GST NVMM TEST] Failed to create pipeline: "
                  << (error ? error->message : "unknown error") << "\n";
        if (error) g_error_free(error);
        return -1;
    }

    GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
    if (!sink) {
        std::cerr << "[GST NVMM TEST] Failed to get appsink by name\n";
        gst_object_unref(pipeline);
        return -1;
    }

    gst_app_sink_set_emit_signals(GST_APP_SINK(sink), FALSE);
    gst_app_sink_set_drop(GST_APP_SINK(sink), TRUE);
    gst_app_sink_set_max_buffers(GST_APP_SINK(sink), 1);

    GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "[GST NVMM TEST] Failed to set pipeline to PLAYING\n";
        gst_object_unref(sink);
        gst_object_unref(pipeline);
        return -1;
    }

    std::cout << "[GST NVMM TEST] Pipeline running, grabbing a few frames...\n";

    // For now fix the capture size to what we requested
    int cap_width    = 1920;
    int cap_height   = 1080;
    int cap_channels = 3;   // final BGR
    size_t pinned_size = static_cast<size_t>(cap_width) * cap_height * cap_channels * sizeof(uchar);

    std::cout << "[CAMERA SETTINGS]\n";
    std::cout << "Resolution: " << cap_width << "x" << cap_height << "\n";
    std::cout << "Channels:   " << cap_channels << " (BGR, stored in d_bgr)\n\n";

    // GPU buffer for RGBA(NVMM) -> BGR
    unsigned char* d_bgr = nullptr;
    cudaMalloc(&d_bgr, static_cast<size_t>(cap_width) * cap_height * cap_channels);

    // Small CUDA stream just for this test
    cudaStream_t gst_stream;
    cudaStreamCreate(&gst_stream);

    // Debug window (optional)
    cv::namedWindow("GST_NVMM_TEST_FRAME", cv::WINDOW_AUTOSIZE);
    cv::Mat dbg_frame(cap_height, cap_width, CV_8UC3);

    int frame_idx = 0;
    bool running = true;

    while (running && frame_idx < 100) {
        GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
        if (!sample) {
            std::cerr << "[GST NVMM TEST] Failed to pull sample (EOS or error)\n";
            break;
        }

        GstCaps* caps = gst_sample_get_caps(sample);
        if (!caps) {
            std::cerr << "[GST NVMM TEST] Sample has no caps\n";
            gst_sample_unref(sample);
            break;
        }

        GstStructure* st = gst_caps_get_structure(caps, 0);
        int w = 0, h = 0;
        gst_structure_get_int(st, "width", &w);
        gst_structure_get_int(st, "height", &h);
        const gchar* fmt = gst_structure_get_string(st, "format");

        if (w != cap_width || h != cap_height) {
            std::cerr << "[GST NVMM TEST] Unexpected resolution " << w << "x" << h
                      << " (expected " << cap_width << "x" << cap_height << ")\n";
        }

        GstBuffer* buffer = gst_sample_get_buffer(sample);
        GstMapInfo map;
        if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            std::cerr << "[GST NVMM TEST] Failed to map buffer\n";
            gst_sample_unref(sample);
            break;
        }

        // map.data is NvBufSurface* because memory:NVMM
        NvBufSurface* surf = reinterpret_cast<NvBufSurface*>(map.data);

        // ---- GPU path: RGBA (NVMM) -> BGR in d_bgr ----
        if (!upload_nvmm_rgba_to_d_bgr(surf, 0,
                                       d_bgr,
                                       cap_width, cap_height,
                                       gst_stream)) {
            std::cerr << "[GST NVMM TEST] upload_nvmm_rgba_to_d_bgr failed\n";
        }

        // Debug: download from d_bgr to show it
        cudaMemcpyAsync(dbg_frame.data,
                        d_bgr,
                        static_cast<size_t>(cap_width) * cap_height * 3,
                        cudaMemcpyDeviceToHost,
                        gst_stream);
        cudaStreamSynchronize(gst_stream);
        cv::imshow("GST_NVMM_TEST_FRAME", dbg_frame);

        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);

        int key = cv::waitKey(1);
        if (key == 27) { // ESC
            running = false;
        }

        if ((frame_idx % 30) == 0) {
            std::cout << "[GST NVMM TEST] Frame " << frame_idx
                      << " | " << w << "x" << h
                      << " | format=" << (fmt ? fmt : "unknown") << "\n";
        }
        ++frame_idx;
    }

    cudaStreamSynchronize(gst_stream);
    cudaStreamDestroy(gst_stream);

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(sink);
    gst_object_unref(pipeline);
    cv::destroyWindow("GST_NVMM_TEST_FRAME");
    cudaFree(d_bgr);
    std::cout << "[GST NVMM TEST] Done. Exiting before full pipeline.\n";

    // TEMP:  exit here.
    // Next wire this into threaded pipeline
    return 0;




    // ----------------------------- MEMORY ALLOCATIONS ----------------------------- //
#if 0
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
    frame_back.create(cap_height, cap_width, CV_8UC3);
    frame_front.create(cap_height, cap_width, CV_8UC3);
    }


    // -------------------------------- START THREADS -------------------------------- //

    std::thread t_frame_capture(frame_capture_thread, std::ref(cap), cap_width, cap_height);

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
    //cam.stop();
    //cam.closeDevice();
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

#endif

    return 0;

}
