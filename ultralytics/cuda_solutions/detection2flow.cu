// assuming yolov8s.engine extracted from yolov8s.onnx generated with nms=False



#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>



using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cout << msg << std::endl;
    }
} gLogger;

// Structure to store detection
struct Detection {
    cv::Rect box;
    float score;
    int class_id;
};

float sigmoid(float x) {
    return 1.f / (1.f + expf(-x));
}

float iou(const cv::Rect& a, const cv::Rect& b) {
    int inter = (a & b).area();
    int uni = a.area() + b.area() - inter;
    return uni > 0 ? static_cast<float>(inter) / uni : 0.f;
}

// Read engine
ICudaEngine* loadEngine(const std::string& engineFile, IRuntime*& runtime) {
    std::ifstream file(engineFile, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open engine file");
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    runtime = createInferRuntime(gLogger);
    return runtime->deserializeCudaEngine(buffer.data(), size);
}

// Preprocess image to CHW normalized
void preprocessImage(const cv::Mat& img, float* gpuInput, cudaStream_t stream, const int img_width = 640, const int img_height = 640) {
    cv::Mat resized, rgb;
    cv::resize(img, resized, cv::Size(img_width, img_height));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    std::vector<float> chw(3 * img_width * img_height);
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < img_height; ++y)
            for (int x = 0; x < img_width; ++x)
                chw[c * img_height * img_width + y * img_width + x] = rgb.at<cv::Vec3b>(y, x)[c] / 255.0f;
    cudaMemcpyAsync(gpuInput, chw.data(), chw.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
    //rgb.convertTo(rgb, CV_32FC3, 1.0f / 255.0f); // normalization
    //cudaMemcpyAsync(gpuInput, rgb.data, rgb.total() * rgb.channels() * sizeof(float), cudaMemcpyHostToDevice, stream);
}

// Postprocess raw output from YOLOv8 nms=False
std::vector<Detection> postprocessYoloOutput_nmsFalse(
    const float* output, int num_anchors, int num_classes,
    float conf_thresh = 0.4f, float iou_thresh = 0.45f)      //0.5 e 0.45
{
    std::vector<Detection> detections;

    for (int i = 0; i < num_anchors; ++i) {
        float cx = output[0 * num_anchors + i];
        float cy = output[1 * num_anchors + i];
        float w  = output[2 * num_anchors + i];
        float h  = output[3 * num_anchors + i];

        int best_class = -1;
        float best_score = 0.f;
        for (int c = 0; c < num_classes; ++c) {
            float cls_score = output[(4 + c) * num_anchors + i];
            if (cls_score > best_score) {
                best_score = cls_score;
                best_class = c;
            }
        }

        /*if(i < 10){
        printf("Anchor %d: cx=%.2f, cy=%.2f, w=%.2f, h=%.2f, score=%.8f, class=%d\n",
               i, cx, cy, w, h, best_score, best_class);
        }*/

        if (best_score < conf_thresh) continue;
        if (best_class != 0) continue;

        
        int x1 = static_cast<int>(cx - w / 2);
        int y1 = static_cast<int>(cy - h / 2);
        int x2 = static_cast<int>(cx + w / 2);
        int y2 = static_cast<int>(cy + h / 2);

        detections.push_back({cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)), best_score, best_class});

    }

    // Apply NMS
    std::sort(detections.begin(), detections.end(), [](auto& a, auto& b) {
        return a.score > b.score;
    });

    std::vector<Detection> final_dets;
    std::vector<bool> suppressed(detections.size(), false);
    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;
        final_dets.push_back(detections[i]);
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (iou(detections[i].box, detections[j].box) > iou_thresh)
                suppressed[j] = true;
        }
    }

    return final_dets;
}


int conf_slider = 45; // maps to 0.25
int iou_slider = 45;  // maps to 0.45
float conf_thresh = 0.45f;
float iou_thresh = 0.45f;

void on_trackbar(int, void*) {
    conf_thresh = conf_slider / 100.0f;
    iou_thresh = iou_slider / 100.0f;
}


int main() {
    std::string engineFile = "./yoloengines/yolov8s_256_dockerborn.engine";

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
