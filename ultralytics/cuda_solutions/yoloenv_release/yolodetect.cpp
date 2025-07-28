#include "yolodetect.h"

// Global variable definitions
int conf_slider = 45;
int iou_slider = 45;
float conf_thresh = 0.45f;
float iou_thresh = 0.45f;

Logger gLogger;

// ---- Function definitions ----

void Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) std::cout << msg << std::endl;
}

float sigmoid(float x) {
    return 1.f / (1.f + expf(-x));
}

/*float iou(const cv::Rect& a, const cv::Rect& b) {
    int inter = (a & b).area();
    int uni = a.area() + b.area() - inter;
    return uni > 0 ? static_cast<float>(inter) / uni : 0.f;
}*/

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

void preprocessImage(const cv::Mat& img, float* gpuInput, cudaStream_t stream, const int img_width, const int img_height) {
    cv::Mat resized, rgb;
    cv::resize(img, resized, cv::Size(img_width, img_height));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    std::vector<float> chw(3 * img_width * img_height);
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < img_height; ++y)
            for (int x = 0; x < img_width; ++x)
                chw[c * img_height * img_width + y * img_width + x] = rgb.at<cv::Vec3b>(y, x)[c] / 255.0f;
    cudaMemcpyAsync(gpuInput, chw.data(), chw.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
}

/*std::vector<Detection> postprocessYoloOutput_nmsFalse(const float* output, int num_anchors, int num_classes, float conf_thresh, float iou_thresh) {
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

        if (best_score < conf_thresh || best_class != 0) continue;

        int x1 = static_cast<int>(cx - w / 2);
        int y1 = static_cast<int>(cy - h / 2);
        int x2 = static_cast<int>(cx + w / 2);
        int y2 = static_cast<int>(cy + h / 2);

        detections.push_back({cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)), best_score, best_class});
    }

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
}*/

void on_trackbar(int, void*) {
    conf_thresh = conf_slider / 100.0f;
    iou_thresh = iou_slider / 100.0f;
}
