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

// TODO : remove older version that created new runtime every time
// ICudaEngine* loadEngine(const std::string& engineFile, IRuntime*& runtime) {
//     std::ifstream file(engineFile, std::ios::binary);
//     if (!file) throw std::runtime_error("Failed to open engine file");
//     file.seekg(0, std::ios::end);
//     size_t size = file.tellg();
//     file.seekg(0);
//     std::vector<char> buffer(size);
//     file.read(buffer.data(), size);
//     runtime = createInferRuntime(gLogger);
//     return runtime->deserializeCudaEngine(buffer.data(), size);
// }

// Function to load a TensorRT engine from file
ICudaEngine* loadEngine(const std::string& engineFile, IRuntime*& runtime) {
    std::ifstream file(engineFile, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open engine file");

    file.seekg(0, std::ios::end);
    size_t size = (size_t)file.tellg();
    file.seekg(0);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);

    // Create runtime if not already created
    if (!runtime) {
        runtime = createInferRuntime(gLogger);
        if (!runtime) throw std::runtime_error("Failed to create TensorRT runtime");
    }

    return runtime->deserializeCudaEngine(buffer.data(), size);
}


void on_trackbar(int, void*) {
    conf_thresh = conf_slider / 100.0f;
    iou_thresh = iou_slider / 100.0f;
}
