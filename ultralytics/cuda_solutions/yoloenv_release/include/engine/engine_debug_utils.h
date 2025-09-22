#pragma once
#include <NvInfer.h>
#include <string>
#include <sstream>
#include <iostream>
#include <utility>

// Pretty-print a TRT dtype
static inline const char* dtypeName(nvinfer1::DataType t) {
    using nvinfer1::DataType;
    switch (t) {
        case DataType::kFLOAT: return "FLOAT";
        case DataType::kHALF:  return "HALF";
        case DataType::kINT8:  return "INT8";
        case DataType::kINT32: return "INT32";
        case DataType::kBOOL:  return "BOOL";
        // Some TRT versions add more types (e.g., BF16). Guard them:
        #if NV_TENSORRT_MAJOR >= 10
        case DataType::kBF16:  return "BF16";
        #endif
        default: return "UNKNOWN";
    }
}

// Size in bytes of a TRT dtype
static inline size_t elemSize(nvinfer1::DataType t) {
    using nvinfer1::DataType;
    switch (t) {
        case DataType::kFLOAT: return 4;
        case DataType::kHALF:  return 2;
        case DataType::kINT32: return 4;
        case DataType::kINT8:  return 1;
        case DataType::kBOOL:  return 1;
        #if NV_TENSORRT_MAJOR >= 10
        case DataType::kBF16:  return 2;
        #endif
        default: return 0;
    }
}

// --- Pretty-print a Dims ---
static inline std::string dimsToStr(const nvinfer1::Dims& d) {
    std::ostringstream os; os << "[";
    for (int i = 0; i < d.nbDims; ++i) { os << d.d[i]; if (i + 1 < d.nbDims) os << "x"; }
    os << "]";
    return os.str();
}

// --- Dump all bindings (names, dims, dtypes) ---
static inline void dumpBindings(const nvinfer1::ICudaEngine* e, const char* tag = "TRT Engine") {
    std::cout << "\n=== " << tag << " ===\n";
    std::cout << "name: " << e->getName()
              << " | profiles: " << e->getNbOptimizationProfiles()
              << " | bindings: " << e->getNbBindings() << "\n";
    for (int i = 0; i < e->getNbBindings(); ++i) {
        const bool isIn = e->bindingIsInput(i);
        auto dims = e->getBindingDimensions(i);
        auto dt   = e->getBindingDataType(i);
        std::cout << "  [" << i << "] " << (isIn ? "INPUT  " : "OUTPUT ")
                  << "\"" << e->getBindingName(i) << "\""
                  << " dims=" << dimsToStr(dims)
                  << " dtype=" << dtypeName(dt) << "\n";
    }
    std::cout << "=======================\n";
}

// --- Simple finders ---
static inline int findFirstInput(const nvinfer1::ICudaEngine* e) {
    for (int i = 0; i < e->getNbBindings(); ++i) if (e->bindingIsInput(i)) return i;
    return -1;
}
static inline int findFirstOutput(const nvinfer1::ICudaEngine* e) {
    for (int i = 0; i < e->getNbBindings(); ++i) if (!e->bindingIsInput(i)) return i;
    return -1;
}
static inline int findBindingByName(const nvinfer1::ICudaEngine* e, const char* name) {
    if (!name) return -1;
    return e->getBindingIndex(name); // returns -1 if not found
}

// --- Shape-based finders (NHWC input, classes=3 output) ---
static inline int findNHWCInput(const nvinfer1::ICudaEngine* e, int H, int W, int C) {
    for (int i = 0; i < e->getNbBindings(); ++i) {
        if (!e->bindingIsInput(i)) continue;
        auto d = e->getBindingDimensions(i);
        if (d.nbDims == 4 && d.d[1] == H && d.d[2] == W && d.d[3] == C) return i; // N,H,W,C (N may be -1)
    }
    return -1;
}
static inline int findClassesOutput(const nvinfer1::ICudaEngine* e, int numClasses) {
    for (int i = 0; i < e->getNbBindings(); ++i) {
        if (e->bindingIsInput(i)) continue;
        auto d = e->getBindingDimensions(i);
        if (d.nbDims == 2 && d.d[1] == numClasses) return i; // [N, numClasses]
    }
    return -1;
}

// --- Resolve input/output robustly (prefer names, fall back to shapes) ---
static inline std::pair<int,int> resolveClsBindings(
    const nvinfer1::ICudaEngine* e,
    const char* preferredInputName,
    const char* preferredOutputName,
    int H = 96, int W = 96, int C = 1, int numClasses = 3)
{
    int in  = findBindingByName(e, preferredInputName);
    int out = findBindingByName(e, preferredOutputName);

    if (in  < 0) in  = findNHWCInput(e, H, W, C);
    if (out < 0) out = findClassesOutput(e, numClasses);

    if (in  < 0) in  = findFirstInput(e);
    if (out < 0) out = findFirstOutput(e);

    return {in, out};
}

// --- Set runtime dims for dynamic batch NHWC and verify ---
static inline bool setDimsNHWC(nvinfer1::IExecutionContext* ctx, int inputIndex,
                               int N, int H, int W, int C)
{
    nvinfer1::Dims4 d{N, H, W, C};
    if (!ctx->setBindingDimensions(inputIndex, d)) {
        std::cerr << "[TRT] setBindingDimensions failed for input " << inputIndex
                  << " dims=" << dimsToStr(d) << "\n";
        return false;
    }
    if (!ctx->allInputDimensionsSpecified()) {
        std::cerr << "[TRT] Not all input dimensions specified after setBindingDimensions\n";
        return false;
    }
    return true;
}



