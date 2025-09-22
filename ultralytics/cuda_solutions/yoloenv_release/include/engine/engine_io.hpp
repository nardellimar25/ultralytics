#pragma once
#include <cstddef>      
#include <NvInfer.h>

// Minimal, host-side descriptor for a TensorRT engine’s I/O.
// Works with TensorRT 8 (enqueueV2) since it carries void** bindings.
struct EngineIO {
  // Execution context + bindings for enqueueV2
  nvinfer1::IExecutionContext* ctx = nullptr;
  void** bindings = nullptr;

  // Dtypes of the input/output tensors
  nvinfer1::DataType inType  = nvinfer1::DataType::kFLOAT;
  nvinfer1::DataType outType = nvinfer1::DataType::kFLOAT;

  // Device pointers actually bound to TRT engine
  void* dIn  = nullptr;               // device buffer for input tensor
  void* dOut = nullptr;               // device buffer for output tensor

  // Element counts (not bytes) for input/output tensors
  std::size_t inElems  = 0;           
  std::size_t outElems = 0;
};
