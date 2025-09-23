# YOLOv8 Human Detection + Action Classifier (CUDA/TensorRT)

This project performs **real-time human detection and action classification** fully on the GPU.  
It combines a YOLOv8 detector with a second action classification model.  

The full pipeline runs end-to-end on the GPU:
1. **Preprocessing** 
2. **YOLOv8 TensorRT inference** 
3. **GPU Postprocessing** 
4. **Action Classifier**  
5. **Action Classifier TensorRT inference**
6. **Visualization Struct** 

---

## Features
- YOLOv8 TensorRT engine (exported with dynamic batch supported, for multistream capability)
- CUDA preprocessing kernels (FP32 + FP16, multibatch)
- CUDA postprocessing kernels (extract detections, compact, NMS, multibatch)
- Action classifier TensorRT engine (exported with dynamic batch supported, for multiple detection capability)
- CUDA Crop detection ROIs
- CUDA Pad to square and rescale
- CUDA Convert to grayscale float [0,1]
- CUDA profiling
- Deployable folder build (`make deploy`)
- Deployable folder package (`make package`)

---

## Directory Structure

yoloenv_release/
├─ include/ # C++/CUDA headers
├─ src/ # C++/CUDA sources
├─ CMakeLists.txt # CMake build configuration
├─ Makefile # Convenience build/package targets
├─ engines/ # (ignored) TensorRT engines
├─ build/ # (ignored) Build artifacts


`.gitignore` excludes:
    build/
    engines/
    .tar
    *.engine
    *.onnx


---

## Requirements
- To edit the repository:
    - NVIDIA GPU with recent driver
    - CUDA Toolkit (≥ 12.x recommended)
    - TensorRT (8.x or 10.x depending on platform)
    - OpenCV (debug visualization)
    - CMake ≥ 3.18, C++14 compiler

- To run the deploy folder:
    - NVIDIA GPU
    - Correct engines extracted on the deployment hardware


---

## Build with CMake and engine extractions
```bash
# Extract the yolo engine *on deploy target*
trtexec --onnx=yolov8s.onnx \
        --saveEngine=engines/yolo.engine \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:4x3x640x640 \
        --maxShapes=images:10x3x640x640 \
        --fp16

# Extract the action classifier engine *on deploy target*
trtexec --onnx=action_cls.onnx \
        --saveEngine=engines/action_cls.engine \
        --minShapes=input:1x96x96x1 \
        --optShapes=input:4x96x96x1 \
        --maxShapes=input:30x96x96x1 \

# Create deploy folder
cd yoloenv_release
make deploy

# Create package tar to export easily
make package

# Run from deploy folder
cd build/deploy
./run.sh

# Profile from deploy folder
cd build/deploy
./run_profile

```

---

## Pipeline Overview

```mermaid
flowchart TD
    A[Input Frames] --> B[Preprocessing (GPU)\nResize, BGR→RGB, Normalize, CHW]
    B --> C[YOLOv8 Engine (TensorRT)]
    C --> D[Postprocessing (GPU)\nConfidence filter, Compact, NMS]
    D --> E[ROI Preprocessing (GPU)\nCrop, Pad to square, Grayscale]
    E --> F[Action Classifier Engine (TensorRT)]
    F --> G[Visualization Struct\nCombine detections + actions]
```

---

## Pipeline Overview

[ Input Frames ]
        │
        ▼
┌────────────────────┐
│ Preprocessing (GPU)│
│ - Resize           │
│ - BGR→RGB          │
│ - Normalize, CHW   │
└────────────────────┘
        │
        ▼
┌────────────────────┐
│ YOLOv8 Engine (TRT)│
│ - Human detection  │
└────────────────────┘
        │
        ▼
┌────────────────────┐
│ Postprocessing(GPU)│
│ - Confidence filter│
│ - Compact + NMS    │
└────────────────────┘
        │
        ▼
┌───────────────────────────┐
│ ROI Preprocessing (GPU)   │
│ - Crop detections         │
│ - Pad to square           │
│ - Grayscale normalization │
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│ Action Classifier (TRT)   │
│ - Classify each detection │
└───────────────────────────┘
        │
        ▼
┌──────────────────────┐
│ Visualization Struct │
│ - Results combined   │
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│ Debug visualization  │
│ - if enabled         │
└──────────────────────┘    