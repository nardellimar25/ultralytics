cmake_minimum_required(VERSION 3.18)
project(YOLODetectCUDA LANGUAGES C CXX CUDA)

# ------------------------------------------------------------
# Toolchain & standards
# ------------------------------------------------------------
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Allow unsupported host compiler if you need it
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler --compiler-options -Wall")

# ------------------------------------------------------------
# Engine paths (defaults; can be overridden with -D…)
# ------------------------------------------------------------
if(NOT DEFINED YOLO_ENGINE_PATH)
    set(YOLO_ENGINE_PATH "engines/yolo.engine")
endif()
if(NOT DEFINED ACTION_ENGINE_PATH)
    set(ACTION_ENGINE_PATH "engines/action_cls_fp16.engine")
endif()

# ------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# TensorRT typical paths (adjust if needed)
include_directories(/usr/include/x86_64-linux-gnu)
include_directories(/usr/local/cuda/include)
# If you have a specific CUDA target path, keep it:
if(EXISTS /usr/local/cuda-12.3/targets/x86_64-linux/include)
  include_directories(/usr/local/cuda-12.3/targets/x86_64-linux/include)
endif()
link_directories(/usr/lib/x86_64-linux-gnu)

# ------------------------------------------------------------
# Project layout include paths
# ------------------------------------------------------------
set(PROJ_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(INC_DIR   ${PROJ_ROOT}/include)
set(SRC_DIR   ${PROJ_ROOT}/src)

include_directories(
    ${INC_DIR}
    ${INC_DIR}/cuda
    ${INC_DIR}/yolo
    ${INC_DIR}/action
    ${INC_DIR}/io
    ${INC_DIR}/engine
    ${INC_DIR}/cuda_frame_decode
    ${OpenCV_INCLUDE_DIRS}
)

# For the separate CUDA frame decoder module
include_directories(${PROJ_ROOT}/src/cuda_frame_decode)

# ------------------------------------------------------------
# Sources
# ------------------------------------------------------------
set(SOURCES
    # main
    ${SRC_DIR}/main/threaded_detections_main.cu

    # yolo
    ${SRC_DIR}/yolo/cuda_yolo_preprocess.cu
    ${SRC_DIR}/yolo/cuda_yolo_postprocess.cu
    ${SRC_DIR}/yolo/yolodetect.cpp

    # action
    ${SRC_DIR}/action/cuda_action_preprocess.cu
    ${SRC_DIR}/action/cuda_action_postprocess.cu

    # shared cuda infra
    ${SRC_DIR}/cuda/cuda_kernels.cu
    ${SRC_DIR}/cuda/cuda_threads.cu

    # io
    ${SRC_DIR}/io/v4l2_mmap_camera.cpp

    # cuda frame decode module (if present)
    ${PROJ_ROOT}/src/cuda_frame_decode/cuda_mjpeg_decode.cu
)

# ------------------------------------------------------------
# Target
# ------------------------------------------------------------
add_executable(yolodetector ${SOURCES})

# Macros visible to the code
target_compile_definitions(yolodetector PRIVATE
    HAVE_OPENCV
    YOLO_ENGINE_PATH="${YOLO_ENGINE_PATH}"
    ACTION_ENGINE_PATH="${ACTION_ENGINE_PATH}"
)

# Warnings
target_compile_options(yolodetector PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unknown-pragmas>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
)

# Link
target_link_libraries(yolodetector PRIVATE
    ${OpenCV_LIBS}
    nvinfer
    nvinfer_plugin
    cudart
    Threads::Threads
)

# CUDA props
set_target_properties(yolodetector PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "all"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
