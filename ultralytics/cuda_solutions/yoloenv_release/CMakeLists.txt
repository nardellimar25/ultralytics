cmake_minimum_required(VERSION 3.18)
project(YOLODetectCUDA LANGUAGES C CXX CUDA)

# Allow unsupported host compiler
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")

# Set CUDA 12 compatibility
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)


# Allow passing ENGINE_PATH from command line, or fallback to default
if(NOT DEFINED ENGINE_PATH)
    set(ENGINE_PATH "engine/yolo.engine")
endif()



# Find OpenCV
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# TensorRT include and libs path
include_directories(/usr/include/x86_64-linux-gnu)
include_directories(/usr/local/cuda/include)
include_directories(/usr/local/cuda-12.3/targets/x86_64-linux/include)
link_directories(/usr/lib/x86_64-linux-gnu)

# Your sources
set(SOURCES
    cuda_kernels.cu
    cuda_preprocess.cu
    cuda_postprocess.cu
    cuda_threads.cu
    threaded_detections_main.cu
    yolodetect.cpp
    v4l2_mmap_camera.cpp

)

# Create executable
add_executable(yolodetector ${SOURCES})

# Engine path definition
target_compile_definitions(yolodetector PRIVATE ENGINE_PATH="${ENGINE_PATH}")


# Link libs
target_link_libraries(yolodetector
    ${OpenCV_LIBS}
    nvinfer
    nvinfer_plugin
    cudart
    Threads::Threads
)

# Use host g++ for C++ compilation
set_target_properties(yolodetector PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "all"
    CUDA_STANDARD 14
)


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -Wall")