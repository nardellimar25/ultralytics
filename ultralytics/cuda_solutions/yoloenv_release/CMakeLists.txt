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

# Allow unsupported host compiler if needed
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler --compiler-options -Wall")

# ------------------------------------------------------------
# Engine paths (defaults; can be overridden with -D…)
# ------------------------------------------------------------
if(NOT DEFINED YOLO_ENGINE_PATH)
  set(YOLO_ENGINE_PATH "engines/yolo.engine")
endif()
if(NOT DEFINED ACTION_ENGINE_PATH)
  set(ACTION_ENGINE_PATH "engines/cls.engine")
endif()

# ------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# -------------------- GStreamer setup (host vs aarch64 cross) --------------------
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message(STATUS "Configuring GStreamer for aarch64 cross-build")

    # Ensure we have a sysroot; if your toolchain already sets CMAKE_SYSROOT,
    # this will just reuse it.
    if(NOT CMAKE_SYSROOT)
        # Fallback; matches your build_cross_aarch64.sh script
        set(CMAKE_SYSROOT "/l4t/targetfs")
    endif()

    # Headers inside the Jetson sysroot
    set(GSTREAMER_INCLUDE_DIRS
        ${CMAKE_SYSROOT}/usr/include/gstreamer-1.0
        ${CMAKE_SYSROOT}/usr/include/glib-2.0
        ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/glib-2.0/include
        ${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu
    )

    # Libraries to link (names only; linker will look in sysroot lib paths)
    set(GSTREAMER_LIBRARIES
        gstreamer-1.0
        gstapp-1.0
        gobject-2.0
        glib-2.0
    )

else()
    # Normal host build: use pkg-config
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-app-1.0)

    set(GSTREAMER_INCLUDE_DIRS ${GST_INCLUDE_DIRS})
    set(GSTREAMER_LIBRARIES    ${GST_LIBRARIES})
endif()


# ---------------- Jetson Multimedia API (nvbufsurface) -----------------
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    # Se il toolchain file non ha già impostato CMAKE_SYSROOT,
    # riutilizziamo il fallback usato nello script build_cross_aarch64.sh
    if(NOT CMAKE_SYSROOT)
        set(CMAKE_SYSROOT "/l4t/targetfs")
    endif()

    set(JETSON_MMAPI_INCLUDE
        "${CMAKE_SYSROOT}/usr/src/jetson_multimedia_api/include"
    )

    message(STATUS "Jetson MMAPI include: ${JETSON_MMAPI_INCLUDE}")
endif()


# ---- TensorRT & CUDA hints (x86_64 vs aarch64) --------------
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  # Jetson
  set(TRT_HINT_INC
      /usr/include/aarch64-linux-gnu
      /usr/local/cuda/include
      /usr/include)
  set(TRT_HINT_LIB
      /usr/lib/aarch64-linux-gnu
      /usr/local/cuda/targets/aarch64-linux/lib
      /usr/local/cuda/lib64
      /usr/lib)
else()
  # Desktop
  set(TRT_HINT_INC
      /usr/include/x86_64-linux-gnu
      /usr/local/cuda/include
      /usr/include)
  set(TRT_HINT_LIB
      /usr/lib/x86_64-linux-gnu
      /usr/local/cuda/targets/x86_64-linux/lib
      /usr/local/cuda/lib64
      /usr/lib)
endif()

find_path(TRT_INCLUDE_DIR NvInfer.h HINTS ${TRT_HINT_INC} REQUIRED)
find_library(TRT_LIB_NVINFER nvinfer HINTS ${TRT_HINT_LIB} REQUIRED)
find_library(TRT_LIB_PLUGIN  nvinfer_plugin HINTS ${TRT_HINT_LIB} REQUIRED)
find_library(TRT_LIB_CUDART  cudart HINTS ${TRT_HINT_LIB} REQUIRED)

message(STATUS "TensorRT include: ${TRT_INCLUDE_DIR}")
message(STATUS "TensorRT libs: ${TRT_LIB_NVINFER}; ${TRT_LIB_PLUGIN}; ${TRT_LIB_CUDART}")


# -------- nvbufsurface (Jetson buffer surface API) ----------
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    find_library(NVBUFSURF_LIB nvbufsurface
        HINTS
            ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu
            ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/tegra
    )
    message(STATUS "nvbufsurface lib: ${NVBUFSURF_LIB}")
    # -------- CUDA Driver + CUDA-EGL (for cuInit, cuGraphicsEGL...) ----------
    find_library(CUDA_DRIVER_LIB cuda
        HINTS
            ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu
            ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/tegra
            /usr/local/cuda/targets/aarch64-linux/lib
            /usr/local/cuda/targets/aarch64-linux/lib/stubs
    )

    message(STATUS "CUDA driver lib: ${CUDA_DRIVER_LIB}")

    find_library(CUDA_EGL_LIB cudaEGL
        NAMES cudaEGL cudaegl
        HINTS
            ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu
            ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/tegra
            /usr/local/cuda/targets/aarch64-linux/lib
            /usr/local/cuda/targets/aarch64-linux/lib/stubs
    )

    if(CUDA_EGL_LIB)
        message(STATUS "CUDA EGL lib:    ${CUDA_EGL_LIB}")
    else()
        message(WARNING "cudaEGL library not found in sysroot; will fall back to H2D memcpy for now.")
    endif()

endif()



# ------------------------------------------------------------
# Project layout include paths
# ------------------------------------------------------------
set(PROJ_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(INC_DIR   ${PROJ_ROOT}/include)
set(SRC_DIR   ${PROJ_ROOT}/src)

include_directories(
  ${INC_DIR}
  ${INC_DIR}/action
  ${INC_DIR}/cuda
  ${INC_DIR}/cuda/kernels
  ${INC_DIR}/cuda/threads
  ${INC_DIR}/cuda_frame_decode
  ${INC_DIR}/engine
  ${INC_DIR}/io
  ${INC_DIR}/yolo

  ${OpenCV_INCLUDE_DIRS}
  ${TRT_INCLUDE_DIR}
  ${GSTREAMER_INCLUDE_DIRS}        
  ${JETSON_MMAPI_INCLUDE}  
  /usr/local/cuda/include
)

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

  # CUDA gpu kernels & threads
  ${SRC_DIR}/cuda/kernels/cls_kernels.cu
  ${SRC_DIR}/cuda/kernels/cuda_kernels.cu
  ${SRC_DIR}/cuda/kernels/frame_kernels.cu
  ${SRC_DIR}/cuda/kernels/yolo_kernels.cu
  ${SRC_DIR}/cuda/threads/cuda_threads_capture.cu
  ${SRC_DIR}/cuda/threads/cuda_threads_inference.cu

  # io TODO: delete old v4l2 read camera
  ${SRC_DIR}/io/v4l2_mmap_camera.cpp

  # cuda frame decode/nvbufsurface
  ${PROJ_ROOT}/src/cuda_frame_decode/cuda_mjpeg_decode.cu
  ${PROJ_ROOT}/src/cuda_frame_decode/cuda_undistort.cu
  ${PROJ_ROOT}/src/cuda_frame_decode/cuda_nvmm_egl_upload.cu
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
  DEBUG_VIS=1
  DEBUG_ONLY_FRAME=0
)

# // Debug visualization flags
# // DEBUG_VIS | DEBUG_ONLY_FRAME
# // 0 0 -> no window displayed
# // 1 0 -> frame + bboxes + blur
# // 1 1 -> only frame

# Warnings
target_compile_options(yolodetector PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unknown-pragmas>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
)

# ------------------------------------------------------------
# Extra Jetson-only libs (only if they were actually found)
# ------------------------------------------------------------
set(EXTRA_JETSON_LIBS "")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    if(NVBUFSURF_LIB)
        list(APPEND EXTRA_JETSON_LIBS ${NVBUFSURF_LIB})
    endif()
    if(CUDA_DRIVER_LIB)
        list(APPEND EXTRA_JETSON_LIBS ${CUDA_DRIVER_LIB})
    endif()
    if(CUDA_EGL_LIB)
        list(APPEND EXTRA_JETSON_LIBS ${CUDA_EGL_LIB})
    endif()
endif()


# Link (use found TRT libs + GStreamer)
target_link_libraries(yolodetector PRIVATE
  ${OpenCV_LIBS}
  ${TRT_LIB_NVINFER}
  ${TRT_LIB_PLUGIN}
  ${TRT_LIB_CUDART}
  Threads::Threads
  ${GSTREAMER_LIBRARIES}            
  ${EXTRA_JETSON_LIBS}                 
)

# On Jetson cross-compile we may not have DLA / cudla libs available.
# TensorRT's libnvinfer.so has optional references to those symbols;
# allow unresolved symbols in shared libs when linking yolodetector.
if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    target_link_options(yolodetector PRIVATE -Wl,--allow-shlib-undefined)
endif()

# ------------------------------------------------------------
# CUDA props
# ------------------------------------------------------------
# Override with -DGPU_SM=72 (Xavier), 87 (Orin), 53 (Nano), etc.
if(NOT DEFINED GPU_SM)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(GPU_SM 87)  # sensible Jetson default (Orin). Override for other SoCs.
  else()
    set(GPU_SM all) # desktop default
  endif()
endif()

set_target_properties(yolodetector PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES "${GPU_SM}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  BUILD_RPATH "\$ORIGIN/../lib"  # allow running directly if libs sit next to exe
)

message(STATUS "CUDA_ARCHITECTURES set to: ${GPU_SM}")
message(STATUS "YOLO_ENGINE_PATH: ${YOLO_ENGINE_PATH}")
message(STATUS "ACTION_ENGINE_PATH: ${ACTION_ENGINE_PATH}")
