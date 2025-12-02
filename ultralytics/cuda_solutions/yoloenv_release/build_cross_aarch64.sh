#!/bin/bash
set -e

echo "============================"
echo "  YOLO AARCH64 CROSS BUILD"
echo "============================"

# -------------------------------------
# 1) Ensure environment
# -------------------------------------
export TRT_LIB_DIR=/usr/lib/aarch64-linux-gnu
SYSROOT=/l4t/targetfs
OCV_DIR="${SYSROOT}/usr/lib/aarch64-linux-gnu/cmake/opencv4"

if [ ! -d "$SYSROOT/usr" ]; then
    echo "❌ ERROR: Jetson sysroot not extracted!"
    echo "Run inside container:"
    echo "   cd /l4t && tar xf toolchain.tar.gz && tar xf targetfs.tbz2"
    exit 1
fi

echo "Using sysroot: $SYSROOT"
echo "Using OpenCV_DIR: $OCV_DIR"

# -------------------------------------
# 2) Clean & configure
# -------------------------------------
cd /workspace/yoloenv_release
rm -rf build

make TARGET_ARCH=aarch64 \
     JETSON=1 \
     JETSON_BUNDLED_TRT=1 \
     CMAKE_OPTS="-DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-jetson.cmake \
                 -DGPU_SM=72 \
                 -DOpenCV_DIR=${OCV_DIR}" \
     deploy

# -------------------------------------
# 3) Add ARM OpenCV runtime libs
# -------------------------------------
DEPLOY_DIR=build/deploy
cd "$DEPLOY_DIR"

echo ">>> Bundling OpenCV 4.5 ARM libs..."
mkdir -p lib

cp ${SYSROOT}/usr/lib/aarch64-linux-gnu/libopencv_core.so.4.5*      lib/ || true
cp ${SYSROOT}/usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.5*   lib/ || true
cp ${SYSROOT}/usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.5* lib/ || true
cp ${SYSROOT}/usr/lib/aarch64-linux-gnu/libopencv_highgui.so.4.5*   lib/ || true
cp ${SYSROOT}/usr/lib/aarch64-linux-gnu/libopencv_videoio.so.4.5*   lib/ || true
cp ${SYSROOT}/usr/lib/aarch64-linux-gnu/libopencv_video.so.4.5*     lib/ || true
cp ${SYSROOT}/usr/lib/aarch64-linux-gnu/libopencv_calib3d.so.4.5*   lib/ || true
echo ">>> Bundling Jetson nvbufsurface / CUDA-EGL libs (best-effort)..."
cp ${SYSROOT}/usr/lib/aarch64-linux-gnu/tegra/libnvbufsurface.so* lib/ || true
cp ${SYSROOT}/usr/lib/aarch64-linux-gnu/tegra/libcudaEGL.so*      lib/ || true

echo ">>> Packaging tarball..."

TARBALL=/workspace/yolodetector_deploy_aarch64_${RANDOM}.tar.gz
tar czvf "$TARBALL" .

echo "======================================="
echo "  BUILD OK!"
echo "  Output: $TARBALL"
echo "======================================="
