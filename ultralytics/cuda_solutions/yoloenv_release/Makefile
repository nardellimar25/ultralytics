# === Deployment Makefile ===

# Build and deploy paths
BUILD_DIR     := build
DEPLOY_DIR    := $(BUILD_DIR)/deploy
BIN_DIR       := $(DEPLOY_DIR)/bin
LIB_DIR       := $(DEPLOY_DIR)/lib
ENGINE_DIR    := $(DEPLOY_DIR)/engine

# Executable and engine paths
TARGET_BIN    := yolodetector
ENGINE_SRC    := ../yoloengines/yolov8s_256_dockerborn.engine
ENGINE_TARGET := $(ENGINE_DIR)/yolov8s_256_dockerborn.engine

# CUDA path (adjust if needed)
CUDA_LIB_DIR  := /usr/local/cuda-12.3/lib64

# Default target
.PHONY: all
all: build

# Step 1: Configure and compile with CMake
.PHONY: build
build:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. && make -j

# Step 2: Package everything for deployment
.PHONY: deploy
deploy: build

	@echo "==> Creating deployable folder..."
	rm -rf $(DEPLOY_DIR)
	mkdir -p $(BIN_DIR) $(LIB_DIR) $(ENGINE_DIR)

	# Copy binary and engine
	cp $(BUILD_DIR)/$(TARGET_BIN) $(BIN_DIR)/
	cp $(ENGINE_SRC) $(ENGINE_TARGET)

	# === Option A: Manual explicit list ===
	# Uncomment these if you prefer fixed control
	# cp /usr/lib/x86_64-linux-gnu/libnvinfer.so.*        $(LIB_DIR)/
	# cp /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.* $(LIB_DIR)/
	# cp $(CUDA_LIB_DIR)/libcudart.so.*                   $(LIB_DIR)/
	# cp /usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2   $(LIB_DIR)/
	# cp /usr/lib/x86_64-linux-gnu/libopencv_*.so.*       $(LIB_DIR)/ || true
	# cp /usr/lib/x86_64-linux-gnu/libQt5*.so.*           $(LIB_DIR)/ || true

	# === Option B: Auto-detect all linked libraries ===
	ldd $(BIN_DIR)/$(TARGET_BIN) | awk '{print $$3}' | grep '^/' | xargs -I{} cp -v {} $(LIB_DIR) || true

	# Copy dynamic linker (required!)
	cp /usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 $(LIB_DIR)/

	# Create run.sh
	@echo '#!/bin/bash' > $(DEPLOY_DIR)/run.sh
	@echo 'DIR="$$(cd "$$(dirname "$$0")" && pwd)"' >> $(DEPLOY_DIR)/run.sh
	@echo '$$DIR/lib/ld-linux-x86-64.so.2 --library-path $$DIR/lib $$DIR/bin/$(TARGET_BIN)' >> $(DEPLOY_DIR)/run.sh
	chmod +x $(DEPLOY_DIR)/run.sh

	@echo "✅ Deployment folder ready. Run: cd $(DEPLOY_DIR) && ./run.sh"

# cleanup
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)


.PHONY: package
package: deploy
	@echo "==> Creating deploy tarball..."
	tar -czvf yolodetector_deploy.tar.gz -C $(DEPLOY_DIR) .
	@echo "✅ Archive created: yolodetector_deploy.tar.gz"

