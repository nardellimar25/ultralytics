# === Deployment Makefile ===

# Build and deploy paths
BUILD_DIR     := build
DEPLOY_DIR    := $(BUILD_DIR)/deploy
BIN_DIR       := $(DEPLOY_DIR)/bin
LIB_DIR       := $(DEPLOY_DIR)/lib
ENGINE_DIR    := $(DEPLOY_DIR)/engine

# Executable and engine paths
TARGET_BIN    := yolodetector
ENGINE_SRC    := ../yoloengines/yolo.engine
ENGINE_TARGET := $(ENGINE_DIR)/yolo.engine

# CUDA path
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

# === Auto-detect all linked libraries ===
	ldd $(BIN_DIR)/$(TARGET_BIN) | awk '{print $$3}' | grep '^/' | xargs -I{} cp -v {} $(LIB_DIR) || true

# Copy dynamic linker (required!)
	cp /usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 $(LIB_DIR)/

# Copy Qt platform plugin (libqxcb)
	mkdir -p $(LIB_DIR)/qt/plugins/platforms
	cp /usr/lib/x86_64-linux-gnu/qt5/plugins/platforms/libqxcb.so $(LIB_DIR)/qt/plugins/platforms/

# Copy all required Qt shared libs for libqxcb
	@for lib in \
		libQt5XcbQpa.so.5 \
		libQt5Gui.so.5 \
		libQt5Core.so.5 \
		libQt5DBus.so.5 \
		libxcb-xinerama.so.0 \
		libxcb-xinput.so.0 ; do \
		path=$$(ldconfig -p | grep -w $$lib | head -n1 | awk '{print $$NF}'); \
		if [ -n "$$path" ]; then \
			cp -v "$$path" $(LIB_DIR)/; \
		else \
			echo "❌ Missing $$lib"; \
		fi; \
	done


# Create run.sh
	@echo '#!/bin/bash' > $(DEPLOY_DIR)/run.sh
	@echo 'DIR="$$(cd "$$(dirname "$$0")" && pwd)"' >> $(DEPLOY_DIR)/run.sh
	@echo 'export QT_QPA_PLATFORM_PLUGIN_PATH=$$DIR/lib/qt/plugins/platforms' >> $(DEPLOY_DIR)/run.sh
	@echo '$$DIR/lib/ld-linux-x86-64.so.2 --library-path $$DIR/lib $$DIR/bin/$(TARGET_BIN)' >> $(DEPLOY_DIR)/run.sh
	chmod +x $(DEPLOY_DIR)/run.sh


# Create run_profile.sh with advanced CPU tracing for nsys
	@echo '#!/bin/bash' > $(DEPLOY_DIR)/run_profile.sh
	@echo 'DIR="$$(cd "$$(dirname "$$0")" && pwd)"' >> $(DEPLOY_DIR)/run_profile.sh
	@echo 'export QT_QPA_PLATFORM_PLUGIN_PATH="$$DIR/lib/qt/plugins/platforms"' >> $(DEPLOY_DIR)/run_profile.sh
	@echo 'NSYS=$$(command -v nsys)' >> $(DEPLOY_DIR)/run_profile.sh
	@echo 'if [ -z "$$NSYS" ]; then echo "❌ nsys not found"; exit 1; fi' >> $(DEPLOY_DIR)/run_profile.sh
	@echo '"$$NSYS" profile --output yolodetector_profile \\' >> $(DEPLOY_DIR)/run_profile.sh
	@echo '  --trace=cuda,nvtx,osrt --stop-on-exit true \\' >> $(DEPLOY_DIR)/run_profile.sh
	@echo '  $$DIR/lib/ld-linux-x86-64.so.2 --library-path $$DIR/lib $$DIR/bin/$(TARGET_BIN)' >> $(DEPLOY_DIR)/run_profile.sh
	@chmod +x $(DEPLOY_DIR)/run_profile.sh


#"$NSYS" profile --output yolodetector_profile \
  --trace=cuda,nvtx,osrt --stop-on-exit true \
  $DIR/lib/ld-linux-x86-64.so.2 --library-path $DIR/lib $DIR/bin/yolodetector

	@echo "✅ Deployment folder and profiler ready."
	@echo " Run: cd $(DEPLOY_DIR) && ./run.sh "
	@echo " Profile: cd $(DEPLOY_DIR) && ./run_profile.sh"


# cleanup
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)


.PHONY: package
package: deploy
	@echo "==> Creating deploy tarball..."
	tar -czvf yolodetector_deploy.tar.gz -C $(DEPLOY_DIR) .
	@echo "✅ Archive created: yolodetector_deploy.tar.gz"

