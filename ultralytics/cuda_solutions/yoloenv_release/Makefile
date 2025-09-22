# ------------------------------------------------------------
# Simple make wrapper around CMake + deploy/packaging
# Usage:
#   make            -> configure & build into ./build
#   make clean      -> remove ./build
#   make deploy     -> create ./deploy with bin/, lib/, engines/, run.sh
#   make package    -> tar.gz of deploy/ into ./packages
# ------------------------------------------------------------

# Deploy paths
ROOT        := $(abspath $(CURDIR))
BUILD_DIR   := build
DEPLOY_DIR  := $(BUILD_DIR)/deploy
BIN_DIR     := $(DEPLOY_DIR)/bin
LIB_DIR     := $(DEPLOY_DIR)/lib
ENGINE_DIR  := $(DEPLOY_DIR)/engines

# Binary name & path
TARGET_BIN       := yolodetector
TARGET_BIN_PATH  := $(BUILD_DIR)/bin/$(TARGET_BIN)

# ---------- Engine sources ----------
ENGINE_SRC_DIR ?= $(ROOT)/engines

# YOLO engine (existing)
YOLO_ENGINE_SRC      := $(ENGINE_SRC_DIR)/yolo.engine
YOLO_ENGINE_TARGET := $(ENGINE_DIR)/yolo.engine

# NEW: Action-classifier engine
ACTION_ENGINE_SRC    := $(ENGINE_SRC_DIR)/action_cls_fp16.engine
ACTION_ENGINE_TARGET := $(ENGINE_DIR)/action_cls_fp16.engine

# --- CMake options ------------------------------------------
CMAKE_OPTS ?=
BUILD_TYPE ?= Release


.PHONY: all build configure rebuild clean deploy package

# ------------------------------------------------------------
# Build targets
# ------------------------------------------------------------
all: build

configure:
	@echo "==> Configuring CMake ($(BUILD_TYPE))..."
	@mkdir -p $(BUILD_DIR)
	@cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(CMAKE_OPTS)

build: configure
	@echo "==> Building..."
	@cmake --build $(BUILD_DIR) -- -j$$(nproc)

rebuild:
	@echo "==> Rebuilding from scratch..."
	@rm -rf $(BUILD_DIR)
	@$(MAKE) build

clean:
	@echo "==> Cleaning build and deploy dirs..."
	@rm -rf $(BUILD_DIR)

# ------------------------------------------------------------
# Deploy: copy binary, resolve and copy shared libs, copy engines, write run.sh
# ------------------------------------------------------------
deploy: build
	@echo "==> Creating deployable folder..."
	@rm -rf $(DEPLOY_DIR)
	@mkdir -p $(BIN_DIR) $(LIB_DIR) $(ENGINE_DIR)

	@echo "==> Copy binary and engines"
	@cp "$(TARGET_BIN_PATH)" "$(BIN_DIR)/"
	@if [ -f "$(YOLO_ENGINE_SRC)" ]; then cp "$(YOLO_ENGINE_SRC)" "$(YOLO_ENGINE_TARGET)"; else echo "⚠️  Missing $(YOLO_ENGINE_SRC)"; fi
	@if [ -f "$(ACTION_ENGINE_SRC)" ]; then cp "$(ACTION_ENGINE_SRC)" "$(ACTION_ENGINE_TARGET)"; else echo "⚠️  Missing $(ACTION_ENGINE_SRC)"; fi

	@echo "==> Auto-detect and copy linked libraries"
	@ldd $(BIN_DIR)/$(TARGET_BIN) | awk '{print $$3}' | grep '^/' | xargs -I{} cp -vn {} $(LIB_DIR) || true

	@echo "==> Copy dynamic linker (best-effort)"
	@if [ -f /usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 ]; then cp -v /usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 $(LIB_DIR)/; \
	elif [ -f /lib64/ld-linux-x86-64.so.2 ]; then cp -v /lib64/ld-linux-x86-64.so.2 $(LIB_DIR)/; \
	else echo '⚠️  ld-linux-x86-64.so.2 not found in common paths'; fi

	@echo "==> (Optional) Qt platform plugin"
	@mkdir -p $(LIB_DIR)/qt/plugins/platforms
	@if [ -f /usr/lib/x86_64-linux-gnu/qt5/plugins/platforms/libqxcb.so ]; then \
		cp -v /usr/lib/x86_64-linux-gnu/qt5/plugins/platforms/libqxcb.so $(LIB_DIR)/qt/plugins/platforms/; \
	fi
	@for lib in \
		libQt5XcbQpa.so.5 libQt5Gui.so.5 libQt5Core.so.5 libQt5DBus.so.5 \
		libxcb-xinerama.so.0 libxcb-xinput.so.0 ; do \
		path=$$(ldconfig -p | grep -w $$lib | head -n1 | awk '{print $$NF}'); \
		if [ -n "$$path" ]; then cp -vn "$$path" $(LIB_DIR)/; else echo "⚠️  Missing $$lib (skip)"; fi; \
	done

	@echo "==> Create run.sh"
	@echo '#!/bin/bash' > $(DEPLOY_DIR)/run.sh
	@echo 'DIR="$$(cd "$$(dirname "$$0")" && pwd)"' >> $(DEPLOY_DIR)/run.sh
	@echo 'export QT_QPA_PLATFORM_PLUGIN_PATH="$$DIR/lib/qt/plugins/platforms"' >> $(DEPLOY_DIR)/run.sh
	@echo '"$$DIR/lib/ld-linux-x86-64.so.2" --library-path "$$DIR/lib" "$$DIR/bin/$(TARGET_BIN)" "$$@"' >> $(DEPLOY_DIR)/run.sh
	@chmod +x $(DEPLOY_DIR)/run.sh

	@echo "==> Create run_profile.sh"
	@echo '#!/bin/bash' > $(DEPLOY_DIR)/run_profile.sh
	@echo 'DIR="$$(cd "$$(dirname "$$0")" && pwd)"' >> $(DEPLOY_DIR)/run_profile.sh
	@echo 'export QT_QPA_PLATFORM_PLUGIN_PATH="$$DIR/lib/qt/plugins/platforms"' >> $(DEPLOY_DIR)/run_profile.sh
	@echo 'NSYS=$$(command -v nsys)' >> $(DEPLOY_DIR)/run_profile.sh
	@echo 'if [ -z "$$NSYS" ]; then echo "❌ nsys not found"; exit 1; fi' >> $(DEPLOY_DIR)/run_profile.sh
	@echo '"$$NSYS" profile --output yolodetector_profile --trace=cuda,nvtx,osrt --stop-on-exit true \\' >> $(DEPLOY_DIR)/run_profile.sh
	@echo '  "$$DIR/lib/ld-linux-x86-64.so.2" --library-path "$$DIR/lib" "$$DIR/bin/$(TARGET_BIN)" "$$@"' >> $(DEPLOY_DIR)/run_profile.sh
	@chmod +x $(DEPLOY_DIR)/run_profile.sh

	@echo "✅ Deployment folder and profiler ready."
	@echo "   Run:     cd $(DEPLOY_DIR) && ./run.sh"
	@echo "   Profile: cd $(DEPLOY_DIR) && ./run_profile.sh"

# ------------------------------------------------------------
# Package deploy/ into a tar.gz
# ------------------------------------------------------------
.PHONY: package
package: deploy
	@echo "==> Creating deploy tarball..."
	tar -czvf yolodetector_deploy_linux.tar.gz -C $(DEPLOY_DIR) .
	@echo "✅ Archive created: yolodetector_deploy_linux.tar.gz"