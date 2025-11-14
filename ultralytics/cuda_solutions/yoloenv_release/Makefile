# ------------------------------------------------------------
# CMake + deploy/packaging (x86_64 & aarch64)
# Usage:
#   make                -> configure & build into ./build
#   make clean          -> remove ./build
#   make deploy         -> create ./build/deploy with bin/, lib/, engines/, run.sh
#   make package        -> tar.gz of deploy/ into repo root
# Env/overrides:
#   BUILD_TYPE=Release|Debug
#   CMAKE_OPTS='-DYOLO_ENGINE_PATH=... -DACTION_ENGINE_PATH=... -DGPU_SM=...'
# ------------------------------------------------------------

ROOT        := $(abspath $(CURDIR))
BUILD_DIR   := build
DEPLOY_DIR  := $(BUILD_DIR)/deploy
BIN_DIR     := $(DEPLOY_DIR)/bin
LIB_DIR     := $(DEPLOY_DIR)/lib
ENGINE_DIR  := $(DEPLOY_DIR)/engines

TARGET_BIN       := yolodetector
TARGET_BIN_PATH  := $(BUILD_DIR)/bin/$(TARGET_BIN)

# ---------- Engines ----------
ENGINE_SRC_DIR ?= $(ROOT)/engines
YOLO_ENGINE_SRC      := $(ENGINE_SRC_DIR)/yolo.engine
YOLO_ENGINE_TARGET   := $(ENGINE_DIR)/yolo.engine
ACTION_ENGINE_SRC    := $(ENGINE_SRC_DIR)/cls.engine
ACTION_ENGINE_TARGET := $(ENGINE_DIR)/cls.engine

# ---------- Build config ----------
CMAKE_OPTS ?=
BUILD_TYPE ?= Release
ARCH       := $(shell uname -m)
TARGET_ARCH ?= $(ARCH)
JETSON ?= 0
JETSON_BUNDLED_TRT ?= 0
TRT_LIB_DIR ?=
JETSON_BUNDLED_LIBS ?= 
.PHONY: all configure build rebuild clean deploy package

# ------------------------------------------------------------
# Build
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
# Deploy
# ------------------------------------------------------------
# deploy: build
# 	@echo "==> Creating deployable folder..."
# 	@rm -rf $(DEPLOY_DIR)
# 	@mkdir -p $(BIN_DIR) $(LIB_DIR) $(ENGINE_DIR)

# 	@echo "==> Copy binary and engines"
# 	@cp "$(TARGET_BIN_PATH)" "$(BIN_DIR)/"
# 	@if [ -f "$(YOLO_ENGINE_SRC)" ]; then cp "$(YOLO_ENGINE_SRC)" "$(YOLO_ENGINE_TARGET)"; else echo "⚠️  Missing $(YOLO_ENGINE_SRC)"; fi
# 	@if [ -f "$(ACTION_ENGINE_SRC)" ]; then cp "$(ACTION_ENGINE_SRC)" "$(ACTION_ENGINE_TARGET)"; else echo "⚠️  Missing $(ACTION_ENGINE_SRC)"; fi

# 	@echo "==> Auto-detect and copy linked libraries"
# 	@ldd "$(BIN_DIR)/$(TARGET_BIN)" | awk '{print $$3}' | grep -E '^/' | xargs -r -I{} cp -vn {} "$(LIB_DIR)" || true

# 	@echo "==> Detect and copy program interpreter (dynamic linker)"
# 	@interp=$$(readelf -l "$(BIN_DIR)/$(TARGET_BIN)" | awk '/Requesting program interpreter/ {gsub(/[\[\]]/,"",$$4); print $$4}'); \
# 	if [ -n "$$interp" ] && [ -f "$$interp" ]; then \
# 	  echo "Interpreter: $$interp"; \
# 	  cp -v "$$interp" "$(LIB_DIR)/"; \
# 	  ln -sf "$$(basename "$$interp")" "$(LIB_DIR)/ld.so"; \
# 	else \
# 	  echo "⚠ could not detect/copy dynamic linker (will try direct run)"; \
# 	fi

# 	@echo "==> (Optional) Qt platform plugin (best-effort)"
# 	@mkdir -p "$(LIB_DIR)/qt/plugins/platforms"
# 	@for cand in \
# 		/usr/lib/*/qt5/plugins/platforms/libqxcb.so \
# 		/usr/lib/*/qt6/plugins/platforms/libqxcb.so \
# 		/usr/lib/qt5/plugins/platforms/libqxcb.so \
# 		/usr/lib/qt6/plugins/platforms/libqxcb.so ; do \
# 		if [ -f "$$cand" ]; then cp -v "$$cand" "$(LIB_DIR)/qt/plugins/platforms/"; break; fi; \
# 	done
# 	@for lib in libQt5XcbQpa.so.5 libQt5Gui.so.5 libQt5Core.so.5 libQt5DBus.so.5 libxcb-xinerama.so.0 libxcb-xinput.so.0 \
# 	            libQt6Gui.so.6 libQt6Core.so.6 ; do \
# 		path=$$(ldconfig -p 2>/dev/null | grep -w $$lib | head -n1 | awk '{print $$NF}'); \
# 		if [ -n "$$path" ]; then cp -vn "$$path" "$(LIB_DIR)/"; fi; \
# 	done

# 	@echo "==> Create run.sh (uses detected linker if present)"
# 	@echo '#!/bin/bash' > "$(DEPLOY_DIR)/run.sh"
# 	@echo 'DIR="$$(cd "$$(dirname "$$0")" && pwd)"' >> "$(DEPLOY_DIR)/run.sh"
# 	@echo 'export LD_LIBRARY_PATH="$$DIR/lib:$$LD_LIBRARY_PATH"' >> "$(DEPLOY_DIR)/run.sh"
# 	@echo 'export QT_QPA_PLATFORM_PLUGIN_PATH="$$DIR/lib/qt/plugins/platforms"' >> "$(DEPLOY_DIR)/run.sh"
# 	@echo 'LINKER="$$DIR/lib/ld.so"' >> "$(DEPLOY_DIR)/run.sh"
# 	@echo 'if [ -x "$$LINKER" ]; then' >> "$(DEPLOY_DIR)/run.sh"
# 	@echo '  exec "$$LINKER" --library-path "$$DIR/lib" "$$DIR/bin/$(TARGET_BIN)" "$$@"' >> "$(DEPLOY_DIR)/run.sh"
# 	@echo 'else' >> "$(DEPLOY_DIR)/run.sh"
# 	@echo '  echo "⚠ linker $$LINKER not found; launching directly" >&2' >> "$(DEPLOY_DIR)/run.sh"
# 	@echo '  exec "$$DIR/bin/$(TARGET_BIN)" "$$@"' >> "$(DEPLOY_DIR)/run.sh"
# 	@echo 'fi' >> "$(DEPLOY_DIR)/run.sh"
# 	@chmod +x "$(DEPLOY_DIR)/run.sh"

# 	@echo "==> Create run_profile.sh (nsys if available)"
# 	@echo '#!/bin/bash' > "$(DEPLOY_DIR)/run_profile.sh"
# 	@echo 'DIR="$$(cd "$$(dirname "$$0")" && pwd)"' >> "$(DEPLOY_DIR)/run_profile.sh"
# 	@echo 'export LD_LIBRARY_PATH="$$DIR/lib:$$LD_LIBRARY_PATH"' >> "$(DEPLOY_DIR)/run_profile.sh"
# 	@echo 'export QT_QPA_PLATFORM_PLUGIN_PATH="$$DIR/lib/qt/plugins/platforms"' >> "$(DEPLOY_DIR)/run_profile.sh"
# 	@echo 'NSYS=$$(command -v nsys || true)' >> "$(DEPLOY_DIR)/run_profile.sh"
# 	@echo 'if [ -z "$$NSYS" ]; then echo "❌ nsys not found"; exit 1; fi' >> "$(DEPLOY_DIR)/run_profile.sh"
# 	@echo 'LINKER="$$DIR/lib/ld.so"' >> "$(DEPLOY_DIR)/run_profile.sh"
# 	@echo 'if [ -x "$$LINKER" ]; then' >> "$(DEPLOY_DIR)/run_profile.sh"
# 	@echo '  exec "$$NSYS" profile --output yolodetector_profile --trace=cuda,nvtx --stop-on-exit true "$$LINKER" --library-path "$$DIR/lib" "$$DIR/bin/$(TARGET_BIN)" "$$@"' >> "$(DEPLOY_DIR)/run_profile.sh"
# 	@echo 'else' >> "$(DEPLOY_DIR)/run_profile.sh"
# 	@echo '  echo "⚠ linker $$LINKER not found; profiling direct exec" >&2' >> "$(DEPLOY_DIR)/run_profile.sh"
# 	@echo '  exec "$$NSYS" profile --output yolodetector_profile --trace=cuda,nvtx --stop-on-exit true "$$DIR/bin/$(TARGET_BIN)" "$$@"' >> "$(DEPLOY_DIR)/run_profile.sh"
# 	@echo 'fi' >> "$(DEPLOY_DIR)/run_profile.sh"
# 	@chmod +x "$(DEPLOY_DIR)/run_profile.sh"

# 	@echo "✅ Deployment folder and profiler ready."
# 	@echo "   Run:     cd $(DEPLOY_DIR) && ./run.sh"
# 	@echo "   Profile: cd $(DEPLOY_DIR) && ./run_profile.sh"

# ------------------------------------------------------------
# Deploy 2.0: with Jetson bundled libs and TensorRT
# ------------------------------------------------------------
deploy: build
ifeq ($(JETSON),1)
  ifeq ($(JETSON_BUNDLED_TRT),1)
	@$(MAKE) deploy_jetson_bundled_trt
  else
	@$(MAKE) deploy_jetson
  endif
else
	@$(MAKE) deploy_native
endif


# --------------------- Native deploy  -----------------------
deploy_native:
	@echo "==> Creating deployable folder (native)..."
	@rm -rf $(DEPLOY_DIR)
	@mkdir -p $(BIN_DIR) $(LIB_DIR) $(ENGINE_DIR)

	@echo "==> Copy binary and engines"
	@cp "$(TARGET_BIN_PATH)" "$(BIN_DIR)/"
	@if [ -f "$(YOLO_ENGINE_SRC)" ]; then cp "$(YOLO_ENGINE_SRC)" "$(YOLO_ENGINE_TARGET)"; else echo "⚠️  Missing $(YOLO_ENGINE_SRC)"; fi
	@if [ -f "$(ACTION_ENGINE_SRC)" ]; then cp "$(ACTION_ENGINE_SRC)" "$(ACTION_ENGINE_TARGET)"; else echo "⚠️  Missing $(ACTION_ENGINE_SRC)"; fi

	@echo "==> Auto-detect and copy linked libraries"
	@ldd "$(BIN_DIR)/$(TARGET_BIN)" | awk '{print $$3}' | grep -E '^/' | xargs -r -I{} cp -vn {} "$(LIB_DIR)" || true

	@echo "==> Detect and copy program interpreter (dynamic linker)"
	@interp=$$(readelf -l "$(BIN_DIR)/$(TARGET_BIN)" | awk '/Requesting program interpreter/ {gsub(/[\[\]]/,"",$$4); print $$4}'); \
	if [ -n "$$interp" ] && [ -f "$$interp" ]; then \
	  echo "Interpreter: $$interp"; \
	  cp -v "$$interp" "$(LIB_DIR)/"; \
	  ln -sf "$$(basename "$$interp")" "$(LIB_DIR)/ld.so"; \
	else \
	  echo "⚠ could not detect/copy dynamic linker (will try direct run)"; \
	fi

	@echo "==> (Optional) Qt platform plugin (best-effort)"
	@mkdir -p "$(LIB_DIR)/qt/plugins/platforms"
	@for cand in \
		/usr/lib/*/qt5/plugins/platforms/libqxcb.so \
		/usr/lib/*/qt6/plugins/platforms/libqxcb.so \
		/usr/lib/qt5/plugins/platforms/libqxcb.so \
		/usr/lib/qt6/plugins/platforms/libqxcb.so ; do \
		if [ -f "$$cand" ]; then cp -v "$$cand" "$(LIB_DIR)/qt/plugins/platforms/"; break; fi; \
	done
	@for lib in libQt5XcbQpa.so.5 libQt5Gui.so.5 libQt5Core.so.5 libQt5DBus.so.5 libxcb-xinerama.so.0 libxcb-xinput.so.0 \
	            libQt6Gui.so.6 libQt6Core.so.6 ; do \
		path=$$(ldconfig -p 2>/dev/null | grep -w $$lib | head -n1 | awk '{print $$NF}'); \
		if [ -n "$$path" ]; then cp -vn "$$path" "$(LIB_DIR)/"; fi; \
	done

	@echo "==> Create run.sh (uses detected linker if present)"
	@echo '#!/bin/bash' > "$(DEPLOY_DIR)/run.sh"
	@echo 'DIR="$$(cd "$$(dirname "$$0")" && pwd)"' >> "$(DEPLOY_DIR)/run.sh"
	@echo 'export LD_LIBRARY_PATH="$$DIR/lib:$$LD_LIBRARY_PATH"' >> "$(DEPLOY_DIR)/run.sh"
	@echo 'export QT_QPA_PLATFORM_PLUGIN_PATH="$$DIR/lib/qt/plugins/platforms"' >> "$(DEPLOY_DIR)/run.sh"
	@echo 'LINKER="$$DIR/lib/ld.so"' >> "$(DEPLOY_DIR)/run.sh"
	@echo 'if [ -x "$$LINKER" ]; then' >> "$(DEPLOY_DIR)/run.sh"
	@echo '  exec "$$LINKER" --library-path "$$DIR/lib" "$$DIR/bin/$(TARGET_BIN)" "$$@"' >> "$(DEPLOY_DIR)/run.sh"
	@echo 'else' >> "$(DEPLOY_DIR)/run.sh"
	@echo '  echo "⚠ linker $$LINKER not found; launching directly" >&2' >> "$(DEPLOY_DIR)/run.sh"
	@echo '  exec "$$DIR/bin/$(TARGET_BIN)" "$$@"' >> "$(DEPLOY_DIR)/run.sh"
	@echo 'fi' >> "$(DEPLOY_DIR)/run.sh"
	@chmod +x "$(DEPLOY_DIR)/run.sh"

	@echo "==> Create run_profile.sh (nsys if available)"
	@echo '#!/bin/bash' > "$(DEPLOY_DIR)/run_profile.sh"
	@echo 'DIR="$$(cd "$$(dirname "$$0")" && pwd)"' >> "$(DEPLOY_DIR)/run_profile.sh"
	@echo 'export LD_LIBRARY_PATH="$$DIR/lib:$$LD_LIBRARY_PATH"' >> "$(DEPLOY_DIR)/run_profile.sh"
	@echo 'export QT_QPA_PLATFORM_PLUGIN_PATH="$$DIR/lib/qt/plugins/platforms"' >> "$(DEPLOY_DIR)/run_profile.sh"
	@echo 'NSYS=$$(command -v nsys || true)' >> "$(DEPLOY_DIR)/run_profile.sh"
	@echo 'if [ -z "$$NSYS" ]; then echo "❌ nsys not found"; exit 1; fi' >> "$(DEPLOY_DIR)/run_profile.sh"
	@echo 'LINKER="$$DIR/lib/ld.so"' >> "$(DEPLOY_DIR)/run_profile.sh"
	@echo 'if [ -x "$$LINKER" ]; then' >> "$(DEPLOY_DIR)/run_profile.sh"
	@echo '  exec "$$NSYS" profile --output yolodetector_profile --trace=cuda,nvtx --stop-on-exit true "$$LINKER" --library-path "$$DIR/lib" "$$DIR/bin/$(TARGET_BIN)" "$$@"' >> "$(DEPLOY_DIR)/run_profile.sh"
	@echo 'else' >> "$(DEPLOY_DIR)/run_profile.sh"
	@echo '  echo "⚠ linker $$LINKER not found; profiling direct exec" >&2' >> "$(DEPLOY_DIR)/run_profile.sh"
	@echo '  exec "$$NSYS" profile --output yolodetector_profile --trace=cuda,nvtx --stop-on-exit true "$$DIR/bin/$(TARGET_BIN)" "$$@"' >> "$(DEPLOY_DIR)/run_profile.sh"
	@echo 'fi' >> "$(DEPLOY_DIR)/run_profile.sh"
	@chmod +x "$(DEPLOY_DIR)/run_profile.sh"

	@echo "✅ Deployment folder and profiler ready (native)."
 	@echo "   Run:     cd $(DEPLOY_DIR) && ./run.sh"
 	@echo "   Profile: cd $(DEPLOY_DIR) && ./run_profile.sh"

# ------------------ Jetson specific deploy  -----------------
# ------------ expects system libs from JetPack --------------
deploy_jetson:
# 	@echo "❌ deploy_jetson not implemented yet; set JETSON=0 or wait for next step."
	@echo "==> Creating Jetson deployable folder (system libs on target)..."
	@rm -rf $(DEPLOY_DIR)
	@mkdir -p $(BIN_DIR) $(LIB_DIR) $(ENGINE_DIR)

	@echo "==> Copy binary and engines"
	@cp "$(TARGET_BIN_PATH)" "$(BIN_DIR)/"
	@if [ -f "$(YOLO_ENGINE_SRC)" ]; then \
		cp "$(YOLO_ENGINE_SRC)" "$(YOLO_ENGINE_TARGET)"; \
	else \
		echo "ℹ️  No yolo.engine in $(ENGINE_SRC_DIR); remember to copy your Xavier yolo.engine later."; \
	fi
	@if [ -f "$(ACTION_ENGINE_SRC)" ]; then \
		cp "$(ACTION_ENGINE_SRC)" "$(ACTION_ENGINE_TARGET)"; \
	else \
		echo "ℹ️  No cls.engine in $(ENGINE_SRC_DIR); remember to copy your Xavier cls.engine later."; \
	fi

	@echo "==> Create run.sh (Jetson: use system libs + optional ./lib)"
	@echo '#!/bin/bash' > "$(DEPLOY_DIR)/run.sh"
	@echo 'DIR="$$(cd "$$(dirname "$$0")" && pwd)"' >> "$(DEPLOY_DIR)/run.sh"
	@echo 'export LD_LIBRARY_PATH="$$DIR/lib:$$LD_LIBRARY_PATH"' >> "$(DEPLOY_DIR)/run.sh"
	@echo 'exec "$$DIR/bin/$(TARGET_BIN)" "$$@"' >> "$(DEPLOY_DIR)/run.sh"
	@chmod +x "$(DEPLOY_DIR)/run.sh"

	@echo "==> Create run_profile.sh (nsys on Jetson if available)"
	@echo '#!/bin/bash' > "$(DEPLOY_DIR)/run_profile.sh"
	@echo 'DIR="$$(cd "$$(dirname "$$0")" && pwd)"' >> "$(DEPLOY_DIR)/run_profile.sh"
	@echo 'export LD_LIBRARY_PATH="$$DIR/lib:$$LD_LIBRARY_PATH"' >> "$(DEPLOY_DIR)/run_profile.sh"
	@echo 'NSYS=$$(command -v nsys || true)' >> "$(DEPLOY_DIR)/run_profile.sh"
	@echo 'if [ -z "$$NSYS" ]; then echo "❌ nsys not found"; exit 1; fi' >> "$(DEPLOY_DIR)/run_profile.sh"
	@echo 'exec "$$NSYS" profile --output yolodetector_profile --trace=cuda,nvtx --stop-on-exit true "$$DIR/bin/$(TARGET_BIN)" "$$@"' >> "$(DEPLOY_DIR)/run_profile.sh"
	@chmod +x "$(DEPLOY_DIR)/run_profile.sh"

	@echo "✅ Jetson deployment folder ready (no extra libs; relies on JetPack)."
 	@echo "   Run:     cd $(DEPLOY_DIR) && ./run.sh"
 	@echo "   Profile: cd $(DEPLOY_DIR) && ./run_profile.sh"

# -------- Jetson deploy with bundled TensorRT / selected libs -------------
deploy_jetson_bundled_trt:
	@echo "❌ deploy_jetson_bundled_trt not implemented yet."
	@echo "   For now, use: make JETSON=1 JETSON_BUNDLED_TRT=0 deploy"


# ------------------------------------------------------------
# Package
# ------------------------------------------------------------
package: deploy
	@echo "==> Creating deploy tarball..."
	@tar -czvf "yolodetector_deploy_$(TARGET_ARCH).tar.gz" -C "$(DEPLOY_DIR)" .
	@echo "✅ Archive created: yolodetector_deploy_$(TARGET_ARCH).tar.gz"
