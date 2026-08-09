# Makefile for the tflm-goldens harness (host-only, macOS/Linux).
#
# Builds main.cc against a tflite-micro checkout at the PINNED SHA
# 18b9e6f2a8c5a9518e588f59c2ba16ef7ef9d551 and runs the 6 zoo models through
# the REAL tflite-micro interpreter (reference kernels).
#
# NOTE on build system: the pinned SHA has NO CMake support (it was removed
# from tflite-micro upstream). The upstream Makefile build
# (tensorflow/lite/micro/tools/make/Makefile) is the supported path at this
# SHA: it downloads third-party deps (flatbuffers, gemmlowp, ruy, kissfft)
# and produces libtensorflow-microlite.a. This Makefile wraps that build and
# compiles/links main.cc against the resulting library.
#
# Usage:
#   make            # extract inputs, build TFLM lib (if needed), build harness
#   make run        # build + run the harness
#   make inputs     # re-extract .bin inputs from the Rust goldens
#   make clean

TFLM_ROOT ?= /tmp/tflite-micro
GOLDENS_DIR ?= ../../hematite-tests/goldens/models

CXX ?= clang++
PYTHON ?= python3

# Mirrors the upstream Makefile's GENDIR computation (host build only):
# gen/<os>_<arch>_default_gcc/
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  TFLM_TARGET := osx
else
  TFLM_TARGET := linux
endif
TFLM_ARCH := $(shell uname -m)
TFLM_GENDIR := $(TFLM_ROOT)/gen/$(TFLM_TARGET)_$(TFLM_ARCH)_default_gcc
TFLM_LIB := $(TFLM_GENDIR)/lib/libtensorflow-microlite.a

TFLM_MAKE_DIR := $(TFLM_ROOT)/tensorflow/lite/micro/tools/make
TFLM_DOWNLOADS := $(TFLM_MAKE_DIR)/downloads

# Same defines the TFLM library was compiled with (reference kernels, static
# memory) — keep the harness ABI-consistent with the library.
COMMON_DEFS := -DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON

INCLUDES := \
  -I. \
  -Igenerated \
  -I$(TFLM_ROOT) \
  -I$(TFLM_DOWNLOADS)/flatbuffers/include \
  -I$(TFLM_DOWNLOADS)/gemmlowp \
  -I$(TFLM_DOWNLOADS)/ruy \
  -I$(TFLM_DOWNLOADS)/kissfft \
  -I$(TFLM_GENDIR)/genfiles

# No optimization flags: deterministic, reference-kernel execution.
CXXFLAGS := -std=c++17 -O0 -Wall -Wextra $(COMMON_DEFS) $(INCLUDES)

BIN := tflm-goldens

.PHONY: all inputs run clean verify-sha

all: $(BIN)

# --- input extraction (byte-identical golden INPUT_DATA -> raw .bin) --------

inputs: generated/cases.inc

generated/cases.inc: extract_inputs.py $(wildcard $(GOLDENS_DIR)/*.rs)
	$(PYTHON) extract_inputs.py --goldens-dir $(GOLDENS_DIR) --out-dir generated

# --- tflite-micro library at the pinned SHA ---------------------------------

$(TFLM_LIB):
	@test -d $(TFLM_ROOT) || (echo "error: TFLM_ROOT $(TFLM_ROOT) missing; clone tflite-micro at the pinned SHA (see README.md)" && exit 1)
	@cd $(TFLM_ROOT) && test "$$(git rev-parse HEAD)" = "18b9e6f2a8c5a9518e588f59c2ba16ef7ef9d551" || (echo "error: TFLM checkout is not at pinned SHA 18b9e6f2a8c5a9518e588f59c2ba16ef7ef9d551" && exit 1)
	cd $(TFLM_ROOT) && make -f $(TFLM_MAKE_DIR)/Makefile microlite -j8

# --- harness ------------------------------------------------------------------

main.o: main.cc generated/cases.inc $(TFLM_LIB)
	$(CXX) $(CXXFLAGS) -c main.cc -o main.o

$(BIN): main.o $(TFLM_LIB)
	$(CXX) main.o $(TFLM_LIB) -o $(BIN)

run: all
	./$(BIN)

verify-sha:
	@cd $(TFLM_ROOT) && git rev-parse HEAD

clean:
	rm -rf $(BIN) main.o generated
