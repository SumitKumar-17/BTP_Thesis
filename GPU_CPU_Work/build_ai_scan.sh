#!/bin/bash
# Build evset_ai_scan binary linked against LibTorch
set -euo pipefail

WORKDIR="/home/rtx3060/Desktop/sumit-GPU-Codes/gpu_cpu_work_1"
TORCH_ROOT="/home/rtx3060/.local/lib/python3.10/site-packages/torch"
NVCC="/usr/local/cuda/bin/nvcc"

INCLUDES="-I${TORCH_ROOT}/include -I${TORCH_ROOT}/include/torch/csrc/api/include -I./src"
LIBS="-L${TORCH_ROOT}/lib -ltorch -ltorch_cuda -lc10 -lc10_cuda -ltorch_cpu"
RPATH="-Wl,-rpath,${TORCH_ROOT}/lib"
FLAGS="-O2 -arch=sm_86 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0"

echo "[Build] Compiling evset_ai_scan..."
cd "$WORKDIR"
$NVCC $FLAGS $INCLUDES src/evset_ai_scan.cu -o evset_ai_scan $LIBS $RPATH
echo "[Build] Done -> $WORKDIR/evset_ai_scan"
