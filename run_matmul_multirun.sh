#!/bin/bash
# Multiple runs of hybrid_victim (with matrix multiply workload)
# Baseline and attack, N trials each, stored with proper structure

set -euo pipefail

WORKDIR="/home/rtx3060/Desktop/sumit-GPU-Codes/gpu_cpu_work_1"
NVCC="/usr/local/cuda/bin/nvcc"
RESULT_DIR="$WORKDIR/results/matmul_analysis"

NUM_TRIALS=${1:-5}
CACHE_KB=12288
WAYS=16
TARGET_SET=0
GPU_ITERS=1000

cd "$WORKDIR"

echo "[Build] Compiling hybrid_victim..."
$NVCC -O3 -arch=sm_86 -I./src -o hybrid_victim src/hybrid_victim.cu -lpthread
echo "[Build] Done."

RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$RESULT_DIR/run_${RUN_TS}"
mkdir -p "$RUN_DIR"

cat > "$RUN_DIR/meta.txt" << META
timestamp=$RUN_TS
num_trials=$NUM_TRIALS
cache_kb=$CACHE_KB
ways=$WAYS
target_set=$TARGET_SET
gpu_iters=$GPU_ITERS
config=matmul_workload_enabled
META

echo "Output dir: $RUN_DIR"
echo ""

for t in $(seq 1 $NUM_TRIALS); do
    echo "=== Trial $t / $NUM_TRIALS ==="
    
    echo "  [Baseline]..."
    printf "seal@123\n" | sudo -S -p "" \
        "$WORKDIR/hybrid_victim" 0 1 $CACHE_KB $WAYS $TARGET_SET $GPU_ITERS \
        > "$RUN_DIR/baseline_trial_${t}.log" 2>&1
    cp "$WORKDIR/results/hybrid_cpu_timings_baseline.csv" \
       "$RUN_DIR/baseline_trial_${t}.csv" 2>/dev/null || true
    
    sleep 1
    
    echo "  [Attack]..."
    printf "seal@123\n" | sudo -S -p "" \
        "$WORKDIR/hybrid_victim" 1 1 $CACHE_KB $WAYS $TARGET_SET $GPU_ITERS \
        > "$RUN_DIR/attack_trial_${t}.log" 2>&1
    cp "$WORKDIR/results/hybrid_cpu_timings_contention.csv" \
       "$RUN_DIR/attack_trial_${t}.csv" 2>/dev/null || true
    
    echo "  Trial $t done."
    sleep 2
done

echo ""
echo "All $NUM_TRIALS trials done. Results: $RUN_DIR"
