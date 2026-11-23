#!/bin/bash
# Multi-run heatmap experiment
# Runs the eviction-set address scan for 16 LLC sets, N times
# Each run stored in results/multirun_heatmap/run_YYYYMMDD_HHMMSS/

set -euo pipefail

WORKDIR="/home/rtx3060/Desktop/sumit-GPU-Codes/gpu_cpu_work_1"
EVSET_BIN="$WORKDIR/evset_address_scan"
RESULT_BASE="$WORKDIR/results/multirun_heatmap"
NVCC="/usr/local/cuda/bin/nvcc"

NUM_RUNS=${1:-3}
POINTS_PER_SET=${2:-64}
REPS_PER_ADDR=${3:-64}
GPU_ITERS=${4:-20000}
CACHE_KB=12288
WAYS=16

# Sets to sweep (evenly distributed across 12288 total sets)
SETS="0 768 1536 2304 3072 3840 4608 5376 6144 6912 7680 8448 9216 9984 10752 11520"

cd "$WORKDIR"

# Rebuild evset_address_scan with latest source
echo "[Build] Compiling evset_address_scan..."
$NVCC -O3 -arch=sm_86 -I./src -o evset_address_scan src/evset_address_scan.cu
echo "[Build] Done."

for run in $(seq 1 $NUM_RUNS); do
    RUN_TS=$(date +%Y%m%d_%H%M%S)
    RUN_DIR="$RESULT_BASE/run_${run}_${RUN_TS}"
    mkdir -p "$RUN_DIR"
    
    echo ""
    echo "========== RUN $run / $NUM_RUNS  [${RUN_TS}] =========="
    echo "Output: $RUN_DIR"
    
    # Meta file
    cat > "$RUN_DIR/meta.txt" << META
run_number=$run
timestamp=$RUN_TS
sets=$SETS
points_per_set=$POINTS_PER_SET
reps_per_addr=$REPS_PER_ADDR
gpu_iters=$GPU_ITERS
cache_kb=$CACHE_KB
ways=$WAYS
binary=$(md5sum $EVSET_BIN | awk '{print $1}')
META

    for set_id in $SETS; do
        echo "  [Set $set_id]..."
        printf "seal@123\n" | sudo -S -p "" \
            "$EVSET_BIN" "$set_id" "$POINTS_PER_SET" "$REPS_PER_ADDR" "$GPU_ITERS" "$CACHE_KB" "$WAYS" \
            > "$RUN_DIR/scan_set_${set_id}.log" 2>&1
        
        # Copy CSV into run directory (evset_address_scan writes to results/ by default)
        if [ -f "$WORKDIR/results/evset_address_scan_set${set_id}.csv" ]; then
            cp "$WORKDIR/results/evset_address_scan_set${set_id}.csv" \
               "$RUN_DIR/scan_set_${set_id}.csv"
        fi
        echo "    done (set $set_id)"
    done
    
    echo "  Run $run complete -> $RUN_DIR"
    sleep 2  # brief pause between runs
done

echo ""
echo "All $NUM_RUNS runs complete. Results in: $RESULT_BASE"
