#!/bin/bash
# Run MiniTransformer (cpu.pt) + TinyYOLOv2 — 3 runs each
set -euo pipefail
WORKDIR="/home/rtx3060/Desktop/sumit-GPU-Codes/gpu_cpu_work_1"
BIN="$WORKDIR/prime_probe_ai_scan"
MODELS_DIR="$WORKDIR/models"
TORCH_LIB="/home/rtx3060/.local/lib/python3.10/site-packages/torch/lib"
CUDNN_LIB="/usr/local/lib/ollama/mlx_cuda_v13"
RESULT_BASE="$WORKDIR/results/prime_probe_heatmap"
SETS="0 768 1536 2304 3072 3840 4608 5376 6144 6912 7680 8448 9216 9984 10752 11520"
POINTS=64; REPS=64; CACHE_KB=12288; WAYS=16; NUM_RUNS=3

mkdir -p "$RESULT_BASE"
cd "$WORKDIR"

# Remove incomplete MiniTransformer run
rm -rf "$RESULT_BASE/MiniTransformer"
mkdir -p "$RESULT_BASE/MiniTransformer"
echo "model=MiniTransformer" > "$RESULT_BASE/MiniTransformer/meta.txt"

MODELS=("MiniTransformer minitransformer_cpu.pt" "TinyYOLOv2 tinyyolo_v2.pt")

echo "================================================"
echo "  Prime+Probe — Remaining models (RTX 3060)"
echo "================================================"

for MODEL_ENTRY in "${MODELS[@]}"; do
    LABEL=$(echo "$MODEL_ENTRY" | awk '{print $1}')
    FNAME=$(echo "$MODEL_ENTRY" | awk '{print $2}')
    MODEL_PT="$MODELS_DIR/$FNAME"
    [ ! -f "$MODEL_PT" ] && echo "[SKIP] $LABEL ($MODEL_PT not found)" && continue

    M_DIR="$RESULT_BASE/$LABEL"
    mkdir -p "$M_DIR"
    echo "model=$LABEL" > "$M_DIR/meta.txt"
    echo ""; echo "  MODEL: $LABEL"

    for run in $(seq 1 $NUM_RUNS); do
        RUN_TS=$(date +%Y%m%d_%H%M%S)
        RUN_DIR="$M_DIR/run_${run}_${RUN_TS}"
        mkdir -p "$RUN_DIR"
        echo "  Run $run/$NUM_RUNS"

        for set_id in $SETS; do
            printf "seal@123\n" | sudo -S -p "" \
                env LD_LIBRARY_PATH="$TORCH_LIB:$CUDNN_LIB" \
                "$BIN" "$MODEL_PT" "$set_id" "$POINTS" "$REPS" \
                "$CACHE_KB" "$WAYS" \
                > "$RUN_DIR/scan_set_${set_id}.log" 2>&1
            CSV="$WORKDIR/results/pp_ai_${FNAME%.pt}_set${set_id}.csv"
            [ -f "$CSV" ] && cp "$CSV" "$RUN_DIR/scan_set_${set_id}.csv"
            echo "    set $set_id done"
        done
        echo "  Run $run complete"
        sleep 2
    done
    echo "  $LABEL complete."
done
echo ""; echo "All remaining models done. Results: $RESULT_BASE"
