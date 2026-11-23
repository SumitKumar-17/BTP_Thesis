#!/bin/bash
# Run LLC heatmap scan for each AI model — 3 runs, 16 sets per run.
# Uses evset_ai_scan which runs model inference + eviction in ONE binary.

set -euo pipefail
WORKDIR="/home/rtx3060/Desktop/sumit-GPU-Codes/gpu_cpu_work_1"
BIN="$WORKDIR/evset_ai_scan"
MODELS_DIR="$WORKDIR/models"
TORCH_LIB="/home/rtx3060/.local/lib/python3.10/site-packages/torch/lib"
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"
RESULT_BASE="$WORKDIR/results/ai_model_heatmap"
SETS="0 768 1536 2304 3072 3840 4608 5376 6144 6912 7680 8448 9216 9984 10752 11520"
POINTS=64; REPS=64; GPU_ITERS=10000; CACHE_KB=12288; WAYS=16; NUM_RUNS=3

mkdir -p "$RESULT_BASE"
cd "$WORKDIR"

# models: "label  filename"
MODELS=(
    "MobileNetV2        mobilenet_v2.pt"
    "SqueezeNet1_0      squeezenet1_0.pt"
    "EfficientNetLite0  efficientnet_lite0.pt"
    "MiniTransformer    minitransformer.pt"
    "TinyYOLOv2         tinyyolo_v2.pt"
)

for MODEL_ENTRY in "${MODELS[@]}"; do
    LABEL=$(echo "$MODEL_ENTRY" | awk '{print $1}')
    FNAME=$(echo "$MODEL_ENTRY" | awk '{print $2}')
    MODEL_PT="$MODELS_DIR/$FNAME"

    if [ ! -f "$MODEL_PT" ]; then
        echo "[SKIP] $LABEL — model file not found: $MODEL_PT"
        continue
    fi

    M_DIR="$RESULT_BASE/$LABEL"
    mkdir -p "$M_DIR"

    echo ""
    echo "================================================"
    echo "  MODEL: $LABEL  ($FNAME)"
    echo "================================================"
    echo "model=$LABEL" > "$M_DIR/meta.txt"
    echo "file=$FNAME"  >> "$M_DIR/meta.txt"

    for run in $(seq 1 $NUM_RUNS); do
        RUN_TS=$(date +%Y%m%d_%H%M%S)
        RUN_DIR="$M_DIR/run_${run}_${RUN_TS}"
        mkdir -p "$RUN_DIR"
        echo ""
        echo "  [$LABEL] Run $run/$NUM_RUNS  [$RUN_TS]"

        for set_id in $SETS; do
            printf "seal@123\n" | sudo -S -p "" \
                env LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}" \
                "$BIN" "$MODEL_PT" "$set_id" "$POINTS" "$REPS" \
                "$GPU_ITERS" 0 "$CACHE_KB" "$WAYS" \
                > "$RUN_DIR/scan_set_${set_id}.log" 2>&1

            # binary writes to results/ai_scan_<model>_set<id>.csv
            CSV="$WORKDIR/results/ai_scan_${FNAME%.pt}_set${set_id}.csv"
            if [ -f "$CSV" ]; then
                cp "$CSV" "$RUN_DIR/scan_set_${set_id}.csv"
            fi
            echo "    set $set_id done"
        done

        echo "  [$LABEL] Run $run complete -> $RUN_DIR"
        sleep 2
    done
    echo "  $LABEL complete."
done

echo ""
echo "All AI model scans complete. Results in: $RESULT_BASE"
