#!/bin/bash
set -euo pipefail

# Usage:
#   ./run_set_sweep_experiment.sh [trials] [gpu_iterations] [sets_csv] [use_sudo] [sudo_password]
# Example:
#   ./run_set_sweep_experiment.sh 3 1000 "0,64,128" 1 "seal@123"

TRIALS=${1:-3}
GPU_ITERS=${2:-1000}
SETS_CSV=${3:-"0,64,128,256"}
USE_SUDO=${4:-0}
SUDO_PASS=${5:-""}

CACHE_KB=12288
CACHE_WAYS=16
RESULT_ROOT="results/set_sweep"

mkdir -p "$RESULT_ROOT"

if [[ "$USE_SUDO" == "1" && -z "$SUDO_PASS" ]]; then
  echo "ERROR: use_sudo=1 requires sudo_password argument"
  exit 1
fi

TOTAL_SETS=$(( CACHE_KB * 1024 / (64 * CACHE_WAYS) ))

echo "=== GPU-CPU SET SWEEP EXPERIMENT ==="
echo "Trials: $TRIALS"
echo "GPU iterations: $GPU_ITERS"
echo "Sets: $SETS_CSV"
echo "LLC sets: $TOTAL_SETS"
echo ""

run_cmd() {
  if [[ "$USE_SUDO" == "1" ]]; then
    printf "%s\n" "$SUDO_PASS" | sudo -S -p "" "$@"
  else
    "$@"
  fi
}

for set_id in ${SETS_CSV//,/ }; do
  if (( set_id < 0 || set_id >= TOTAL_SETS )); then
    echo "[SKIP] set $set_id out of range [0,$TOTAL_SETS)"
    continue
  fi

  echo "=== Target LLC set $set_id ==="

  SET_DIR="$RESULT_ROOT/set_${set_id}"
  mkdir -p "$SET_DIR"

  for trial in $(seq 1 "$TRIALS"); do
    echo "[Run] set=$set_id trial=$trial baseline"
    run_cmd ./hybrid_victim 0 1 "$CACHE_KB" "$CACHE_WAYS" "$set_id" "$GPU_ITERS" \
      > "$SET_DIR/baseline_set${set_id}_trial${trial}.log"
    cp results/hybrid_cpu_timings_baseline.csv "$SET_DIR/baseline_set${set_id}_trial${trial}.csv"

    echo "[Run] set=$set_id trial=$trial attack"
    run_cmd ./hybrid_victim 1 1 "$CACHE_KB" "$CACHE_WAYS" "$set_id" "$GPU_ITERS" \
      > "$SET_DIR/attack_set${set_id}_trial${trial}.log"
    cp results/hybrid_cpu_timings_contention.csv "$SET_DIR/attack_set${set_id}_trial${trial}.csv"
  done

  echo ""
done

echo "Sweep finished. Output: $RESULT_ROOT"
