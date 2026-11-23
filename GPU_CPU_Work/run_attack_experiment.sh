#!/bin/bash

# ============================================================================
# Attack Experiment Script
# Runs victim and attacker concurrently to measure cache contention
# ============================================================================

echo "=== GPU-CPU CACHE CONTENTION ATTACK EXPERIMENT ==="
echo ""
echo "This experiment runs the victim and attacker processes concurrently"
echo "to demonstrate cache eviction attacks from GPU on CPU LLC."
echo ""

# Create experiment directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DIR="results/experiment_${TIMESTAMP}"
mkdir -p ${EXPERIMENT_DIR}

echo "Experiment directory: ${EXPERIMENT_DIR}"
echo ""

# ============================================================================
# Experiment 1: Memory Thrashing Attack
# ============================================================================

echo "=== Experiment 1: Memory Thrashing Attack ==="
echo "Starting attacker (mode 2: thrashing)..."

# Start attacker in background
./attacker 2 10000 > ${EXPERIMENT_DIR}/attacker_thrashing.log 2>&1 &
ATTACKER_PID=$!

# Give attacker time to start
sleep 0.5

echo "Starting victim..."
./victim 1000 > ${EXPERIMENT_DIR}/victim_under_thrashing.log 2>&1

# Wait for attacker to complete
wait $ATTACKER_PID

echo "Experiment 1 complete."
echo ""

# ============================================================================
# Experiment 2: Eviction Set Attack
# ============================================================================

echo "=== Experiment 2: Eviction Set Attack ==="
echo "Starting attacker (mode 1: eviction sets)..."

# Start attacker in background
./attacker 1 5000 > ${EXPERIMENT_DIR}/attacker_eviction.log 2>&1 &
ATTACKER_PID=$!

# Give attacker time to build eviction sets
sleep 1.0

echo "Starting victim..."
./victim 1000 > ${EXPERIMENT_DIR}/victim_under_eviction.log 2>&1

# Wait for attacker to complete
wait $ATTACKER_PID

echo "Experiment 2 complete."
echo ""

# ============================================================================
# Analysis
# ============================================================================

echo "=== Extracting Results ==="

# Extract statistics from logs
echo "Baseline (victim alone):"
grep "Mean Latency" docs/baseline_experiment.log || echo "  (Run baseline first)"

echo ""
echo "Under Thrashing Attack:"
grep "Mean Latency" ${EXPERIMENT_DIR}/victim_under_thrashing.log

echo ""
echo "Under Eviction Attack:"
grep "Mean Latency" ${EXPERIMENT_DIR}/victim_under_eviction.log

echo ""
echo "=== Experiment Complete ==="
echo "Results saved to: ${EXPERIMENT_DIR}"
echo ""
echo "To analyze further, examine:"
echo "  - ${EXPERIMENT_DIR}/victim_under_thrashing.log"
echo "  - ${EXPERIMENT_DIR}/victim_under_eviction.log"
echo "  - results/victim_timings_*.csv"
