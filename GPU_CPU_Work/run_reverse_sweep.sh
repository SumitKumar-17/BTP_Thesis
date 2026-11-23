#!/bin/bash
# run_reverse_sweep.sh
# Sweep 16 evenly-spaced LLC sets for the reverse prime+probe experiment
# (CPU attacker, GPU-UVM victim)
#
# Usage: bash run_reverse_sweep.sh
# Outputs: reverse_pp_set<N>.csv for each set

LLC_KB=12288      # RTX 3060 host LLC size in KB
ASSOC=16          # associativity
REPS=100          # repetitions per set
N_POINTS=32       # congruent lines per set

N_SETS=$(( LLC_KB * 1024 / (ASSOC * 64) ))   # 12288
STEP=$(( N_SETS / 16 ))                        # 768

echo "LLC: ${LLC_KB} KB, ${ASSOC}-way, ${N_SETS} sets"
echo "Sweeping 16 sets with step ${STEP}"
echo "---"

for i in $(seq 0 15); do
    SET=$(( i * STEP ))
    echo "[$(date +%T)] Set ${SET} ..."
    sudo ./reverse_prime_probe ${SET} ${N_POINTS} ${REPS} ${LLC_KB} ${ASSOC}
done

echo "---"
echo "Done. CSV files: reverse_pp_set*.csv"
