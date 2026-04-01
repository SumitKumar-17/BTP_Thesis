# Exact LLC Set Congruence Demonstration (GPU -> CPU Contention)

**Date:** February 28, 2026
**Host:** `rtx` (`/home/rtx3060/Desktop/sumit-GPU-Codes/gpu_cpu_work_1`)
**GPU:** NVIDIA RTX 3060
**Goal:** Prove that GPU attacker and CPU victim are operating on the exact same CPU LLC set using physical-address mapping, then demonstrate timing contention.

## 1. What Was Added

The experiment code was extended to support exact set validation:

- `src/timing_utils.h`
  - `virtual_to_physical()` using `/proc/self/pagemap`
  - `get_cache_set_index_from_phys()` for LLC set index from physical address
- `src/hybrid_victim.cu`
  - validation mode that scans managed-memory cache lines
  - collects lines mapping to the same LLC set as the CPU probe line
  - targeted GPU kernel (`gpu_targeted_set_kernel`) that touches only those congruent indices
  - fixed CPU probe index during validation runs for same-address/same-set measurement

## 2. Run Commands Used

```bash
cd /home/rtx3060/Desktop/sumit-GPU-Codes/gpu_cpu_work_1
/usr/local/cuda/bin/nvcc -O3 -arch=sm_86 -I./src -o hybrid_victim src/hybrid_victim.cu -lpthread

# root run required for physical-address PFN visibility
sudo ./hybrid_victim 0 1 12288 16 | tee results/exact_set_baseline_root.log
sudo ./hybrid_victim 1 1 12288 16 | tee results/exact_set_attack_root.log

python3 analyze_results.py | tee results/exact_set_analysis_stdout.log
```

## 3. Proof of Exact Same-Set Mapping

From `results/exact_set_attack_root.log`:

- `Target set index: 0 (physical)`
- `Congruent candidates found: 88`
- `Candidates with physical mapping: 88`
- `vtop failures while scanning: 0`
- `Launching TARGETED GPU kernel on 88 congruent indices...`

Mapping sample excerpt (all physical set 0):

- Probe: `idx=0`, `PA=0x166e00000`, `Set=0`
- Target[1]: `idx=196608`, `PA=0x166ec0000`, `Set=0`
- Target[2]: `idx=393216`, `PA=0x166f80000`, `Set=0`
- Target[3]: `idx=589824`, `PA=0x11ae40000`, `Set=0`

This demonstrates CPU probe line and GPU target lines are physically congruent in the same LLC set.

## 4. Demonstration Results (Baseline vs Targeted Same-Set Attack)

From `results/hybrid_cpu_timings_baseline.csv` and `results/hybrid_cpu_timings_contention.csv` produced by the above run:

- Baseline mean: `144.63` cycles
- Attack mean: `1196.13` cycles
- Mean slowdown: `8.27x`

- Baseline max: `1084` cycles
- Attack max: `452548` cycles
- Max slowdown: `417.48x`

Eviction-signature events (`latency > 10x attack median`):

- 4 / 1000 accesses (`0.40%`)
- Largest spikes: `452548`, `374854`, `228026` cycles

## 5. Interpretation

This run provides two independent pieces of evidence:

1. **Mechanistic proof (exact same set):**
   Physical-address based mapping confirms CPU probe and GPU attack addresses are in the same LLC set.

2. **Behavioral proof (timing contention):**
   CPU access latencies exhibit large attack-induced spikes (hundreds of thousands of cycles), and overall mean latency rises significantly under targeted GPU activity.

Together, these confirm GPU-to-CPU LLC contention on exact congruent sets in this setup.

## 6. Output Files

- `results/exact_set_baseline_root.log`
- `results/exact_set_attack_root.log`
- `results/exact_set_analysis_stdout.log`
- `results/hybrid_cpu_timings_baseline.csv`
- `results/hybrid_cpu_timings_contention.csv`
- `results/analysis_report.txt`
- `results/plots/timeline.png`
- `results/plots/histogram.png`
- `results/plots/boxplot.png`
