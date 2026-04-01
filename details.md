# GPU-CPU Cache Contention Attack: Full Work Log and Study Notes

This document summarizes all work completed during the project session on the remote server:

- Remote host: `ssh rtx`
- Main project path: `/home/rtx3060/Desktop/sumit-GPU-Codes/gpu_cpu_work_1`
- Goal: Demonstrate and analyze a GPU-induced CPU cache contention side-channel attack, then improve rigor with set validation and repeated-trial analysis.

---

## 1. Project Understanding and Initial Study

We inspected all core files in the repo and confirmed architecture:

- `src/hybrid_victim.cu`: primary CPU-side timing victim with concurrent GPU contention
- `src/attacker.cu`: separate attacker modes (thrashing + eviction-style)
- `src/victim.cu`: GPU-side timing baseline (less suitable for CPU LLC impact)
- `src/timing_utils.h`: timing + stats helpers
- `analyze_results.py`: baseline/attack analysis + plots
- `run_attack_experiment.sh`: automation script
- docs (`README.md`, `RESEARCH_FINDINGS.md`, logs)

Key understanding:

- Attack signal is visible when timing is measured on CPU (`rdtsc`) while GPU perturbs shared memory.
- GPU-only timing (`victim.cu`) does not capture CPU LLC side effect strongly.

---

## 2. Build and Baseline Validation


Environment verified:

- GPU: RTX 3060
- CUDA runtime/toolkit available
- UVM/managed memory enabled

Ran core workflow:

1. `./hybrid_victim 0` (baseline)
2. `./hybrid_victim 1` (contention)
3. `python3 analyze_results.py`

Observed strong attack signature:

- Mean and max latency increased significantly under contention
- Rare extreme spikes (tail events) dominated attack signature
- Plots and report generated successfully in `results/`

---

## 3. Important Issue Found in Original Automation

`run_attack_experiment.sh` used `victim` instead of `hybrid_victim`, so it mostly reported stable GPU-clock numbers and did not represent CPU LLC contention quality.

Conclusion:

- Correct demonstration path is the hybrid CPU-timed path, not GPU-only victim timing.

---

## 4. “Same Cache Set” Clarification and Verification Work

You asked whether exact same-set targeting was truly proven.

Initial status:

- Attack effect existed, but exact set congruence was not fully proven in unprivileged runs due pagemap restrictions.

### 4.1 Code upgrades made

#### `src/timing_utils.h`
Added:

- physical address translation via `/proc/self/pagemap`
- helpers for physical set-index computation
- explicit error handling strings for translation failures

#### `src/hybrid_victim.cu`
Added/extended:

- set-validation mode
- congruent-index collection for target set
- targeted GPU kernel on congruent indices
- mapping sample print (probe + target lines with set IDs)
- fallback path if congruent set discovery insufficient

### 4.2 Exact-set proof run

A privileged/root run was executed to bypass PFN restrictions and confirm exact physical mapping.

Results from exact-set run:

- physical target set resolved (`Set 0`)
- many congruent physical lines found (e.g. ~88)
- pagemap failures: 0
- targeted kernel launched on those congruent indices
- CPU latency under attack increased strongly vs baseline

This provided direct mechanistic evidence (same physical set) + behavioral evidence (latency spikes).

---

## 5. Demonstration Batch Runs and Aggregate Evidence

Multiple repeated runs were executed (baseline + attack pairs).

Aggregate behavior observed:

- large mean slowdown (order-of-magnitude class)
- very large max-latency amplification
- rare eviction-signature events (low percentage but strong spikes)

Key interpretation:

- attack is bursty and tail-driven
- median often remains close to baseline
- p99/max carry most of the side-channel signal

---

## 6. NDSS Paper / vGPU-TLB Repo Review and Mapping to This Project

You shared NDSS paper + repo (`vgpu-tlb-exploit`).

Outcome of review:

- Useful for methodology (Prime+Probe rigor, repeated trials, structured per-set analysis)
- Not directly reusable hash formulas for our current CPU LLC target
- Their `get_hash_l2/l3` logic is for GPU TLB behavior in their platform model

So we adapted the methodology, not the exact hash implementation.

---

## 7. New Methodology Features Implemented (Paper-Inspired)

### 7.1 Set sweep support

`src/hybrid_victim.cu` extended with:

- target-set override argument
- configurable GPU attack duration (iterations)

### 7.2 Sweep runner script

Added:

- `run_set_sweep_experiment.sh`

Purpose:

- run baseline+attack for multiple target sets
- run repeated trials per set
- optional sudo/root execution for physical mapping

### 7.3 Repeated-trial confidence analysis

Added:

- `analyze_set_sweep.py`

Outputs:

- per-trial metrics CSV
- per-set summary CSV with 95% CI
- markdown report

Metrics tracked:

- mean ratio
- p99 ratio
- max ratio
- eviction-signature %

---

## 8. Set Sweep Demo Results Produced

Sweep executed for sets like `0, 64, 128` with repeated trials.

Result files:

- `results/set_sweep/set_sweep_trials.csv`
- `results/set_sweep/set_sweep_summary.csv`
- `results/set_sweep/set_sweep_report.md`

Findings:

- strong mean slowdown observed across tested sets
- confidence intervals computed and reported
- tail amplification and eviction event rates quantified per set

---

## 9. Additional Graph Pack Generated

From sweep results, these plots were generated:

- `results/set_sweep/plots/mean_ratio_ci.png`
- `results/set_sweep/plots/tail_ratios_log.png`
- `results/set_sweep/plots/eviction_rate_ci.png`
- `results/set_sweep/plots/trial_scatter_mean_vs_p99.png`

Purpose:

- compare set-to-set effect size
- visualize uncertainty (CI)
- show tail behavior and trial stability

---

## 10. “Per Address in Eviction Set” Plot Request and Fix

You requested plotting timing for all addresses in eviction set.

### 10.1 First attempt

- generated median-based per-address plot
- appeared flat (~constant), ratio near 1

Reason:

- median hides rare spike behavior
- attack is tail-heavy, not median-shift-heavy

### 10.2 Improved per-address scanner

Added new tool:

- `src/evset_address_scan.cu`

It computes per-address:

- baseline median, p99, max
- attack median, p99, max
- ratios for each

Generated outputs:

- `results/evset_address_scan_set0.csv`
- `results/evset_address_scan_set0_ratio_panel.png`
- `results/evset_address_scan_set0_latency_panel.png`

Interpretation of improved plots:

- median ratio ~1.0 (flat)
- p99 mostly near 1 with occasional bump
- max ratio large spikes for subset of addresses

Conclusion:

- same-set contention exists but appears mainly as rare high-latency events

---

## 11. Core Scientific Conclusions from This Work

1. GPU->CPU cache contention attack works in this setup.
2. CPU-side timing (`rdtsc`) is essential for observing signal.
3. Exact same physical LLC set mapping was demonstrated in privileged validation runs.
4. Attack behavior is bursty/tail-driven (max/p99) rather than constant median shift.
5. Repeated trials + CI provide stronger research-grade evidence than single-run snapshots.

---

## 12. Where to Study First

Recommended reading order in remote repo:

1. `src/hybrid_victim.cu`
2. `src/timing_utils.h`
3. `run_set_sweep_experiment.sh`
4. `analyze_set_sweep.py`
5. `src/evset_address_scan.cu`
6. `results/` + `results/set_sweep/` artifacts
7. `docs/EXACT_SET_DEMONSTRATION_2026-02-28.md`
8. `docs/RESEARCH_FINDINGS.md` (updated section)

---

## 13. Suggested Next Steps (if continuing)

1. Increase trial count (e.g., 10+ per set) for tighter CI.
2. Sweep more sets and compare effect heterogeneity.
3. Add CPU affinity and fixed frequency tuning for lower noise.
4. Add hypothesis tests (e.g., Mann-Whitney U on per-trial metrics).
5. Produce final paper-style figures and consolidated appendix table.

---

## 14. Important Note About This Local `details.md`

This file is written in your **local workspace path**:

- `/home/sumitk/Desktop/Semester-8-IIT-KGP/cs47006/gpu_cpu_work_1/details.md`

But most experimental code/runs/artifacts were produced in the **remote server repo**:

- `/home/rtx3060/Desktop/sumit-GPU-Codes/gpu_cpu_work_1`

Use this file as the study map, and open the remote files for raw evidence.




  Within one target cache set, you have many congruent addresses.
  Most show small/no increase, some show medium spikes, and a few show very large spikes.

  That means:

  - contention impact is uneven across addresses in the same set,
  - rare worst-case conflicts create the very high peaks,
  - those high peaks are the strongest attack signal.