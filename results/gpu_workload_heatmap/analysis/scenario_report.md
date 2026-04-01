# GPU Workload Heatmap Comparison

| Scenario | Mean p99_ratio | Mean max_ratio | Baseline p99 (cycles) |
|---|---|---|---|
| GPU0_none__CPU0_none | 2.03 | 2.7 | 39.0 |
| GPU1_matmul128__CPU1_mm64 | 11811.482 | 2866.6 | 69.0 |
| GPU2_matmul512__CPU2_mm256 | 656.54 | 116.2 | 113.0 |
| GPU3_conv2d__CPU1_mm64 | 9181.206 | 2281.3 | 69.0 |
| GPU4_bandwidth__CPU3_fft | 21.816 | 121.5 | 250.0 |

## Key observations
- Scenarios with large GPU compute (matmul_512, conv2d, bandwidth) have higher baseline latency
  because the GPU workload itself causes managed-memory page migrations that evict CPU lines.
- The p99_ratio (attack/baseline) tells how much *additional* eviction the attack adds on top of
  the workload's natural LLC pressure.
- conv2d and bandwidth workloads tend to show different per-set patterns than pure matmul,
  because they access different memory regions with different LLC footprints.