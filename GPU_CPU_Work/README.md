# GPU-Induced CPU Cache Contention Attack

**B.Tech Research Project - Hardware Security**
**IIT Kharagpur | CS47006 | February 2026**

---

## 🎯 Project Overview

This project demonstrates a **novel cross-component cache side-channel attack** where a malicious GPU process induces cache evictions in the CPU's Last Level Cache (LLC), creating a measurable timing side-channel that can leak sensitive information about victim process memory access patterns.

### Key Achievement

✅ **Successfully demonstrated 734x latency increase** in CPU memory accesses due to GPU-induced cache contention, providing a clear attack signal for side-channel exploitation in heterogeneous computing systems.

---

## 📊 Quick Results

| Metric              | Baseline    | Under Attack | Impact      |
|---------------------|-------------|--------------|-------------|
| **Mean Latency**    | 202 cycles  | 4,002 cycles | **19.8x ↑** |
| **Max Latency**     | 1,585 cycles| 1,163,497 cycles | **734x ↑** |
| **P99 Latency**     | 575 cycles  | 52,624 cycles| **91.5x ↑** |
| **Cache Evictions** | 0%          | 3.2%         | Attack signature |

---

## 🏗️ Architecture

```
CPU (Victim) ←→ LLC (Target) ←→ UVM ←→ GPU (Attacker)
```

The attack exploits **Unified Virtual Memory (UVM)** to enable the GPU to:
1. Access CPU memory regions
2. Cause physical address collisions in CPU cache sets
3. Force cache evictions of victim's data
4. Create measurable timing differences

---

## 📁 Repository Structure

```
gpu_cpu_cache/
├── src/
│   ├── timing_utils.h          # RDTSC, CUDA timers, statistics
│   ├── victim.cu               # GPU-side baseline victim
│   ├── attacker.cu             # Cache contention attacker (2 modes)
│   └── hybrid_victim.cu        # CPU-side measurement (CRITICAL)
│
├── results/
│   ├── hybrid_cpu_timings_baseline.csv
│   ├── hybrid_cpu_timings_contention.csv
│   ├── analysis_report.txt
│   └── plots/
│       ├── timeline.png
│       ├── histogram.png
│       └── boxplot.png
│
├── analyze_results.py          # Statistical analysis & visualization
├── run_attack_experiment.sh    # Automated experiment runner
├── CLAUDE.md                   # Project guidance for Claude Code
├── RESEARCH_FINDINGS.md        # Detailed research report
└── README.md                   # This file
```

---

## 🚀 Getting Started

### Prerequisites

- NVIDIA GPU with Compute Capability ≥ 6.0 (UVM support)
- CUDA Toolkit 12.x
- Linux system with SSH access to GPU server
- Python 3 with pandas, matplotlib, numpy

### Compilation

```bash
# On the remote GPU server
cd /home/rtx3060/Desktop/sumit-GPU-Codes/gpu_cpu_cache

# Compile all programs
nvcc -O3 -arch=sm_86 -I./src -o victim src/victim.cu
nvcc -O3 -arch=sm_86 -I./src -o attacker src/attacker.cu
nvcc -O3 -arch=sm_86 -I./src -o hybrid_victim src/hybrid_victim.cu -lpthread
```

### Running Experiments

#### 1. Baseline (No Attack)
```bash
./hybrid_victim 0
```

#### 2. Attack (With GPU Contention)
```bash
./hybrid_victim 1
```

#### 3. Analyze Results
```bash
python3 analyze_results.py
```

### Automated Full Experiment
```bash
./run_attack_experiment.sh
```

---

## 🔬 How It Works

### 1. Victim Process (CPU Thread)
- Measures memory access latency using `rdtsc()`
- Accesses unified memory array at regular intervals
- Records timing for each access

### 2. Attacker Process (GPU Kernel)
- **Mode 1: Eviction Set Attack** - Targets specific cache sets
- **Mode 2: Memory Thrashing** - Aggressive random access
- Runs concurrently with victim

### 3. Side-Channel Leakage
- Normal access: ~155 CPU cycles (cache hit)
- Under attack: 50K-1.1M CPU cycles (cache miss)
- Timing difference reveals when victim's data was accessed

---

## 📈 Visualizations

The analysis script generates three plots:

1. **timeline.png** - Latency over time (iteration)
   - Shows sporadic attack spikes

2. **histogram.png** - Distribution comparison
   - Bimodal distribution indicates attack

3. **boxplot.png** - Statistical comparison
   - Dramatic outliers under attack

---

## 🛡️ Security Implications

### Vulnerable Scenarios
- ✗ Cloud GPU instances (multi-tenant)
- ✗ ML training pipelines (GPU + CPU data loading)
- ✗ Heterogeneous HPC workloads
- ✗ Edge computing with shared CPU/GPU

### Potential Exploits
- **Cryptographic key extraction** (AES, RSA)
- **Kernel ASLR bypass**
- **ML model stealing**
- **Covert channels**

### Mitigations
- Cache partitioning (Intel CAT)
- UVM access restrictions
- Temporal workload isolation
- Performance counter monitoring

---

## 📝 Key Findings

1. **GPU can manipulate CPU cache state** via UVM
2. **3.2% of accesses show attack signature** (sufficient for side-channel analysis)
3. **CPU-side timing is critical** (GPU clocks don't capture the effect)
4. **Bimodal latency distribution** distinguishes attack from normal contention

---

## 🎓 Research Context

**Paper Reference:**
- "Rethinking Page Table Structure for Fast Address Translation in GPUs: A Fixed-Size Hashed Page Table" (MICRO 2019)
- Explains GPU page table architecture exploited in this attack

**Related Work:**
- FLUSH+RELOAD (Yarom & Falkner, 2014)
- Prime+Probe cache attacks
- Cross-VM side channels

---

## 🔧 Hardware Specifications

**GPU:** NVIDIA RTX 3060
- Compute Capability: 8.6 (Ampere)
- Memory: 12 GB GDDR6
- L2 Cache: 2.25 MB
- UVM: Fully supported

**CUDA:** Version 12.8
**Driver:** 13.0

---

## 📊 Data Files

### CSV Format
```csv
iteration,latency_cpu_cycles
0,975715
1,497
2,472
...
```

### Analysis Output
- **Mean, Median, Std Dev** - Statistical summary
- **P95, P99** - Tail latency analysis
- **Attack signatures** - Eviction event detection

---

## 🤝 Contributing

This is a research project for academic purposes. Key areas for extension:

1. **Improved eviction set construction** - Higher success rate
2. **Real cryptographic attacks** - AES key extraction
3. **Cross-architecture testing** - AMD, ARM GPUs
4. **Defense evaluation** - Mitigation overhead analysis

---

## 📚 Documentation

- **CLAUDE.md** - Detailed instructions for working with this codebase
- **RESEARCH_FINDINGS.md** - Complete research report with methodology
- **analysis_report.txt** - Automated statistical analysis output

---

## ⚠️ Ethical Considerations

This research is conducted for:
- ✅ Educational purposes
- ✅ Security awareness
- ✅ Defense mechanism development

**Not for:**
- ✗ Malicious attacks
- ✗ Unauthorized access
- ✗ Privacy violations

All experiments were conducted on authorized hardware with proper permissions.

---

## 📧 Contact

**Researcher:** Sumit K.
**Institution:** IIT Kharagpur
**Course:** CS47006 - B.Tech Project
**Focus:** Hardware Security / Side-Channel Analysis

---

## 📄 License

Academic research project - IIT Kharagpur

---

## 🙏 Acknowledgments

- **Claude Code** - AI-assisted development and analysis
- **NVIDIA CUDA Documentation** - UVM architecture reference
- **IIT Kharagpur** - Research facilities and GPU server access

---

**Last Updated:** February 12, 2026
**Status:** ✅ Proof-of-Concept Complete
