Big Picture

  The CPU is the victim.
  It repeatedly does some real CPU work, then measures how long one memory access takes.

  The GPU is the attacker.
  It does some real GPU work, and while doing that, it also keeps accessing memory lines chosen to collide with the CPU victim in the CPU LLC.

  If the GPU activity pushes the CPU’s victim line out of cache, the next CPU probe becomes slow. That slow probe is the attack signal.

  What the CPU side does

  In hybrid_victim.cu:

  - cpu_measurement_thread() is the victim thread.
  - Before each timed probe, it runs cpu_matrix_workload_step(i).
  - That function does a small 16x16 matrix multiplication on CPU.
  - After that compute work, it times one load from g_data[index] using rdtsc().

  So one CPU iteration is:

  1. Do real CPU compute.
  2. Read one chosen memory address from shared managed memory.
  3. Measure cycles for that single read.
  4. Store latency.

  That gives a timeline of CPU memory access latency under normal vs attacked conditions.

  What the GPU side does

  There are two GPU attack kernels now:

  1. gpu_compute_contention_kernel(...)

  - Broad attack.
  - Does a 128x128 matrix-multiply-style compute workload on GPU.
  - Also touches the whole shared array.
  - This creates wide cache/memory pressure.

  2. gpu_targeted_set_kernel(...)

  - Targeted attack.
  - Also does the GPU matrix workload.
  - But for contention, it only touches addresses from one congruent set list.

  So the GPU is not just “spinning” anymore.
  It is doing actual compute plus cache-pressure accesses.

  Which sets are being attacked

  That depends on the run arguments.

  The code accepts:

  - target_set_override
  - cache_size_kb
  - cache_ways

  In your run:

  - cache_size_kb = 12288
  - cache_ways = 16

  So total sets are computed as:

  sets = (cache_size_kb * 1024) / (64 * ways)
       = (12288 * 1024) / (64 * 16)
       = 12288 sets

  In the run you did:

  - target_set_override = 0

  So the attack was aimed at LLC set 0.

  How the code finds those target addresses

  Function:

  - collect_congruent_indices(...)

  It scans the managed array g_data.
  For each cache-line-spaced element, it computes a set index.
  If that set index equals the target set, it stores that index in congruent_indices_host.

  So this builds a list like:

  - address A -> set 0
  - address B -> set 0
  - address C -> set 0
  - ...

  Those are different memory lines, but all mapping to the same LLC set.

  How many sets are attacked

  In targeted mode:

  - Only one target set is attacked at a time.
  - But within that set, many addresses are touched.

  In your recent run:

  - target set = 0
  - congruent candidates found = about 85

  So:

  - not 85 sets
  - 1 set
  - about 85 addresses inside that same set

  How many addresses does the GPU touch

  In targeted mode:

  - all addresses in the congruent list
  - in your run, about 85

  The GPU kernel loops over those indices every iteration.

  So the attacker behavior is:

  1. Do matrix multiply work on GPU
  2. Touch all ~85 congruent addresses from set 0
  3. Repeat for gpu_iterations

  In your run:

  - gpu_iterations = 500

  So it keeps revisiting those same-set addresses many times.

  What the CPU probes

  The CPU usually measures one probe address:

  - g_probe_index

  When validation is enabled, the code sets:

  - g_probe_index = congruent_indices_host[0]

  So the victim probe is one address from the same target set.

  That means:

  - CPU probes one line in set 0
  - GPU hammers many other lines in set 0

  This creates set-associativity pressure.
  If enough competing lines are active, the CPU’s probe line gets evicted.

  Why spikes do not happen every iteration

  Because eviction is not guaranteed on every probe.

  The spike depends on:

  - when GPU touched the set
  - whether the CPU line was still resident
  - scheduling overlap
  - replacement behavior in the cache

  So the pattern is:

  - many normal probes
  - some very slow probes

  That is why the attack is bursty.

  What “which sets” means here

  Current code can do two modes conceptually:

  1. Probe-derived set

  - attack the same set as the original probe address

  2. Override set

  - explicitly choose a set number, like 0, 64, 128

  You already used this in set-sweep experiments.

  So the framework can attack:

  - one chosen set per run
  - different sets across runs

  What the GPU is attacking exactly

  Strictly speaking, the GPU is not “attacking the CPU core.”
  It is attacking the CPU LLC residency of the victim line.

  More precisely:

  - target = CPU LLC set occupancy
  - mechanism = GPU accesses to congruent managed-memory lines
  - effect = victim line eviction / miss / delayed reload
  - observation = higher CPU probe latency

  Simple mental model

  Think of one cache set as a small shelf with limited slots.

  - CPU keeps one important book on that shelf.
  - GPU keeps placing many other books onto the same shelf.
  - Sometimes CPU’s book gets pushed off.
  - When CPU asks for it next time, it must fetch it from farther away.
  - That extra fetch time is your latency spike.

  In your current run

  The code was effectively doing:

  1. CPU:

  - 16x16 matrix multiply
  - then timed probe of one address in set 0

  2. GPU:

  - 128x128 matrix multiply workload
  - plus repeated accesses to ~85 addresses also mapped to set 0

  3. Result:

  - baseline mean around 140 cycles
  - attack mean around 9074 cycles
  - max around 1.64M cycles

  So yes, with this code, the GPU is attacking one chosen LLC set at a time, using many congruent addresses inside that set.