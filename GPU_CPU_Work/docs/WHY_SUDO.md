# Why We Need Sudo Access

## Short Answer

We need `sudo` to read **physical addresses** from `/proc/self/pagemap`.  
Without physical addresses, we cannot compute which LLC cache set a memory  
location maps to — and the entire attack depends on targeting the correct set.

---

## The Core Formula

```
LLC set index = (physical_address / 64) % num_sets
```

- `64` = cache line size in bytes
- `num_sets` = 12,288  (for RTX 3060's 12MB LLC: 12288×64×16 = 12MB)

This formula requires the **physical address** — virtual addresses don't work  
because ASLR randomizes the virtual→physical mapping on every run.

---

## Why `/proc/self/pagemap` Needs Root

Linux exposes virtual→physical mapping in `/proc/self/pagemap`.  
Each 8-byte entry encodes:
- Bit 63: page present flag
- Bits 0–54: **Page Frame Number (PFN)**

`physical_address = PFN × page_size + page_offset`

**Since Linux kernel 4.0**, PFN bits are **zeroed for unprivileged processes**  
as a security hardening measure (CVE protection against physical memory probing).

```c
// from timing_utils.h — virtual_to_physical()
uint64_t pfn = entry & ((1ULL << 55) - 1);
if (pfn == 0) return -4;   // <- this happens without root
```

Without root → `pfn = 0` → `vtop()` returns error `-4`  
→ cannot compute LLC set → cannot find congruent addresses → attack fails.

---

## What "Congruent Addresses" Means

For a targeted eviction of LLC set N, we need many cache lines that  
**all map to set N** (same set, competing for the same 16 ways).  
These are called **congruent addresses**.

Without physical addresses, we cannot verify which addresses are congruent.  
We would be evicting random sets instead of the target set → no signal.

---

## What Happens Without vs With Sudo

| | Without sudo | With sudo |
|---|---|---|
| `/proc/self/pagemap` PFN | Always 0 | Correct physical PFN |
| `vtop()` result | Error -4 | Success |
| LLC set computation | Impossible | Exact |
| Congruent address selection | Random guessing | Physically verified |
| Attack signal | Weak/nonexistent | Strong, targeted |

---

## The Code Path (evset_ai_scan.cu)

```c
// For each candidate address in the 64MB managed array:
uint64_t pa = 0;
if (vtop((void*)&data[i], &pa) == 0) {          // needs root for PFN
    uint64_t set = (pa / 64) % num_sets;         // true LLC set
    if (set == target_set) cong.push_back(i);    // collect congruent
}
```

Without root, `vtop()` always fails → `cong` stays empty → binary exits early.

---

## Why Not Just Use Virtual Addresses?

Virtual addresses change every run (ASLR). The LLC set a virtual address  
maps to is determined by the **physical** address bits [11:6] (for 64-byte  
lines in a 12MB/16-way cache). Two virtual addresses that look far apart  
may be congruent physically, and two that look adjacent may not be.

---

## The sudo Command Used

```bash
printf "seal@123\n" | sudo -S -p "" \
    env LD_LIBRARY_PATH="$TORCH_LIB" \
    ./evset_ai_scan models/mobilenet_v2.pt 0 64 64 10000 0 12288 16
```

- `sudo -S` — reads password from stdin (pipe)
- `env LD_LIBRARY_PATH=...` — passes LibTorch path through sudo
  (sudo strips most env vars by default, so we explicitly pass it)
- The binary itself uses root privilege only for `open("/proc/self/pagemap")`

---

## Summary

> Sudo is needed **only** to read physical page frame numbers from  
> `/proc/self/pagemap`. This is a Linux security restriction since kernel 4.0.  
> The CUDA attack itself requires no GPU or system privileges — only the  
> address translation step does.
