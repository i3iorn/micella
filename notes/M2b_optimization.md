# M2b — kernel optimization round: what was tried, what won

Baseline after M2: int8 factors + CUDA graph = 483 us/layer (all 7 matrices,
main+residual pairs) -> 57.5 tok/s implied, ~214 GB/s effective factor reads
(above the ~168 GB/s DRAM spec thanks to L2 hits on x/h/scales).

Since the int8 path is already ABOVE DRAM bandwidth, the compute units (SIMD
FMA, tensor cores) are idle - more compute buys nothing. Only fewer bytes can
speed this up. So the optimization attempts were:

## 1. Tensor cores / tl.dot — skipped (justified)
The op is a GEMV chain at M=1: bandwidth-bound, not FLOP-bound. Tensor cores
would add bytes (need M>=16 tiles => replicating x 16x), not remove them.

## 2. 1-bit packed factors, unpacked in-kernel — tried, LOSES in Triton
Rationale: factors are 1 bit on disk; keeping them packed in VRAM is 8x fewer
bytes (65 MB/layer -> 8 MB). But unpacking bits inside Triton costs more than
the saved bandwidth:

| variant | layer time | note |
|---|---|---|
| 3D broadcast expansion (w>>b, reshape) | 8842 us | register blowout, 18x slower |
| 2D per-bit static_range loop | 782-1501 us (unstable) | best ~1.6x slower than int8 |
| int8 (unchanged) | 483 us | still the winner |

num_warps sensitivity is extreme on the packed U kernel: (4,4) = 2852 us per
gate pair vs (4,8) = 131 us - 22x swing. Even at best config, packed never
beats int8 in graph mode. Conclusion: revisit packing in hand-written CUDA
(LOP3/PRMT bit tricks or int8 tensor-core dot) later; it matters mainly for
VRAM CAPACITY (M4: 30B MoE at 0.1 bpw -> ~1 GB int8 vs ~125 MB packed), not
for speed on this checkpoint.

## 3. Measurement gotchas found
- Laptop GPU clock ramp: cold runs (SM clock 1492 MHz idle -> 1972+ under
  load) made one bench read 3x slower than a repeat run. Always warm up; never
  trust a single cold number.
- Standalone (non-graph) repeated timing of one pair keeps its factors L2-hot
  and flatters it; only graph-level (whole layer) numbers are meaningful.

## Current best configuration (locked in)
- int8 factors resident in VRAM, hoisted s_vu = v1*u2, reused h/y buffers
- 2 kernels per factor pair (V-reduce, U-reduce), num_warps 4/4
- whole layer captured in one CUDA graph
- 57.5 tok/s implied at 0.55 bpw (matmul floor only, no attention/norms yet)

The 50 tok/s target is met at the matmul floor for the 8B @0.55bpw checkpoint.
Next (M2.5) wires this into the real model to get the true end-to-end number.
