# M2 findings — fused factored GEMV kernels (Triton, int8 factors)

Artifacts: kernels/factored_gemm.py, kernels/bench_factored.py,
kernels/verify_graph.py, kernels/results/m2_microbench.json.

## Results (RTX 3050 6GB, decode M=1, real Qwen3-8B@0.55bpw shapes)

| mode | per layer | per token (x36) | implied tok/s |
|---|---|---|---|
| torch bf16 factors (cast-free) | 2.1 ms | 75.7 ms | 13.2 |
| triton int8, eager | 0.83 ms | 30.1 ms | 33.3 |
| **triton int8 + CUDA graph** | **0.305 ms** | **11.0 ms** | **91.2** |

Correctness: max rel err 0.3% vs fp32 reference on all four shapes, verified
both eager and inside a replayed CUDA graph. Effective factor-read bandwidth in
graph mode: ~214 GB/s (L2 hits on x/h make it exceed the ~168 GB/s DRAM spec).

The 50 tok/s target is beaten at the kernel level for THIS 0.55bpw checkpoint
(91 tok/s matmul-only floor); at 0.1 bpw the factor set shrinks ~5x, leaving
enormous headroom for attention/KV/norms/sampling overhead.

## What the numbers say
- Eager mode is WDDM-launch-bound: ~59 us per factor pair regardless of size
  (2 launches). Per-token eager = 504 launches x overhead. CUDA graphs remove
  it: 28 launches captured into one graph replay at ~10 us each of real work.
- Big shapes (gate_up, down) reach ~110 GB/s eager, 214 in-graph. Small shapes
  (kv) are pure overhead in eager, free in a graph.

## triton-windows 3.9 pitfalls (both bit us hard, keep for all future kernels)
1. **Use-after-free of argument temporaries**: launching with an inlined
   temporary (`kernel[grid](x, v1 * u2, ...)`) lets the intermediate be freed
   and its memory reused while the async kernel still reads it -> zeros/NaN,
   non-deterministic. Always bind temporaries to a variable that outlives the
   launch. (Triton on Windows does not pin arg tensors with the allocator.)
2. Masked OOB: any BLOCK division not dividing the dim needs explicit masks;
   unmasked loads of the tail block silently produce garbage (no crash).
3. Triton cache can serve stale binaries after source edits?? — happened once
   (identical wrong value before/after a mask fix); cleared with
   `rm -rf ~/.triton/cache`. If a fix "does nothing", clear the cache first.
4. CUDA graph capture of Triton launches works (torch.cuda.graph), but all
   buffers must be pre-allocated and stable across replays.

## Design facts for M2.5 (integrate into the real model)
- s_vu = v1*u2 is a per-layer constant: precompute once at load, never per
  token.
- h/y buffers: allocate once per matrix, reuse per token (graph-compatible).
- A full decode step = 36 layers x (7 matrices x 2 pairs x 2 kernels) = 1008
  kernels; graph-capture the whole step. Attention (sdpa), norms, residual
  adds, embedding lookup and lm_head must run as torch ops captured in the
  same graph — torch ops are graph-safe; only buffer addresses must be static.
- Greedy sampling keeps the graph static; top-k/top-p sampling changes control
  flow per token — for graphs, sample on a fixed-shape logits buffer.

## Next
M2.5: replace LittleBitLinear.forward with the int8+graph path inside the real
model, measure end-to-end decode tok/s on the 8B checkpoint. That number is
the honest baseline for the M3/M4 work.
