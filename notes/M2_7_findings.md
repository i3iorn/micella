# M2.7 — manual fusion round (after torch.compile was ruled out)

Changes: fused RMSNorm Triton kernel (145 modules, decode path, fp32 accum),
epilogue-fused residual (the residual pair's U-kernel adds the main pair's
output inside the kernel, removing one add per quantized matrix).

## Results (benchmarks/results/m25b_graph.json, Qwen3-8B @0.55bpw, RTX 3050)

| stage | graph tok/s | per token |
|---|---|---|
| M2.5 (RoPE tables + Triton lm_head) | 11.20 | 89.3 ms |
| M2.7 (+fused RMSNorm +epilogue fusion) | **12.97** | **77.1 ms** |

+16%. Smaller than the eager profiler suggested: the elementwise soup costs
much less inside a CUDA graph than the eager profiler's per-kernel overhead
made it look. Eager-mode speed DROPPED (4.4 -> 3.95) because the RMSNorm patch
adds Triton-launch Python overhead per call - only the graph path matters now.

## CRITICAL bug found: Triton cache poisoning (3rd cache incident, worst one)
The new _scale_u_res_kernel ran from a cache entry that produced SILENTLY
WRONG numerics (output == 0, rel err 1.0) while still being fast. The m25b
"regression" to 2.1-2.2 tok/s was caused by this poisoned cache, not by the
patches: after `rm -rf ~/.triton/cache` the same code measured 12.97 tok/s.
Rule going forward: after ANY kernel source change, clear the cache and
validate numerics before trusting any measurement. A "slowdown" or "wrong
result" with bit-identical behavior across runs = clear the cache first.

## Current state (12.97 tok/s = 77 ms/token)
- factored matmuls: ~17 ms (M2 floor)
- attention (mem-efficient sdpa): ~22-32 ms - biggest remaining op; flash
  unavailable on Windows torch 2.6
- soup after fusion: ~10-15 ms
- lm_head (Triton GEMV): ~7 ms
- remainder: KV-cache ops, mask handling, graph replay overhead

## Next (M2.8)
Custom M=1 GQA decode-attention kernel reading the StaticCache KV directly
(replaces sdpa). Expected -15 to -25 ms/token -> ~16-18 tok/s on this
checkpoint. Note eager-mode numbers are no longer meaningful; always measure
the graph path.
