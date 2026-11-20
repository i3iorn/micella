# M2.6 — torch.compile: negative result on this stack

Target: fuse the ~30-40 ms/step elementwise soup (norms, residuals, rope
indexing, attention glue) that sits between our factored-matmul kernels.

## Results (benchmarks/results/m26*.json)

| strategy | tok/s | vs fast-eager 4.68 | compile time |
|---|---|---|---|
| whole model, dynamic=False | 0.20 | 0.04x (25x SLOWER) | 246 s |
| per-decoder-layer (36x) | 0.30 | 0.06x (16x SLOWER) | 555 s |

Tokens match eager in both cases - correctness fine, performance catastrophic.

## Why it fails here (diagnosed)
1. Our 252 custom Triton kernel calls inside the forward force graph breaks at
   every LittleBitLinear; compiled fragments become tiny and numerous.
2. HF 4.51's dynamic-cache path (StaticCache.update, cache_position handling)
   triggers guards/recompiles when compiled at whole-model granularity.
3. WDDM + Windows: inductor's generated code gets no launch-overhead relief
   (same ~30 us/launch tax), and small fragments amplify it.
4. inductor printed "Not enough SMs to use max_autotune_gemm mode" - the
   3050 laptop's 20 SMs are below its tuning threshold.

## Conclusion
torch.compile (torch 2.6.0+cu124, Windows, custom-Triton-interleaved model) is
NOT a viable fusion path for this project. The elementwise soup must be fused
manually. Next candidates (M2.7), in expected-value order:
1. Fused RMSNorm Triton kernel (73 tiny pow/mean/rsqrt kernels per step -> 36
   launches, one table-safe implementation).
2. Fuse bias-add + residual-add into the U-kernel epilogue (removes ~500
   add/mul kernels per step).
3. Custom M=1 GQA decode attention kernel (replaces 22-32 ms sdpa).
Combined realistic target: ~89 ms -> ~45-55 ms/token (~18-22 tok/s) on this
0.55bpw checkpoint.

## Environment notes for any future compile attempt
- Inductor needs MSVC: run under benchmarks/run_msvc.bat (vcvars64).
- torch 2.6 needs triton-windows==3.2.x (3.9 removed AttrsDescriptor that
  torch._inductor.runtime.hints imports).
