# M2.5/M2.5b — integrated model results (Qwen3-8B @ 0.55bpw, RTX 3050)

End-to-end greedy decode, same StaticCache loop for all variants:

| stage | tok/s | per token | VRAM |
|---|---|---|---|
| M1 stock loader (spilled to RAM) | 1.70 | 588 ms | 9.46 GB |
| fast kernels eager (int8 factors, graph-ready buffers) | 4.10 | 244 ms | 5.87 GB |
| + whole-step CUDA graph | 9.68 | 103 ms | 5.87 GB |
| + RoPE lookup tables + Triton lm_head GEMV | **11.2** | **89 ms** | 5.89 GB |

Cumulative: **6.6x** over the stock baseline. Greedy tokens match eager at every
stage (graph replay verified byte-identical token ids).

## Profiler findings (what eats the 89 ms)
- factored matmuls: ~17 ms (the M2 floor - no longer dominant)
- mem-efficient attention: ~22-32 ms (36 layers x 614 us - GQA decode path)
- elementwise/norm/reshape soup: ~30-40 ms (hundreds of tiny kernels per step)
- lm_head: was 32 ms via cublas (39 GB/s!), now ~7 ms via Triton GEMV
  (reads 1.24 GB once)
- RoPE: was ~16 ms/step of pow/sin/cos per step, now ~0 (precomputed tables,
  ONE shared copy across layers - HF puts a single rotary on the model)

## Engine facts learned
- Module.to() MOVES params and buffers independently: a buffer aliasing a
  parameter's storage gets duplicated (+3.8 GB bug). Never register param
  aliases as buffers; read `param.data` in forward.
- Instance-level forward patching requires types.MethodType (plain function
  assignment loses self and breaks Module.__call__ dispatch).
- Flash-attention backend is NOT available in torch 2.6.0+cu124 on Windows
  ("No available kernel"); mem-efficient is the ceiling for sdpa here.
- fill_ ops captured inside a CUDA graph replay their capture-time constants -
  token/position buffer updates must be eager ops between replays.
- Patched overheads ran on both eager and graph path -> the eager/graph token
  comparison stays a valid correctness gate.

## Where the next wins are (in order)
1. torch.compile (inductor + triton-windows) to fuse the elementwise soup -
   potentially the single biggest remaining chunk (~30-40 ms).
2. Custom attention kernel for M=1 GQA decode (KV-cache-aware GEMV over the
   cache) - replaces the 22-32 ms sdpa.
3. Later (M4 capacity): 1-bit packed factors in hand-written CUDA.
