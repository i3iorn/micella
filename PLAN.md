# Project Micella

**Goal**: 50 tok/s decoding a 30B+ MoE model on an RTX 3050 6GB laptop GPU
(Ampere, ~4.4 TFLOPS fp16, ~180 GB/s VRAM bandwidth; Ryzen 7840HS, 16 GB RAM,
1 TB NVMe).

## The physics

Naive approaches are bandwidth-dead on this hardware:

| approach | bytes read per token | max tok/s on this machine |
|---|---|---|
| 30B-A3B fp16 | ~6 GB | 0.03 (VRAM) / 0.01 (RAM) |
| int4 weights | ~1.5 GB | ~0.1 (VRAM) |
| disk-streamed int4 (colibri) | ~1.5 GB | ~0.002 (NVMe 3.5 GB/s) |

LittleBit factorization at eff_bit 0.1 shrinks the per-token active weight read
to ~37 MB (factors of ~3B active params, bf16). Two consequences:

1. **The whole compressed model's factors (~0.4-1 GB for 30B-A3B) fit in 6 GB
   VRAM at once.** No expert streaming needed - the opposite of colibri's design.
2. Per-token bandwidth drops ~40x vs int4; VRAM bandwidth stops being the
   bottleneck, FLOPs do not (0.3 TFLOPS needed vs ~4 available).

The catch: the factored forward is TWO chained GEMMs
`y = (((x*v2) @ V^T) * (v1*u2)) @ U^T * u1` (~2x FLOPs) and nobody ships
LittleBit-quantized MoE checkpoints, nor a fast kernel for this op.

## Language roles

- **Python** - research loop: model loading, quantization experiments, eval
  harness, Triton kernel authoring. PyTorch/transformers only exist here.
- **Triton (Python-embedded) -> CUDA C** - the fused factored-matmul kernel.
  Prototype in Triton (minutes per iteration), port the final version to
  CUDA C/CUTLASS only if Triton leaves >20% on the table.
- **Rust** - the standalone inference runtime/CLI (safetensors mmap, tokenizer,
  generation loop, kernel dispatch) once the kernel design is frozen. Rust is
  the wrong tool for GPU kernels today and the right one for a small
  dependency-free long-lived engine binary.
- **C** - fallback only: if CUTLASS/CUDA port is needed, or a CPU int8 path for
  prefill offload.

## Progress log (run notes in notes/, results in benchmarks/results/)
- M1 done: stock 8B@0.55bpw = 1.7 tok/s (9.46 GB, WDDM spill), int8 5.87 GB.
- M2 done: fused int8 factor kernels + CUDA graph = 57.5 tok/s matmul floor.
- M2b done: 1-bit packed factors LOSE to int8 in Triton (documented); int8+graph locked in.
- M2.5/M2.5b done: fast path in real model, whole-step CUDA graph, RoPE tables,
  Triton lm_head -> 11.2 tok/s end-to-end (5.9 GB), tokens verified identical.
- M2.6 done (NEGATIVE): torch.compile fails on this stack (whole model 0.2
  tok/s, per-layer 0.3 tok/s vs 4.68 eager; graph breaks at 252 custom Triton
  kernels + dynamic-cache recompiles + WDDM launch tax). Elementwise fusion
  must be manual -> M2.7 (fused RMSNorm, epilogue-fused residual/bias, custom
  M=1 GQA attention kernel).
- M2.7 done: fused RMSNorm + epilogue-fused residual -> 12.97 tok/s (77 ms)
  graph decode, +16% over M2.5. Found 3rd Triton-cache incident (silent wrong
  numerics); cache-clear + numeric validation now mandatory after kernel edits.
- M2.8 done: custom M=1 GQA decode-attention kernel (registry-over-sdpa,
  graph-safe device-side length via mask tensor) -> 14.1 tok/s (70.9 ms),
  +8.7%; cumulative 8.3x over stock. Found+fixed: unmasked tail chunk OOB
  (kernel fine only at block-multiple sizes) and a benchmark bootstrap bug
  that had corrupted ALL earlier reference decodes. Single-op wins are now
  - M3 in progress: post-training pipeline built (tools/posttrain.py + eval_ppl.py);
    NEGATIVE so far - naive SVD(+ITQ) post-training gives ppl 5e7 vs official QAT
    22.9 (bf16 0.6B baseline 19.6). Root cause: SVD signs near-orthogonal to W;
    weight-space calibrations don't help. Next: activation-aware scale
    calibration (fit u1_k per row on captured activations), then ASVD-style
    whitening. See notes/M3_findings.md.

## Milestones

### M1 - Baseline: stock LittleBit 8B on the 3050  (this task)
Load `LittleBitLLM/littlebit-qwen3-8b-0.55bpw` with the stock repo loader
(`quantization.hub.LittleBitModel`), measure:
- VRAM: allocated / reserved after load, factor bytes vs embed+lm_head bytes
- decode tok/s: stock PyTorch two-matmul path, greedy, 128 generated tokens,
  batch 1, bf16
- prefill tok/s for reference

Exit criteria: real numbers in `benchmarks/results/m1_baseline.json`, generation
works end-to-end. Everything later is measured against this.

Known risk: naive load wants ~10 GB VRAM (factors 7.7 GB bf16 + embed/lm_head
2.5 GB) > 6 GB. Mitigation ladder, cheapest first:
1. int8 factor storage + per-forward cast to bf16 (halves factor memory)
2. CPU-offload embed_tokens + lm_head (~2.5 GB), only factors on GPU
3. combination of both => ~6.3 GB -> fits

### M2 - Fused factored-matmul kernel
Triton kernel computing both chained GEMMs + scale products in one launch,
factors read from int8/bf16 in VRAM. Benchmark vs M1's PyTorch path.
Target: >=5x M1 decode tok/s on the same checkpoint.

### M3 - Quality: post-training factorization without QAT
The official checkpoints are QAT-trained with a 30B+ teacher (impossible here).
Research question: how much quality survives with (a) plain SVD init,
(b) activation-aware SVD (ASVD/Whitened-SVD style), (c) cheap rank-1 residual
refinement, all teacher-free, at 0.1-0.3 bpw on Qwen3-4B.
Deliverable: perplexity table official-QAT vs post-training at matched bpw.

### M4 - 30B+ MoE end-to-end
Pick target (Qwen3-Coder-30B-A3B / Qwen3.6-35B-A3B). Factorize all expert
linears (M3 method), all factors resident in VRAM, M2 kernel for every matmul,
KV cache budgeted. Measure. Iterate on bpw vs quality vs tok/s until either
50 tok/s or the wall is clearly identified.

## Non-goals (explicitly)
- Training LittleBit checkpoints with QAT on this machine.
- Expert streaming from disk (colibri path) - bandwidth analysis above shows
  factors-in-VRAM dominates it at this bpw.
- Multi-GPU, quantized KV cache beyond what PyTorch gives for free, speculative
  decoding (phase 2+ if M4 lands near target).

## Environment
- venv: `C:/Users/micro/.venv-littlebit` (torch 2.6.0+cu124, transformers
  4.51.3 - pinned by LittleBit README; see skill latent-factorization for
  install commands and pitfalls)
- checkpoints: `models/`
- results: `benchmarks/results/`

## STATUS: CONCLUDED (2026-11-20)

M1/M2 delivered (8.3x runtime, 14.1 tok/s on 8B@0.55bpw, 5.9GB VRAM).
M3 closed with a documented negative result + partial revival (activation-space
scale fitting); M4 kill criteria fired per PLAN-M4.md gate 8. Final tag: v1.0.
Successor project: Wren (Code/Projects/wren) - the opposite direction: take a
modern 3-5B model and make it as capable as possible (QLoRA + distillation +
eval-gated iteration on the same 6GB GPU).
