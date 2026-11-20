# micella — sub-1-bit LLM inference on a 6 GB laptop GPU

Experiments with SamsungLabs' [LittleBit](https://github.com/SamsungLabs/LittleBit)
sub-1-bit LLM compression (NeurIPS'25 / ICML'26): how much inference speed can
be extracted from a LittleBit checkpoint on a 6 GB RTX 3050 laptop GPU, and
whether the compression itself can be reproduced without Samsung's QAT
training pipeline.

Hardware baseline: RTX 3050 6 GB Laptop (Ampere sm_86), Ryzen 7 7840HS,
16 GB RAM, Windows 11, WDDM.

## Headline results

End-to-end greedy decode of `LittleBitLLM/littlebit-qwen3-8b-0.55bpw`
(Qwen3-8B compressed to 0.55 bits/weight), tokens verified against eager at
every stage:

| stage | tok/s | per token | VRAM |
|---|---|---|---|
| stock LittleBit loader (M1) | 1.70 | 589 ms | 9.5 GB (WDDM spill) |
| + fast kernels, eager | 4.10 | 244 ms | 5.9 GB |
| + whole-decode-step CUDA graph | 9.68 | 103 ms | 5.9 GB |
| + RoPE tables + Triton lm_head | 11.20 | 89 ms | 5.9 GB |
| + fused RMSNorm + epilogue-fused residual | 12.97 | 77 ms | 5.9 GB |
| + custom M=1 GQA attention kernel (**M2.8**) | **14.10** | **71 ms** | 5.9 GB |

**8.3x over the stock pipeline.** Matmul-only floor at this checkpoint:
~57 tok/s implied (bandwidth-bound at ~214 GB/s effective factor reads).

Matmul-floor scaling: at 0.55 bpw the full 7-matrix/layer factor set is
3.85 GiB int8; the whole compressed model fits in 6 GB VRAM at once - no
expert streaming needed (the opposite of colibri-style disk streaming, which
loses badly at these bit widths: ~0.002 tok/s from an NVMe).

## What is here

- `kernels/` - Triton kernels: factored two-GEMV chain (`factored_gemm.py`),
  fused RMSNorm, epilogue-fused residual, M=1 GQA decode attention with
  online softmax over the StaticCache, Triton lm_head GEMV, and the model
  fast-path patching (`fast_model.py`). Microbenchmarks + correctness tests.
- `benchmarks/` - every measurement script (M1 baseline, graph decode,
  torch.compile attempts) and raw results (`benchmarks/results/*.json`).
- `tools/` - post-training factorization pipeline (`posttrain.py`: streaming
  weight load, SVD, optional ITQ, residual pair, standard packed checkpoint
  output) and the wikitext2 ppl evaluator (`eval_ppl.py`).
- `notes/` - per-milestone findings, including all debugging forensics.
- `PLAN.md` - the milestone ladder and progress log.

## Negative result: post-training factorization without QAT (M3)

Question: can anyone compress a model sub-1-bit without Samsung's QAT
training (which needs the uncompressed teacher and multi-GPU resources)?
If yes, anyone could compress any model on one machine.

Answer: **no, not with SVD-based init** - with an important nuance discovered
along the way. Measured on Qwen3-0.6B/8B (wikitext2, identical 2048-token
chunk protocol):

| config | ppl |
|---|---|
| bf16 baseline (0.6B) | 19.6 (20.28 on 20-chunk protocol) |
| official QAT 8B @ 0.55 bpw | 22.9 |
| our SVD post-training @ 0.55 bpw (± ITQ) | ~5e7 |
| **+ activation-space scale fit, 1 bpw, one layer** | **35.0 (functioning!)** |

The revival: refitting ONLY the scale scalars on real captured activations
(minimizing ||Wx - Rx||, the activation metric, instead of ||W - R||) takes
a 1 bpw factorization from destroyed to functioning. The signs were never
the sole problem - the scales were. Best teacher-free single-layer damage at
1 bpw: +59% ppl (ridge-regularized alternating fit of per-row and per-latent
scales); 28-layer compaction needs ~+2%/layer, so the gap is ~30x and is
dominated by sign geometry that only end-to-end training fixes. Full
ablation and the pre-written kill criteria in `notes/M3_findings.md` and
`PLAN-M4.md`.

Diagnostics that matter (full detail in `notes/M3_findings.md`):
- Per-layer weight reconstruction error is a **bad proxy**: official QAT
  factors measure 0.79 rel err (cos 0.61) and work; SVD init measures 0.92
  (cos 0.39) and is catastrophically broken. The quality cliff lives between
  cos 0.39 and 0.61.
- SVD-derived signs are nearly **orthogonal to the weight matrix**
  (|approx|_F = 0.38 |W|_F; norm-matching rescaling makes the error WORSE -
  the signature of an uncorrelated fit). The QAT training itself shapes the
  sign geometry and scales; the SVD init does not.
- Calibrations tried and dead: joint-LSQ per-latent scales (no change),
  global norm rescale (worse), undamped ICM sign flipping (one marginal gain
  then diverges). ITQ (LittleBit-2's init rotation) changed nothing.
- The metrics lesson: weight-space objectives (||W - R||) optimize the wrong
  thing; the failing metric is ||Wx - Rx|| under real activations.
  Untried follow-ups: activation-aware per-row scale fitting, ASVD-style
  input whitening, damped sign search under the activation objective.

## Why the speedups work (the transferable lessons)

1. **WDDM launch overhead is ~30 us/launch** - a 1000-launch decode step pays
   ~30 ms/tok in overhead before any math. Whole-decode-step CUDA graphs
   (with static buffers, fills OUTSIDE the graph) remove it.
2. **LittleBit's runtime memory is NOT the disk size**: factors unpack to
   bf16 in VRAM (~eff_bit x fp16 size). Storing factors int8 and casting
   per-forward is a net LOSS (~10 GB cast traffic/tok); int8 pays only inside
   a kernel that dequantizes per-tile.
3. **Windows never OOMs**: WDDM silently spills to system RAM. Compare
   `torch.cuda.memory_allocated()` against physical VRAM.
4. Eager profilers inflate small-op costs ~5x; only graph-mode numbers are
   meaningful for launch-bound models.
5. triton-windows 3.x pitfalls: use-after-free of argument temporaries,
   cache poisoning producing silently-wrong numerics, unmasked tail chunks
   reading adjacent memory. All documented with repro cases in `notes/`.

`torch.compile` was tried and is a dead end on this stack (whole-model and
per-layer both 16-25x SLOWER than eager: graph breaks at custom Triton
kernels + dynamic-cache recompile guards + WDDM launch tax).

## Reproducing

```bash
# venv (Windows): torch 2.6.0+cu124, transformers 4.51.3, triton-windows 3.2.x
uv pip install --python <venv> --extra-index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 numpy==2.2.4 transformers==4.51.3 datasets==3.4.1 lm-eval==0.4.8 \
  accelerate bitsandbytes sentencepiece tqdm pytorch-optimizer GPUtil mpi4py \
  tensorboardX "triton-windows==3.2.0.post11"

# baseline
python benchmarks/m1_baseline.py models/littlebit-qwen3-8b-0.55bpw naive
# full optimized pipeline (eager + graph decode, token-verified)
python benchmarks/m25b_graph.py models/littlebit-qwen3-8b-0.55bpw --new-tokens 64
# kernels microbench + correctness
python kernels/bench_factored.py
python kernels/test_gqa_decode.py
```

Checkpoint: `huggingface.co/LittleBitLLM/littlebit-qwen3-8b-0.55bpw`
(license CC BY-NC 4.0, Samsung). Paths in scripts assume the LittleBit repo
cloned at `C:/Users/micro/Code/Projects/littlebit` - adjust `REPO` in the
scripts. On Windows, inductor/compile experiments need MSVC; the Triton
kernels here run from plain git-bash (no vcvars needed except torch.compile).

## Licence

Code in this repo: MIT. LittleBit itself: CC BY-NC 4.0 (Samsung) - compressed
checkpoints inherit that restriction.
