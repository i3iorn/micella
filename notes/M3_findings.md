# M3 — post-training factorization without QAT (in progress)

Question: can teacher-free factorization replace Samsung's QAT pipeline?

## Tools built (working)
- tools/posttrain.py: streams weights off disk -> SVD (GPU, fp32) -> optional
  ITQ (LittleBit-2 joint rotation) -> sign-quantized int8 factors + rank-1
  scalars -> residual pair -> standard packed LittleBit checkpoint (0.6B:
  342 MB vs 1.2 GB bf16). Loads through the stock LittleBitModel loader.
- tools/eval_ppl.py: uniform wikitext2 protocol (2048-tok chunks, first N).

## Results (wikitext2 ppl, 40-chunk protocol)
| config | ppl |
|---|---|
| Qwen3-0.6B bf16 | 19.56 |
| 0.6B post-train SVD @0.55bpw (residual, no ITQ) | 5.7e7 |
| 0.6B post-train SVD+ITQ @0.55bpw | 5.0e7 |
| 8B official QAT @0.55bpw (20 chunks) | 22.86 |

## Key diagnostics
1. Per-layer weight reconstruction rel err is a BAD proxy: official QAT
   factors measure 0.79 (main+residual) vs our SVD init 0.92 — similar, yet
   functioning vs catastrophic. What matters is cos similarity of the
   reconstruction: official ~0.61, ours ~0.39. The cliff between 0.39 and
   0.61 cos-sim is the difference between working and destroyed.
2. The binary reconstruction is nearly UNCORRELATED with W: |approx|_F =
   0.38|W|_F and global rescaling makes rel err WORSE (1.11) - classic sign of
   an (almost) orthogonal fit. SVD signs alone carry little information.
3. Cheap calibrations tried and dead: joint-LSQ per-latent scales (no change),
   global norm rescale (worse), ICM sign flipping (0.92 -> 0.89 first sweep,
   then diverges - needs damping + a better objective).
4. Bugs found: int8 truncation instead of sign quantization (fixed - was the
   first catastrophic-ppl cause, STILL catastrophic after fix), missing
   tokenizer files in save (fixed).

## Why QAT wins (working theory)
QAT trains (a) the signs to align with activations end-to-end, (b) the scale
scalars to compensate the binary loss. Teacher-free weight-space objectives
(||W - R||) optimize the wrong metric: what matters is f(x) matching, i.e.
||Wx - Rx|| under real activations.

## Next experiments (ranked)
1. **Activation-aware scale calibration** (highest odds, cheap): capture each
   Linear's input x on ~1k calibration tokens from the ORIGINAL model; per
   output row fit u1_k = <y_ref_k, yhat_k>/<yhat_k, yhat_k> (and optionally
   refit c_j) minimizing ||Wx - Rx||. One calibration pass + per-layer 1D/LSQ
   solves. Directly targets the failing metric.
2. Activation-aware SVD (ASVD-style): whiten by input activation RMS before
   decomposing (changes WHICH directions survive binarization).
3. Damped ICM / local sign search with the activation-space objective.
4. Accept 0.1-0.3 bpw may simply need distillation (KD from the fp model on
   CPU is possible for small models - the KDTrainer loss exists in the repo).

## Verdict so far
Naive post-training FAILS (negative result, now documented). The pipeline is
in place; experiment 1 is the immediate next step before declaring M3 dead.
