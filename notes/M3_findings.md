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

## UPDATE (same day): activation-space scale fitting WORKS - M3 revived

Following the WSP literature (validation-gated, localized edits; weightBench's
data-free-compression failure track), gate 2/3 were rebuilt around the
activation metric. Results on Qwen3-0.6B layer-0 MLP @ 1 bpw:

| method | activation cos | spliced ppl delta |
|---|---|---|
| naive SVD init (M3 original) | 0.56-0.59 | +7e7% (ppl 1.4e8) |
| + ITQ (LittleBit-2 rotation) | 0.70-0.76 | - |
| + activation-space u1/c fit | 0.71-0.79 | **+72% (ppl 35.0)** |
| official QAT 8B @0.55 (control) | 0.62-0.76 | works (ppl 22.9) |

Three decisive facts:
1. **Activation-space scale fitting is the missing science ingredient.** Same
   signs, same structure - refitting ONLY the scales on real activations
   turns a destroyed model (ppl 1.4e8) into a functioning one (ppl 35).
2. The 0.95 activation-cos gate was miscalibrated: the OFFICIAL working QAT
   model measures 0.62-0.76 per-linear (row minima 0.07-0.27). cos is a
   diagnostic; ppl is the gate. (Second gate recalibrated by a control -
   same lesson as gate 1.)
3. A single layer at 1 bpw costs +72% ppl - still ~35x too much for 28-layer
   compaction (need ~+2%/layer). Remaining gap is the sign geometry (cos
   0.71-0.79 vs what QAT achieves end-to-end) and possibly calibration
   breadth (2048 tokens for ~3800 fit params).

## FINAL (same day): ablation results and the verdict

Ridge-stabilized ablation (single-layer ppl delta @ 1 bpw, 0.6B layer-0 MLP):
| config | ppl delta |
|---|---|
| u1/c fit + ridge, 2k calib | **+58.8%** (best) |
| + v2 input-side LSQ fit, 2k | +177% (overfits) |
| + v2 fit, 8k calib | +76.9% |
| whitening | diverges (ill-conditioned) |
| (no ridge, 2k, earlier) | +85.4% |

Findings:
- Ridge is essential (85 -> 59); the v2 fit consistently overfits (drop it);
  whitening needs conditioning work or is a dead end here.
- Best teacher-free single-layer damage at 1 bpw: +59% ppl. 28-layer
  compaction needs ~+2%/layer. Gap ~30x, dominated by sign geometry which
  only end-to-end training (QAT) fixes.

## VERDICT (per PLAN-M4.md gate 8, written before starting)
KILL CRITERION FIRES: 1 bpw factorization fails the quality bar with all
method combinations (SVD/ITQ init, activation-space u1/c/v2 fits, ridge,
whitening) - and the runtime constraint compounds it: 30B params at 1 bpw
needs PACKED storage (3.75 GB) since int8 factors are 30 GB; the packed
kernel only made sense if quality held. 2 bpw would need 7.5 GB packed >
6 GB VRAM. => a 30B MoE factorized on 6 GB VRAM at working quality is out
of reach without QAT/distillation training. Recorded honestly; the
activation-space fitting discovery (catastrophic -> functional) and the
calibration-gate discipline are the reusable positives.

Honest alternatives within the original goal ("big model, fast, 6 GB"):
- smaller MoE (OLMoE-1B-7B: 7B params -> 1 bpw packed = 0.9 GB, trivially
  fits; 50 tok/s likely reachable with the M2 runtime),
- higher-bpw dense LittleBit checkpoints already published (27B @ 1bit
  Haverbex - needs transformers 5.x + packed kernels),
- colibri-style hybrid serving (int4 experts, GPU dense trunk).
