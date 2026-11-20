# M4 plan — 30B MoE toward 50 tok/s on 6 GB VRAM (WSP-informed)

Goal (unchanged): run a 30B+ parameter MoE at ~50 tok/s on the RTX 3050 6GB.

Constraint math (drives every gate below):
- Qwen3-Coder-30B-A3B: ~30B params, ~48 layers x 128 experts top-8, Apache-2.0.
- int8 factors in VRAM need eff_bit <= ~0.12 bpw  -> unproven/deep regime.
- PACKED 1-bit factors need  ~1.0 bpw -> 33 Gbit ~ 4.1 GB -> fits 6 GB.
  => quality target is 1 bpw; runtime target is a packed-factor CUDA kernel.
- Validation principle (from WSP literature): never ship an unvalidated edit.
  Every compression step is gated on measured model-level quality, not weight
  reconstruction error (which M3 proved to be a bad proxy).

Reference points to beat: colibri serves this exact model at int4 experts
(4 bpw, ~19 GB container) on CPU/RAM at 6-10 tok/s. Our bar: all factors in
6 GB VRAM at 1 bpw at >= 50 tok/s, ppl within +10% of colibri's int4 serving.

---

## 1. Validation infrastructure (do first — everything else gates on it)

Goal: a cheap, trustworthy per-layer and model-level quality signal.
- Build a calibration + validation set: ~2k tokens of held-out text (not
  wikitext2 test; use e.g. C4 validation slice), fixed prompts.
- Metrics: (a) per-Linear activation match: mean cos(y_ref, y_hat) over
  captured inputs; (b) blockwise ppl after compressing a prefix of layers;
  (c) full-model ppl on the 40-chunk protocol (tools/eval_ppl.py, already
  built).
- Tool: capture activations once from the bf16 model (hooks), store per-Linear
  inputs (~2k x 2048 bf16 per layer, ~30 GB total on disk - store per-layer
  means/covariances instead: mean vector + Gram matrix of inputs, enough for
  ||Wx - Rx|| fitting and cos checks).

TEST: reproduce colibri's int4 quality story end-to-end on 1 layer of the
30B model (compress its MLP block to int4 with simple per-channel scales,
measure activation cos >= 0.999 and block-ppl delta < 1%).
GATE: infrastructure produces stable, repeatable numbers (same config twice
-> same ppl to 3 decimals). PASS = proceed; FAIL = fix harness first.

## 2. Single-expert factorization at 1 bpw (the decisive experiment)

Goal: factorize ONE expert FFN of layer 0 (3 matrices: gate/up/down,
2048x512-ish at intermediate 512... verify real shapes) to ~1 bpw factors.
- Methods, in order: SVD init; + activation-space scale fit (u1_k per row via
  <y_ref, y_hat>/<y_hat, y_hat> on calibration activations); + joint-LSQ
  per-latent c_j in activation space; + ITQ before rank-1 decomposition.
- Metric: activation cos(y_ref, y_hat) per matrix (NOT weight rel err).

TEST: 1 bpw single-expert factorization reaches activation cos >= 0.95 with
at least one method combination.
GATE: cos >= 0.95 (marginal), >= 0.98 (strong). FAIL (all methods < 0.95) =
1 bpw factorization of expert matrices is not reachable teacher-free; drop to
investigating 2 bpw (still fits if trunk is int4) or conclude. Either way this
gate costs ~1 day and decides the project's direction.

## 3. Whole-layer acceptance: the gate that makes edits safe

Goal: compress ALL 7 linears of decoder layer 0 (attention + shared + 128
experts) and gate on block-level quality.
- Compress layer 0 of the real 30B model, splice into the otherwise-bf16
  model (bf16 trunk fits in 6 GB? layer-0 factors ~70 MB bf16 - yes).
- Metric: full-model ppl delta vs bf16 on the 40-chunk protocol + generation
  sanity (greedy 128 tokens, no token-0 loops).

TEST: ppl delta <= +2% and sane generation.
GATE: PASS = the per-layer validation methodology is sound; repeat for a
mid-layer (24) to check depth-dependence, then proceed. FAIL = diagnose which
matrix type breaks it (attention q/o vs expert FFN) - attention matrices are
the usual suspects (softmax sensitivity); if only attention fails, compress
experts-only and keep attention bf16 (still useful: experts are 25B of 30B).

## 4. All-experts compression: the "30B MoE running" milestone

Goal: every expert (48 layers x 128) at 1 bpw; attention + shared + embed/lm
head stay bf16/int8.
- Storage: experts ~ 25B params x 1 bpw packed = ~3.1 GB packed factors on
  disk + trunk in int8 (~2 GB RAM via colibri's own conversion tooling or our
  own converter).
- Runtime for THIS milestone: hybrid - compressed experts via our int8 Triton
  path (factors unpacked on the fly per layer, only the active 8 experts'
  factors touched per token - at 1 bpw packed, active bytes/token ~ 3 MB ->
  trivial), trunk bf16 on GPU.

TEST: end-to-end generation works; ppl within +10% of colibri int4 serving;
measure tok/s (expect 10-20 tok/s with trunk on GPU + CPU expert decode
overlap - NOT yet the 50 target).
GATE: ppl pass + tok/s >= 10. This is the "30B MoE runs on this machine"
deliverable. FAIL here with passing layer gates = integration/memory-bandwidth
problem, not a compression problem.

## 5. Packed-factor CUDA kernel (the 50 tok/s unlock)

Goal: factors stored PACKED 1-bit in VRAM (4.1 GB total) + fused unpack-GEMV
kernel; M2b's deferred item, now justified by the capacity math.
- CUDA (not Triton - M2b showed Triton loses on bit extraction): 1-bit
  extraction via LOP3/PRMT, or int8 tensor-core dot on packed nibbles
  (sign tricks: bit -> ±1 via xor with 0x80 style); dequant per-tile in
  registers, two chained GEMVs fused per the M2 kernel design.
- Success bar from M2: >= 2x the int8 graph path at equal-or-better bandwidth.

TEST: microbench on the M2 shape set (7 matrices), correctness vs fp32 ref,
then whole-decode-step graph on the 8B @1bpw Haverbex checkpoint as a
real-model validation before touching the 30B.
GATE: >= 100 tok/s implied on 8B @1bpw (vs 57 at 0.55 int8) and bit-exact
correctness. FAIL = stay int8; then the 30B needs 0.12 bpw (gate 2 verdict)
or partial residency.

## 6. Full residency: all factors in 6 GB VRAM

Goal: packed expert factors (3.1 GB) + int8 trunk (~2 GB) + KV cache
(~0.5 GB at 2k ctx with GQA 8:1... verify; Qwen3-Coder is GQA so KV is small)
-> everything resident, no streaming.
- CUDA graph the whole decode step (M2.8 pattern, now with packed kernels).

TEST: tok/s at 2k context >= 50; ppl unchanged vs milestone-4 hybrid.
GATE: PASS = GOAL ACHIEVED (primary success criterion). 40-49 tok/s =
investigate launch/fusion gaps (attention kernel in packed form, KV write
fusion). < 40 = profile; the remaining wall will be attention (M2.8's kernel
reads K/V once per token - at 2k ctx that is ~2.5 MB/layer -> ~90 MB/token
-> bandwidth-bound ~ 2 ms... not the wall; expect the wall to be launch count
or the trunk GEMVs).

## 7. Quality ceiling mapping (the deliverable regardless of tok/s)

Goal: characterize the quality/tok/s/space tradeoff surface:
{1, 2, 4} bpw x {experts-only, all-linear} x {residual on/off} -> ppl table
vs colibri int4 and vs bf16.
- Reuse gates 2-4 machinery; ~9 configs x ~1 h each.

TEST: at least 6 of 9 configs complete with clean measurements.
GATE: this table IS the paper/repo-update artifact. PASS = publish (repo
README + results JSONL); FAIL = document which configs are unstable.

## 8. Honest-kill criteria (written down before starting)

The project STOPS (documents and closes) if any of these hold:
- Gate 2 fails at 1 AND 2 bpw with all method combinations -> factorized MoE
  on 6 GB is out of reach teacher-free; record the negative result and stop.
- Gate 4 passes but gate 5 cannot reach >= 1.5x int8 in CUDA after 2 serious
  kernel iterations -> ship the hybrid (gate 4 state) as final and stop.
- Any gate requires running QAT/training -> explicitly out of scope (that is
  the thing the whole plan exists to avoid).

Sequencing: 1 -> 2 (kill gate) -> 3 -> 4 (first user-visible deliverable)
-> 5 (kill gate #2) -> 6 (goal) -> 7 (publication) with gate 8 checked at
every transition.

Immediate next action: gate 1 (validation infrastructure on the 30B model,
needs Qwen3-Coder-30B-A3B download ~19 GB).
