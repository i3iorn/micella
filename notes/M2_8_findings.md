# M2.8 — custom M=1 GQA decode-attention kernel

Replaces HF's mem-efficient sdpa on decode steps (q_len==1) via the
ALL_ATTENTION_FUNCTIONS["sdpa"] registry override (config stays "sdpa" so HF's
mask building is untouched; prefill q_len>1 falls back to eager attention math).

## Kernel design
- One Triton program per q head (32); each reads its kv head's FULL StaticCache
  K/V (address-stable -> graph-safe) with an ONLINE softmax over cache_len
  chunks; additive bf16 mask (device-derived from cache_position) excludes
  beyond-L entries. No host-side length reads -> CUDA-graph compatible.
- Output: per-module static buffer, returned head-major.
- Verified vs sdpa on 8 (MAXL, L) shapes incl. adversarial non-multiples.

## Results (Qwen3-8B @0.55bpw, RTX 3050, graph decode)
| stage | tok/s | per token |
|---|---|---|
| M2.7 | 12.97 | 77.1 ms |
| M2.8 | **14.10** | **70.9 ms** |

+8.7%. Smaller than the eager profiler's 22-32ms attention estimate suggested
(same lesson as M2.7: eager profiler times are inflated by per-kernel overhead;
the graph-mode attention was already ~15ms). Cumulative: 1.7 -> 14.1 = 8.3x.

## Bugs found on the way (both real, both instructive)
1. **Unmasked tail chunk in the attention kernel**: MAXL not a multiple of
   BLOCK_L -> OOB reads of adjacent memory (no crash) whose garbage scores
   entered the softmax -> token-0-loop degenerate output. My original test
   sizes (64/256) were all block multiples and HID it. Rule: test every kernel
   with sizes that are NOT multiples of the block size.
2. **Benchmark bootstrap bug (m25b)**: an extra post-prefill forward at
   position 0 overwrote prompt KV slot 0, and the decode loop ran at positions
   0..k OVER the prompt's cache. Every pre-M2.8 "reference" text (including
   the coherent-looking "The sea is...") was generated from a corrupted cache.
   Now: first token from prefill's last-position logits, decode starts at
   position n; graph-vs-eager gate compares position-aligned continuations.
3. Eager-mode decode with the dispatcher is slow (0.73 tok/s) for reasons not
   yet investigated - the graph path is the product; eager is only a gate.

## Updated speed budget (70.9 ms/token)
factored matmuls ~17 | attention ~10-15 (was ~22-30) | lm_head ~7 | soup+KV
+replay ~30. The next meaningful step is no longer obvious single-op work:
remaining wins are deeper fusion (attention into the factored o_proj, KV-cache
write fused into k/v kernels) or moving to M3/M4 where the matmul floor
dominates. The 50 tok/s target remains an M4 (0.1bpw MoE) property.
