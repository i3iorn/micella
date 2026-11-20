# M1 findings — LittleBit Qwen3-8B @ 0.55bpw on RTX 3050 6GB (WDDM, shared-memory spillover)

Checkpoint: LittleBitLLM/littlebit-qwen3-8b-0.55bpw (2.97 GB packed on disk).
Machine: RTX 3050 6GB laptop + 16 GB RAM + Ryzen 7840HS. Greedy decode, batch 1, bf16.

## Numbers (benchmarks/results/m1_{naive,int8}.json)

| variant | VRAM allocated | decode tok/s | prefill (7 tok) |
|---|---|---|---|
| naive (bf16 factors) | 9.46 GB | **1.70** | 2.99 tok/s |
| int8 factors (post-load cast) | 5.86 GB | 1.06 | 2.59 tok/s |

Load time ~70 s both (CPU unpack of packed factors dominates).

## Interpretation

1. **No OOM ever fires on Windows/WDDM.** torch happily allocates 9.46 GB on a
   6 GB card: the excess silently spills to system RAM via shared GPU memory.
   Never rely on OOM to detect non-fit; compare torch.cuda.memory_allocated()
   against the card's physical 6 GB.
2. **Both paths are per-token factor-bandwidth bound, not compute bound.**
   - naive: 9.46 GB resident, ~7.7 GB factors spill to RAM; every token re-reads
     spilled factors over PCIe (~13 GB/s) -> 1.7 tok/s.
   - int8: factors fit in VRAM, but `_compute_forward` casts U/V int8->bf16 on
     EVERY forward (`V.to(x.dtype)`), i.e. ~10 GB of cast traffic per token ->
     1.06 tok/s. Post-load int8 conversion without a cast-aware kernel is a
     net loss.
3. Token quality is visibly degraded at 0.55 bpw (repetitive loops on greedy
   decoding with a short prompt). Nothing alarming for a bandwidth benchmark,
   but worth a proper ppl eval before trusting quality at low bpw (M3 input).

## Implication for M2 (fused kernel)

The kernel must read factors ONCE per token, in their stored precision, and
dequantize inside the tile loop (Triton: load int8 block, convert in registers,
two chained GEMMs + the v1*u2/v1*u1 scale products in one launch). Expected
ceiling on this card: factors ~3.85e9 elements int8 = 3.85 GB -> ~180 GB/s VRAM
bandwidth / 3.85 GB ~ 45 tok/s upper bound for THIS checkpoint (0.55 bpw).
At 0.1 bpw the factor set is ~7x smaller -> bandwidth stops being the limit.

## Setup gotchas (fixed in code)
- Import transformers.models.qwen3.modeling_qwen3 BEFORE the repo's
  torch.device("meta") load path; otherwise import fails with
  "Tensor.item() cannot be called on meta tensors" (transformers 4.51).
- LittleBitModel.from_pretrained: pass device exactly once (it forwards kwargs
  to snapshot_download).
