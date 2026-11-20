"""M2.5: end-to-end decode on the real model, fast kernel path vs stock.

Same greedy loop (StaticCache, manual argmax) for both variants; only the
LittleBitLinear forward differs. Variant 'fast' = int8 factors + Triton
kernels; 'stock' = bf16 factors + stock forward (9.5 GB, WDDM-spilled).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path("C:/Users/micro/Code/Projects/littlebit")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))

import transformers.models.qwen3.modeling_qwen3  # noqa: E402,F401
from quantization.hub import LittleBitModel  # noqa: E402
from transformers import AutoTokenizer, StaticCache  # noqa: E402

from fast_model import patch_model_fast  # noqa: E402
from m1_baseline import to_int8_factors  # noqa: E402


def load(model_dir, variant):
    model = LittleBitModel.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16,
                                           device="cpu")
    if variant == "fast":
        to_int8_factors(model)
        n = patch_model_fast(model)
        print(f"[fast] patched {n} LittleBitLinear modules")
    model.to("cuda")
    model.eval()
    model.config.use_cache = True
    return model


@torch.no_grad()
def decode(model, tok, prompt, new_tokens, max_cache_len):
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    cfg = model.model.config if hasattr(model, "model") else model.config
    cache = StaticCache(config=cfg, max_batch_size=1, max_cache_len=max_cache_len,
                        device="cuda", dtype=torch.bfloat16)
    n_prompt = ids.shape[1]
    cache_position = torch.arange(n_prompt, device="cuda")
    model(ids, cache_position=cache_position, past_key_values=cache, use_cache=True)
    nxt = torch.zeros(1, 1, dtype=torch.long, device="cuda")
    cache_pos1 = torch.zeros(1, dtype=torch.long, device="cuda")
    generated = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(new_tokens):
        pos = n_prompt + i
        cache_pos1.fill_(pos)
        out = model(nxt, cache_position=cache_pos1, past_key_values=cache, use_cache=True)
        tok_id = out.logits[:, -1].argmax(-1)
        generated.append(tok_id.item())
        nxt = tok_id.reshape(1, 1)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    text = tok.decode(generated)
    return {"decode_tok_s": round(new_tokens / dt, 2),
            "per_token_ms": round(dt / new_tokens * 1000, 2),
            "sample_text": text[:150]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("variant", choices=["fast", "stock"])
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--prompt", type=str, default="Write a short paragraph about the sea:")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    t0 = time.perf_counter()
    model = load(args.model_dir, args.variant)
    print(f"loaded in {time.perf_counter() - t0:.0f}s | "
          f"vram {torch.cuda.memory_allocated() / 2**30:.2f} GB")
    res = decode(model, tok, args.prompt, args.new_tokens, args.new_tokens + 512)
    res.update(variant=args.variant, model=str(args.model_dir))
    print(json.dumps(res, indent=2))
    out = Path(__file__).parent / "results" / f"m25_{args.variant}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
