"""M2.6b: compile per-decoder-layer instead of the whole model.

Full-model compile recompiles per step (dynamic cache / cache_position guards)
-> 0.2 tok/s. Compiling each Qwen3DecoderLayer keeps cache updates eager and
reuses one compiled graph across 36 identical layers.
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
sys.path.insert(0, str(Path(__file__).parent))

import transformers.models.qwen3.modeling_qwen3  # noqa: E402,F401
from transformers import AutoTokenizer, StaticCache  # noqa: E402

from m25b_graph import load  # noqa: E402

PROMPT = "Write a short paragraph about the sea:"


@torch.no_grad()
def decode(model, tok, new_tokens):
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    n = ids.shape[1]
    cache = StaticCache(config=model.config, max_batch_size=1,
                        max_cache_len=new_tokens + n + 8,
                        device="cuda", dtype=torch.bfloat16)
    cache_position = torch.arange(n, device="cuda")
    model(ids, cache_position=cache_position, past_key_values=cache, use_cache=True)
    nxt = torch.zeros(1, 1, dtype=torch.long, device="cuda")
    pos = torch.tensor([n], device="cuda")
    out = model(nxt, cache_position=pos, past_key_values=cache, use_cache=True)
    tok_id = int(out.logits[:, -1].argmax())
    tokens = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(new_tokens):
        nxt.fill_(tok_id)
        pos.fill_(n + i)
        out = model(nxt, cache_position=pos, past_key_values=cache, use_cache=True)
        tok_id = int(out.logits[:, -1].argmax())
        tokens.append(tok_id)
    torch.cuda.synchronize()
    return tokens, (time.perf_counter() - t0) / new_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-tokens", type=int, default=48)
    args = ap.parse_args()
    torch._dynamo.config.cache_size_limit = 128

    tok = AutoTokenizer.from_pretrained("models/littlebit-qwen3-8b-0.55bpw")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load("models/littlebit-qwen3-8b-0.55bpw")
    # LittleBitModel wraps Qwen3ForCausalLM wraps Qwen3Model - find layers by
    # type instead of attribute paths
    layers = [m for m in model.modules()
              if type(m).__name__ == "Qwen3DecoderLayer"]
    assert len(layers) == model.config.num_hidden_layers, len(layers)

    toks_e, dt_e = decode(model, tok, args.new_tokens)
    print(f"fast-eager : {1 / dt_e:6.2f} tok/s | {toks_e[:6]}")

    t0 = time.perf_counter()
    for layer in layers:
        layer.forward = torch.compile(layer.forward, dynamic=False)
    toks_c, dt_c = decode(model, tok, args.new_tokens)
    print(f"layer-compiled after {time.perf_counter() - t0:.0f}s compile: "
          f"{1 / dt_c:6.2f} tok/s | {toks_c[:6]}")
    print(f"tokens match eager: {toks_c == toks_e} | speedup {dt_e / dt_c:.2f}x")

    res = {"eager_tok_s": round(1 / dt_e, 2), "compiled_tok_s": round(1 / dt_c, 2),
           "speedup": round(dt_e / dt_c, 2),
           "tokens_match": toks_c == toks_e,
           "compile_seconds": round(time.perf_counter() - t0, 0)}
    out = Path(__file__).parent / "results" / "m26b_layers_compile.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
