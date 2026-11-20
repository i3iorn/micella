"""M2.6: torch.compile on the fast-patched model's decode step.

Measures: fast-eager (baseline) vs compiled, then compiled + manual CUDA
graph on top. Tokens must match eager.
Run through benchmarks/run_msvc.bat (inductor needs MSVC on PATH).
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
def decode(model, tok, new_tokens, compiled=False):
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    n = ids.shape[1]
    cache = StaticCache(config=model.config, max_batch_size=1,
                        max_cache_len=new_tokens + n + 8,
                        device="cuda", dtype=torch.bfloat16)
    fn = compiled_model if compiled else model
    cache_position = torch.arange(n, device="cuda")
    model(ids, cache_position=cache_position, past_key_values=cache, use_cache=True)
    nxt = torch.zeros(1, 1, dtype=torch.long, device="cuda")
    pos = torch.tensor([n], device="cuda")
    model(nxt, cache_position=pos, past_key_values=cache, use_cache=True)
    tok_id = int(model(nxt, cache_position=pos, past_key_values=cache,
                       use_cache=True).logits[:, -1].argmax())
    tokens = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(new_tokens):
        nxt.fill_(tok_id)
        pos.fill_(n + i)
        out = fn(nxt, cache_position=pos, past_key_values=cache, use_cache=True)
        tok_id = int(out.logits[:, -1].argmax())
        tokens.append(tok_id)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return tokens, dt / new_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-tokens", type=int, default=48)
    args = ap.parse_args()
    torch._dynamo.config.cache_size_limit = 64

    tok = AutoTokenizer.from_pretrained("models/littlebit-qwen3-8b-0.55bpw")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load("models/littlebit-qwen3-8b-0.55bpw")
    print(f"vram: {torch.cuda.memory_allocated() / 2**30:.2f} GB")

    toks_e, dt_e = decode(model, tok, args.new_tokens)
    print(f"fast-eager : {1 / dt_e:6.2f} tok/s ({dt_e * 1000:.1f} ms/tok) | {toks_e[:6]}")

    global compiled_model
    compiled_model = torch.compile(model, mode="default", dynamic=False)
    t0 = time.perf_counter()
    toks_c, dt_c = decode(model, tok, args.new_tokens, compiled=True)
    print(f"compile took {time.perf_counter() - t0:.0f}s")
    print(f"compiled   : {1 / dt_c:6.2f} tok/s ({dt_c * 1000:.1f} ms/tok) | {toks_c[:6]}")
    match = toks_c == toks_e
    print(f"tokens match eager: {match}")

    res = {"eager_tok_s": round(1 / dt_e, 2), "compiled_tok_s": round(1 / dt_c, 2),
           "speedup": round(dt_e / dt_c, 2), "tokens_match": bool(match),
           "compile_seconds": round(time.perf_counter() - t0, 0)}
    out = Path(__file__).parent / "results" / "m26_compile.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
