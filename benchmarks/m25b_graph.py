"""M2.5b: CUDA-graph-captured greedy decode step on the fast-patched model.

Design: StaticCache + static token/position buffers; the decode step (all
36 layers, attention, norms, lm_head) captured in one CUDA graph; per replay:
argmax the captured logits buffer, fill buffers, replay.
Correctness gate: generated text must match the eager fast run exactly.
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
from quantization.hub import LittleBitModel  # noqa: E402
from transformers import AutoTokenizer, StaticCache  # noqa: E402

from fast_model import patch_attention, patch_model_fast  # noqa: E402
from m1_baseline import to_int8_factors  # noqa: E402


def load(model_dir):
    from fast_model import patch_decode_overheads
    model = LittleBitModel.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16,
                                           device="cpu")
    to_int8_factors(model)
    patch_model_fast(model)
    model.to("cuda")
    patch_decode_overheads(model, model.config.max_position_embeddings)
    patch_attention(model)
    model.eval()
    model.config.use_cache = True
    return model


@torch.no_grad()
def run(model, tok, prompt, new_tokens, capture=True, cache_len=None):
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    cfg = model.config
    n_prompt = ids.shape[1]
    cache = StaticCache(config=cfg, max_batch_size=1,
                        max_cache_len=cache_len or (new_tokens + n_prompt + 8),
                        device="cuda", dtype=torch.bfloat16)
    tok_buf = torch.zeros(1, 1, dtype=torch.long, device="cuda")
    pos_buf = torch.zeros(1, dtype=torch.long, device="cuda")

    def forward_only():
        # reads tok_buf/pos_buf at replay time; NO fills inside (a captured
        # fill_ would reset the buffers to capture-time values every replay)
        return model(tok_buf, cache_position=pos_buf,
                     past_key_values=cache, use_cache=True)

    # prefill eagerly (fills KV cache 0..n-1); first predicted token comes
    # from the prefill's last-position logits - NO extra call at pos 0 (that
    # overwrote prompt KV slot 0 and made every earlier "reference" decode
    # run at positions 0..k over the prompt's cache: degenerate by design).
    cache_position = torch.arange(n_prompt, device="cuda")
    out = model(ids, cache_position=cache_position,
                past_key_values=cache, use_cache=True)
    tokens = [int(out.logits[:, -1].argmax())]
    tok_buf.fill_(tokens[0])
    pos_buf.fill_(n_prompt)

    if capture:
        for _ in range(3):  # untimed warm steps (also warm cache slots n..n+2)
            out = forward_only()
            tokens.append(int(out.logits[:, -1].argmax()))
            tok_buf.fill_(tokens[-1])
            pos_buf.add_(1)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = forward_only()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(new_tokens):
            graph.replay()
            tokens.append(int(out.logits[:, -1].argmax()))
            tok_buf.fill_(tokens[-1])
            pos_buf.add_(1)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        n_timed_start = 4  # warm produced tokens[1..3]; timed starts at idx 4
    else:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(new_tokens):
            out = forward_only()
            tokens.append(int(out.logits[:, -1].argmax()))
            tok_buf.fill_(tokens[-1])
            pos_buf.add_(1)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        n_timed_start = 1
    return tokens[n_timed_start:], dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--new-tokens", type=int, default=64)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load(args.model_dir)
    print(f"vram after load: {torch.cuda.memory_allocated() / 2**30:.2f} GB")

    prompt = "Write a short paragraph about the sea:"
    n_prompt = len(tok(prompt).input_ids)
    cache_len = args.new_tokens + n_prompt + 8  # SAME cache size for both runs
    # eager reference first (correctness gate; positions n..n+15)
    toks_e, dt_e = run(model, tok, prompt, 16, capture=False, cache_len=cache_len)
    text_e = tok.decode(toks_e)
    print(f"eager : {16 / dt_e:.2f} tok/s | {toks_e[:8]}")

    toks_g, dt_g = run(model, tok, prompt, args.new_tokens, capture=True,
                       cache_len=cache_len)
    text_g = tok.decode(toks_g)
    # graph tokens start 3 warm steps later; greedy continuation must align
    ref = toks_e[3:3 + len(toks_g)]
    mism = [i for i, (a, b) in enumerate(zip(ref, toks_g)) if a != b]
    match = not mism
    if mism:
        print(f"first divergence at token {mism[0]}: eager {ref[mism[0]]} "
              f"vs graph {toks_g[mism[0]]} (of {len(ref)} compared)")
    res = {"graph_decode_tok_s": round(args.new_tokens / dt_g, 2),
           "graph_per_token_ms": round(dt_g / args.new_tokens * 1000, 2),
           "matches_eager": bool(match),
           "n_compared": len(ref), "n_mismatch": len(mism),
           "eager_first16": text_e[:60], "graph_text": text_g[:150]}
    print(json.dumps(res, indent=2))
    out = Path(__file__).parent / "results" / "m25b_graph.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
