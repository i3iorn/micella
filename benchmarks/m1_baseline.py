"""M1: stock LittleBit loader on the RTX 3050 6GB - VRAM breakdown + decode tok/s.

Variants:
  naive  - stock load, factors bf16 on GPU (expect OOM at 0.55bpw on 6GB)
  int8   - post-load U/V factors converted to int8 (halves factor memory)

Usage:
  python m1_baseline.py <model_dir> <variant> [--new-tokens N]
"""
import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO = Path("C:/Users/micro/Code/Projects/littlebit")
sys.path.insert(0, str(REPO))

# Warm transformers' lazy model-class import BEFORE the repo enters its
# torch.device("meta") context; importing inside it fails with
# "Tensor.item() cannot be called on meta tensors" (transformers 4.51 quirk).
import transformers.models.qwen3.modeling_qwen3  # noqa: E402, F401

from quantization.hub import LittleBitModel  # noqa: E402
from quantization.modules import LittleBitLinear  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def vram_snapshot():
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 3),
    }


def to_int8_factors(model):
    """Store U/V (and residual) factors as int8; forward casts on the fly."""
    n_moved, bytes_saved = 0, 0
    for mod in model.modules():
        if not isinstance(mod, LittleBitLinear):
            continue
        for name in ("U", "V", "U_R", "V_R"):
            p = getattr(mod, name, None)
            if p is not None and p.dtype == torch.bfloat16:
                bytes_saved += p.numel()  # 2 bytes -> 1 byte
                p.data = p.data.to(torch.int8)
                n_moved += 1
    return n_moved, bytes_saved


def load_model(model_dir, variant):
    if variant != "naive":
        model = LittleBitModel.from_pretrained(
            str(model_dir), torch_dtype=torch.bfloat16, device="cpu")
        moved, saved = to_int8_factors(model)
        print(f"[int8] moved {moved} factor tensors, saved {saved / 2**30:.2f} GiB")
        model.to("cuda")
        return model
    return LittleBitModel.from_pretrained(str(model_dir), **kwargs)


@torch.no_grad()
def decode_bench(model, tok, prompt, new_tokens):
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(ids, max_new_tokens=new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    return {"decode_tok_s": round(new_tokens / dt, 2), "gen_seconds": round(dt, 2),
            "sample_text": text[:200]}


@torch.no_grad()
def prefill_bench(model, tok, prompt):
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model(ids)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {"prompt_tokens": ids.shape[1], "prefill_tok_s": round(ids.shape[1] / dt, 2)}


def gpu_proc_mib():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=used_memory",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True)
    vals = [int(v) for v in out.stdout.split() if v.strip().isdigit()]
    return max(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("variant", choices=["naive", "int8"])
    ap.add_argument("--new-tokens", type=int, default=128)
    args = ap.parse_args()

    result = {"variant": args.variant, "model": str(args.model_dir)}
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    try:
        t0 = time.perf_counter()
        model = load_model(args.model_dir, args.variant)
        result["load_seconds"] = round(time.perf_counter() - t0, 1)
        result["vram_after_load"] = vram_snapshot()
        result["gpu_process_mib"] = gpu_proc_mib()
        model.eval()
        result["prefill"] = prefill_bench(model, tok, "The theory of relativity states that")
        result["decode"] = decode_bench(model, tok, "Write a short paragraph about the sea:", args.new_tokens)
        result["vram_after_decode"] = vram_snapshot()
        result["status"] = "ok"
    except torch.cuda.OutOfMemoryError as e:
        result["status"] = "OOM"
        result["error"] = str(e)[:300]
        torch.cuda.empty_cache()
    print(json.dumps(result, indent=2))
    out_path = Path(__file__).parent / "results" / f"m1_{args.variant}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
