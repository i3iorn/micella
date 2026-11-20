"""Chunked wikitext2 perplexity for LittleBit checkpoints (and fp baselines).

Same protocol for every config: fixed seqlen 2048, first N chunks of the
wikitext2 test set, teacher-forced NLL -> PPL. int8 factors after load (stock
forward casts on the fly - correctness over speed).
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path("C:/Users/micro/Code/Projects/littlebit")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))

import transformers.models.qwen3.modeling_qwen3  # noqa: F401,E402
from quantization.hub import LittleBitModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from utils.datautils import get_eval_loaders  # noqa: E402

from m1_baseline import to_int8_factors  # noqa: E402


@torch.no_grad()
def ppl_eval(model, tokenizer, seqlen=2048, n_chunks=40):
    testenc = get_eval_loaders("wikitext2", tokenizer)
    testenc = testenc.input_ids
    nsamples = min(n_chunks, testenc.numel() // seqlen)
    nlls = []
    for i in range(nsamples):
        batch = testenc[:, (i * seqlen):((i + 1) * seqlen)].cuda()
        out = model(batch, use_cache=False)
        logits = out.logits
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = batch[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss.float() * (seqlen - 1))
        if (i + 1) % 10 == 0:
            print(f"  chunk {i + 1}/{nsamples} running ppl "
                  f"{torch.exp(torch.stack(nlls).sum() / ((i + 1) * (seqlen - 1))):.3f}")
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * (seqlen - 1)))
    return ppl.item(), nsamples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id", help="LittleBit dir OR HF repo for fp baseline")
    ap.add_argument("--fp", action="store_true", help="evaluate as plain fp model")
    ap.add_argument("--chunks", type=int, default=40)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    t0 = time.perf_counter()
    if args.fp:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, torch_dtype=torch.bfloat16, device_map="cuda",
            low_cpu_mem_usage=True)
    else:
        model = LittleBitModel.from_pretrained(args.model_id,
                                               torch_dtype=torch.bfloat16,
                                               device="cpu")
        to_int8_factors(model)
        model.to("cuda")
    model.eval()
    print(f"loaded in {time.perf_counter() - t0:.0f}s | "
          f"vram {torch.cuda.memory_allocated() / 2**30:.2f} GB")

    ppl, n = ppl_eval(model, tok, n_chunks=args.chunks)
    res = {"model": args.model_id, "fp": args.fp, "chunks": n,
           "wikitext2_ppl": round(ppl, 4), "tag": args.tag}
    print(json.dumps(res, indent=2))
    out = Path(__file__).parent / "results" / "m3_ppl.jsonl"
    out.parent.mkdir(exist_ok=True)
    with open(out, "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
