"""Isolate m25b's degenerate text: exact prompt/cache-size, our attention
kernel returned as model compute, health checks on prefill + every step."""
import sys
from pathlib import Path

REPO = Path("C:/Users/micro/Code/Projects/littlebit")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import transformers.models.qwen3.modeling_qwen3  # noqa: F401,E402
from transformers import AutoTokenizer, StaticCache  # noqa: E402

from m25b_graph import load  # noqa: E402

model = load("models/littlebit-qwen3-8b-0.55bpw")
tok = AutoTokenizer.from_pretrained("models/littlebit-qwen3-8b-0.55bpw")
ids = tok("Write a short paragraph about the sea:", return_tensors="pt").input_ids.cuda()
n = ids.shape[1]
print("prompt tokens:", n)
cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=110,
                    device="cuda", dtype=torch.bfloat16)
tok_buf = torch.zeros(1, 1, dtype=torch.long, device="cuda")
pos_buf = torch.zeros(1, dtype=torch.long, device="cuda")


def health(tag, lg):
    print(f"{tag}: |logits|={lg.abs().mean().item():.4f} "
          f"nan={bool(torch.isnan(lg).any())} argmax={int(lg.argmax())}")


with torch.no_grad():
    out = model(ids, cache_position=torch.arange(n, device="cuda"),
                past_key_values=cache, use_cache=True)
    health("prefill last", out.logits[:, -1])
    tok_buf.fill_(int(out.logits[:, -1].argmax()))
    pos_buf.fill_(n)
    for i in range(6):
        o = model(tok_buf, cache_position=pos_buf, past_key_values=cache,
                  use_cache=True)
        health(f"step {i} (pos {n + i})", o.logits[:, -1])
        tok_buf.fill_(int(o.logits[:, -1].argmax()))
        pos_buf.add_(1)
