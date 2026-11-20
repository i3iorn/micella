"""Probe the decode-path attention internals: mask shape, cache layout,
attention function registry - inputs for the M2.8 kernel design."""
import sys
from pathlib import Path

REPO = Path("C:/Users/micro/Code/Projects/littlebit")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))
sys.path.insert(0, str(Path(__file__).parent))

import torch
import transformers.models.qwen3.modeling_qwen3  # noqa: F401,E402
import transformers.modeling_utils as mu  # noqa: E402
from transformers import AutoTokenizer, StaticCache  # noqa: E402

from m25b_graph import load  # noqa: E402

print("ALL_ATTENTION_FUNCTIONS type:", type(mu.ALL_ATTENTION_FUNCTIONS).__name__,
      "| keys:", list(mu.ALL_ATTENTION_FUNCTIONS.keys()))

tok = AutoTokenizer.from_pretrained("models/littlebit-qwen3-8b-0.55bpw")
model = load("models/littlebit-qwen3-8b-0.55bpw")
ids = tok("Hello world", return_tensors="pt").input_ids.cuda()
n = ids.shape[1]
cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=64,
                    device="cuda", dtype=torch.bfloat16)

caught = {}


def spy(module, query, key, value, attention_mask=None, scaling=None, dropout=0.0, **kw):
    caught["q_shape"] = tuple(query.shape)
    caught["k_shape"] = tuple(key.shape)
    caught["k_strides"] = tuple(key.stride())
    caught["mask"] = None if attention_mask is None else (
        tuple(attention_mask.shape), str(attention_mask.dtype),
        attention_mask[0, 0, 0, :min(12, attention_mask.shape[-1])].tolist())
    caught["scaling"] = scaling
    from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward
    return eager_attention_forward(module, query, key, value, attention_mask,
                                   scaling=scaling, dropout=dropout, **kw)


mu.ALL_ATTENTION_FUNCTIONS["sdpa"] = spy
with torch.no_grad():
    model(ids, cache_position=torch.arange(n, device="cuda"),
          past_key_values=cache, use_cache=True)
    pos = torch.tensor([n], device="cuda")
    nxt = torch.zeros(1, 1, dtype=torch.long, device="cuda")
    model(nxt, cache_position=pos, past_key_values=cache, use_cache=True)
for k, v in caught.items():
    print(f"{k}: {v}")
print("cache K layer0:", tuple(cache.key_cache[0].shape),
      "strides:", tuple(cache.key_cache[0].stride()))
