"""Diff our M=1 GQA kernel vs sdpa on REAL in-model attention inputs."""
import sys
from pathlib import Path

REPO = Path("C:/Users/micro/Code/Projects/littlebit")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import transformers.models.qwen3.modeling_qwen3  # noqa: F401,E402
import transformers.modeling_utils as mu  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import kernels.fast_model as fm  # noqa: E402
from m25b_graph import load  # noqa: E402

stats = []


def spy(module, query, key, value, attention_mask=None, scaling=None, dropout=0.0, **kw):
    if query.shape[2] == 1 and len(stats) < 3:
        ours, _ = fm.micella_attention(module, query, key, value,
                                       attention_mask=attention_mask,
                                       scaling=scaling, dropout=dropout, **kw)
        ref = F.scaled_dot_product_attention(query, key, value,
                                             attn_mask=attention_mask,
                                             enable_gqa=True)
        err = (ours.float() - ref.float()).abs().max().item()
        stats.append((tuple(query.shape), tuple(key.shape),
                      None if attention_mask is None else tuple(attention_mask.shape),
                      err,
                      float(ours.float().abs().mean()), float(ref.float().abs().mean()),
                      str(scaling)))
    from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward
    return eager_attention_forward(module, query, key, value, attention_mask,
                                   scaling=scaling, dropout=dropout, **kw)


mu.ALL_ATTENTION_FUNCTIONS["sdpa"] = spy
tok = AutoTokenizer.from_pretrained("models/littlebit-qwen3-8b-0.55bpw")
model = load.__wrapped__("models/littlebit-qwen3-8b-0.55bpw") if hasattr(load, "__wrapped__") else None
if model is None:
    # load without the attention patch (load() registers our kernel); rebuild manually
    from m1_baseline import to_int8_factors
    from fast_model import patch_model_fast
    from quantization.hub import LittleBitModel
    model = LittleBitModel.from_pretrained("models/littlebit-qwen3-8b-0.55bpw",
                                           torch_dtype=torch.bfloat16, device="cpu")
    to_int8_factors(model)
    patch_model_fast(model)
    model.to("cuda")
    from fast_model import patch_decode_overheads
    patch_decode_overheads(model, model.config.max_position_embeddings)
    model.eval()

ids = tok("Hello world", return_tensors="pt").input_ids.cuda()
n = ids.shape[1]
from transformers import StaticCache
cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=64,
                    device="cuda", dtype=torch.bfloat16)
with torch.no_grad():
    model(ids, cache_position=torch.arange(n, device="cuda"),
          past_key_values=cache, use_cache=True)
    nxt = torch.zeros(1, 1, dtype=torch.long, device="cuda")
    pos = torch.tensor([n], device="cuda")
    model(nxt, cache_position=pos, past_key_values=cache, use_cache=True)
for s in stats:
    print(f"q{s[1] and s[0]} k{s[1]} mask{s[2]} err={s[3]:.5f} "
          f"|ours|={s[4]:.4f} |ref|={s[5]:.4f} scale={s[6]}")
