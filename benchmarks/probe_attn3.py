"""Every-call verification: our attention vs sdpa on all decode steps/layers,
plus NaN scan of logits, to localize the m25b text regression."""
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
from transformers import AutoTokenizer, StaticCache  # noqa: E402

import kernels.fast_model as fm  # noqa: E402
from m1_baseline import to_int8_factors  # noqa: E402
from fast_model import patch_decode_overheads, patch_model_fast  # noqa: E402
from quantization.hub import LittleBitModel  # noqa: E402

bad = []
calls = [0]


def spy(module, query, key, value, attention_mask=None, scaling=None, dropout=0.0, **kw):
    idx = calls[0]
    calls[0] += 1
    if query.shape[2] == 1:
        ours, _ = fm.micella_attention(module, query, key, value,
                                       attention_mask=attention_mask,
                                       scaling=scaling, dropout=dropout, **kw)
        ref = F.scaled_dot_product_attention(query, key, value,
                                             attn_mask=attention_mask,
                                             enable_gqa=True)
        err = (ours.float() - ref.float()).abs().max().item()
        nan = bool(torch.isnan(ours).any())
        if len(bad) < 5 and (err > 0.02 or nan):
            k_finite = bool(torch.isfinite(key.float()).all())
            v_finite = bool(torch.isfinite(value.float()).all())
            bad.append((idx, err, nan, "k_finite", k_finite, "v_finite", v_finite,
                        float(ref.abs().mean()), float(ours.abs().mean())))
        from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward
        eager_out = eager_attention_forward(module, query, key, value, attention_mask,
                                            scaling=scaling, dropout=dropout, **kw)
        # m25b-equivalent: return OUR output as the model's compute
        ours, _ = fm.micella_attention(module, query, key, value,
                                       attention_mask=attention_mask,
                                       scaling=scaling, dropout=dropout, **kw)
        return ours, None
    from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward
    return eager_attention_forward(module, query, key, value, attention_mask,
                                   scaling=scaling, dropout=dropout, **kw)


mu.ALL_ATTENTION_FUNCTIONS["sdpa"] = spy

model = LittleBitModel.from_pretrained("models/littlebit-qwen3-8b-0.55bpw",
                                       torch_dtype=torch.bfloat16, device="cpu")
to_int8_factors(model)
patch_model_fast(model)
model.to("cuda")
patch_decode_overheads(model, model.config.max_position_embeddings)
model.eval()

tok = AutoTokenizer.from_pretrained("models/littlebit-qwen3-8b-0.55bpw")
ids = tok("Write a short paragraph about the sea:", return_tensors="pt").input_ids.cuda()
n = ids.shape[1]
cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=256,
                    device="cuda", dtype=torch.bfloat16)
with torch.no_grad():
    model(ids, cache_position=torch.arange(n, device="cuda"),
          past_key_values=cache, use_cache=True)
    nxt = torch.zeros(1, 1, dtype=torch.long, device="cuda")
    pos = torch.tensor([n], device="cuda")
    for i in range(6):
        out = model(nxt, cache_position=pos, past_key_values=cache, use_cache=True)
        lg = out.logits[:, -1]
        print(f"step {i}: |logits|={lg.abs().mean().item():.4f} "
              f"nan={bool(torch.isnan(lg).any())} argmax={int(lg.argmax())}")
        nxt.fill_(int(lg.argmax()))
        pos.add_(1)
print("calls:", calls[0], "| first bad:", bad[:3] if bad else "NONE")
