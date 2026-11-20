"""Compare factor statistics: official QAT checkpoint vs our post-trained one."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, "C:/Users/micro/Code/Projects/littlebit")
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))

import transformers.models.qwen3.modeling_qwen3  # noqa: F401,E402
from quantization.hub import LittleBitModel  # noqa: E402
from quantization.modules import LittleBitLinear  # noqa: E402
from m1_baseline import to_int8_factors  # noqa: E402


def probe(path, tag):
    m = LittleBitModel.from_pretrained(path, torch_dtype=torch.bfloat16, device="cpu")
    to_int8_factors(m)
    for n, mod in m.model.named_modules():
        if isinstance(mod, LittleBitLinear) and "layers.0.self_attn.q_proj" in n:
            print(f"[{tag}] {n}")
            print(f"  U uniq={mod.U.data.unique().tolist()[:4]} "
                  f"shape={tuple(mod.U.shape)} |U|.mean={mod.U.data.float().abs().mean():.3f}")
            for s in ("u1", "u2", "v1", "v2"):
                t = getattr(mod, s).data.float()
                print(f"  {s}: mean={t.mean():.6f} std={t.std():.6f} "
                      f"absmean={t.abs().mean():.6f}")
            Uq, Vq = mod.U.data.float(), mod.V.data.float()
            u1, u2 = mod.u1.data.float(), mod.u2.data.float()
            v1, v2 = mod.v1.data.float(), mod.v2.data.float()
            W_impl = (Uq * (u1.t() @ u2)) @ (Vq * (v1.t() @ v2))
            print(f"  |W_impl|_F={W_impl.norm():.3f}  split={mod.split_dim}")
            break
    del m


probe("models/littlebit-qwen3-8b-0.55bpw", "official-8B")
probe("models/posttrain-0.6b-055", "posttrain-0.6B")
