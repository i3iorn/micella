import sys
from pathlib import Path

sys.path.insert(0, "C:/Users/micro/Code/Projects/littlebit")
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))

import transformers.models.qwen3.modeling_qwen3  # noqa: F401,E402
from quantization.hub import LittleBitModel  # noqa: E402
from quantization.modules import LittleBitLinear  # noqa: E402
import torch

m = LittleBitModel.from_pretrained("models/littlebit-qwen3-8b-0.55bpw",
                                   torch_dtype=torch.bfloat16, device="cpu")
lbl = [n for n, _ in m.named_modules() if isinstance(_, LittleBitLinear)][:3]
mlp = [n for n, _ in m.named_modules() if "layers.0.mlp" in n][:5]
print("LBL:", lbl)
print("MLP:", mlp)
print("type m:", type(m).__name__, "| m.model:", type(m.model).__name__)
