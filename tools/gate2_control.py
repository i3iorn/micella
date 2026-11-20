"""Control: official QAT factors' ACTIVATION-space cos (the working regime).

Captures layer-0 mlp inputs by running the official 8B LittleBit checkpoint
(5.9 GB, fits), then compares y_ref (original bf16 W, streamed from
models/qwen3-8b) vs y_hat (QAT factor-implied W) on those real activations.
Completes the evidence for the gate-2 verdict.
"""
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, "C:/Users/micro/Code/Projects/littlebit")
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import transformers.models.qwen3.modeling_qwen3  # noqa: F401,E402
from quantization.hub import LittleBitModel  # noqa: E402
from quantization.modules import LittleBitLinear  # noqa: E402
from calibrate import act_cos, calib_text  # noqa: E402
from m1_baseline import to_int8_factors  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

PREFIX = "model.layers.0.mlp"


def load_orig_W(name):
    p = Path("models/qwen3-8b/model.safetensors.index.json")
    if p.exists():
        f = json.loads(p.read_text())["weight_map"][name]
    else:
        f = "model.safetensors"
    with safe_open(Path("models/qwen3-8b") / f, framework="pt", device="cpu") as h:
        return h.get_tensor(name).float().cuda()


def main():
    tok = AutoTokenizer.from_pretrained("models/littlebit-qwen3-8b-0.55bpw")
    model = LittleBitModel.from_pretrained("models/littlebit-qwen3-8b-0.55bpw",
                                           torch_dtype=torch.bfloat16, device="cpu")
    to_int8_factors(model)
    model.to("cuda").eval()

    store = {}
    handles = []

    def hook(name):
        def fn(mod, inp, out):
            if name not in store:
                x = inp[0].detach()
                store[name] = x.reshape(-1, x.shape[-1]).to(torch.float16).cpu()
        return fn

    from quantization.modules import LittleBitLinear
    for name, mod in model.named_modules():
        if isinstance(mod, (torch.nn.Linear, LittleBitLinear)) and "layers.0.mlp" in name:
            handles.append(mod.register_forward_hook(hook(name)))
    if not handles:
        raise RuntimeError("no layer-0 mlp linears found")
    ids = calib_text(tok, n_tokens=2048)
    with torch.no_grad():
        model(ids.cuda(), use_cache=False)
    for h in handles:
        h.remove()

    # factor-implied W per matrix (bf16 -> fp32, from the loaded int8 model)
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, LittleBitLinear) and "layers.0.mlp" in name:
            Uq = mod.U.data.float()
            Vq = mod.V.data.float()
            u1, u2 = mod.u1.data.float(), mod.u2.data.float()
            v1, v2 = mod.v1.data.float(), mod.v2.data.float()
            W_hat = (Uq * (u1.t() @ u2)) @ (Vq * (v1.t() @ v2))
            if mod.residual:
                Uqr, Vqr = mod.U_R.data.float(), mod.V_R.data.float()
                u1r, u2r = mod.u1_R.data.float(), mod.u2_R.data.float()
                v1r, v2r = mod.v1_R.data.float(), mod.v2_R.data.float()
                W_hat = W_hat + (Uqr * (u1r.t() @ u2r)) @ (Vqr * (v1r.t() @ v2r))
            leaf = name.split(".")[-1]
            W_ref = load_orig_W(f"{PREFIX}.{leaf}.weight")
            x = store[name].float().cuda()
            mean_cos, min_cos = act_cos(W_ref, W_hat, x)
            out[leaf] = {"mean": round(mean_cos, 4), "min": round(min_cos, 4)}
            print(f"official-QAT {leaf}: act cos mean={mean_cos:.4f} min={min_cos:.4f}")
            del W_ref, W_hat
    Path("tools/results/gate2_control.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
