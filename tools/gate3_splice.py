"""Gate 3 (splice test on 0.6B): factorize layer-0 MLP at 1 bpw with the best
method (ITQ + activation-space scale fit), splice into the bf16 model, and
measure the real ppl damage. Also runs the naive (no-actfit) variant for
comparison. The cos numbers above said these should both work; this decides.
"""
import json
import sys
from pathlib import Path

import torch
import safetensors.torch as st

sys.path.insert(0, str(Path(__file__).parent))
from calibrate import load_calib  # noqa: E402
from posttrain import svd_factor  # noqa: E402
from eval_ppl import ppl_eval  # noqa: E402
from gate2_test import fit_activation_scales  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL = "models/qwen3-0.6b"
CACHE = "models/qwen3-0.6b/calib"
MAT_NAMES = ("gate_proj", "up_proj", "down_proj")


def split_for(out_f, in_f, eff_bit, residual=False):
    base = out_f + in_f + 16
    s = 32 * (out_f + in_f) if residual else 16 * (out_f + in_f)
    num = out_f * in_f * eff_bit - s
    den = 2 * base if residual else base
    cap = min(out_f, in_f)
    return min(max(int(num / den // 8 * 8), 8), cap // 8 * 8)


def factor_matrix(W, x, eff_bit, use_itq=True, residual=True, actfit=True):
    """Returns factored LittleBitLinear-style params for W at eff_bit."""
    out_f, in_f = W.shape
    split = split_for(out_f, in_f, eff_bit, residual)
    U, V, u1r, u2r, v1r, v2r = svd_factor(W, split, "cuda", use_itq=use_itq)
    Uq = torch.where(U >= 0, 1.0, -1.0)
    Vq = torch.where(V >= 0, 1.0, -1.0)
    c = (u2r.flatten() * v1r.flatten()).float()
    u1 = u1r.flatten().float() / u1r.flatten().abs().mean().clamp_min(1e-9)
    v2 = v2r.flatten().float()
    if actfit:
        u1, c = fit_activation_scales(Uq, Vq, u1, c, W, x, v2)
    W_hat = (Uq * (u1[:, None] * c[None, :])) @ (Vq * v2[None, :])
    if residual:
        split_r = split_for(out_f, in_f, eff_bit, True)
        U2, V2, u1r2, u2r2, v1r2, v2r2 = svd_factor(W - W_hat, split_r, "cuda",
                                                    use_itq=use_itq)
        Uq2 = torch.where(U2 >= 0, 1.0, -1.0)
        Vq2 = torch.where(V2 >= 0, 1.0, -1.0)
        c2 = (u2r2.flatten() * v1r2.flatten()).float()
        u1b = u1r2.flatten().float() / u1r2.flatten().abs().mean().clamp_min(1e-9)
        v2b = v2r2.flatten().float()
        if actfit:
            u1b, c2 = fit_activation_scales(Uq2, Vq2, u1b, c2, W - W_hat, x, v2b)
        W_hat = W_hat + (Uq2 * (u1b[:, None] * c2[None, :])) @ (Vq2 * v2b[None, :])
    return W_hat


def splice(model, layer, tag):
    sd = st.load_file(f"{MODEL}/model.safetensors")
    blk = model.model.layers[0]
    x_by_name = {}
    for n in MAT_NAMES:
        x_by_name[n] = load_calib(CACHE, f"model.layers.0.mlp.{n}")
    for n in MAT_NAMES:
        m = getattr(blk.mlp, n)
        W_ref = sd[f"model.layers.0.mlp.{n}.weight"].float().cuda()
        W_hat = factor_matrix(W_ref, x_by_name[n], 1.0, actfit=(tag == "actfit"))
        m.weight.data = W_hat.to(m.weight.dtype)
        del W_ref, W_hat


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
        low_cpu_mem_usage=True).eval()
    p0, _ = ppl_eval(base, tok, n_chunks=20)
    print(f"bf16 baseline ppl: {p0:.4f}")

    results = {"ppl_bf16": round(p0, 4)}
    for tag in ("naive", "actfit"):
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
            low_cpu_mem_usage=True).eval()
        splice(model, 0, tag)
        p, _ = ppl_eval(model, tok, n_chunks=20)
        results[f"ppl_layer0mlp_1bpw_{tag}"] = round(p, 4)
        results[f"delta_pct_{tag}"] = round((p - p0) / p0 * 100, 3)
        print(f"{tag}: ppl {p:.4f} (delta {results[f'delta_pct_{tag}']}%)")
        del model
        torch.cuda.empty_cache()
    out = Path(__file__).parent / "results" / "gate3_splice.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
