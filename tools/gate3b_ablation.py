"""Gate 3b: ablation of activation-fit improvements, single-layer ppl delta.

Configs (all 1 bpw, layer-0 MLP of Qwen3-0.6B, ITQ + u1/c activation fit):
  a) base          - 2048 calib tokens, u1+c alternation (the +72% config)
  b) +more tokens  - 8192 calib tokens
  c) +v2 fit       - b + exact 1D LSQ for the input-side scale
  d) +whitening    - c + ASVD: W scaled by activation RMS pre-SVD, folded
                     back into v2 (free)
Each config reports spliced ppl + delta. Gate bar: single-layer delta should
fall well below +72%; ~+10% per layer would make 28-layer compaction viable.
"""
import json
import sys
from pathlib import Path

import torch
import safetensors.torch as st

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "C:/Users/micro/Code/Projects/littlebit")
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))

import transformers.models.qwen3.modeling_qwen3  # noqa: F401,E402
from posttrain import svd_factor  # noqa: E402
from eval_ppl import ppl_eval  # noqa: E402
from calibrate import calib_text  # noqa: E402
from gate2_test import split_for  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL = "models/qwen3-0.6b"
MAT_NAMES = ("gate_proj", "up_proj", "down_proj")


@torch.no_grad()
def get_calib_x(model, tok, n_tokens):
    ids = calib_text(tok, n_tokens=n_tokens)
    store = {}

    def hook(name):
        def fn(mod, inp, out):
            if name not in store:
                x = inp[0].detach()
                store[name] = x.reshape(-1, x.shape[-1]).to(torch.float16).cpu()
        return fn

    handles = [mod.register_forward_hook(hook(f"{n}"))
               for n, mod in model.named_modules()
               if isinstance(mod, torch.nn.Linear) and "layers.0.mlp" in n]
    model(ids.cuda(), use_cache=False)
    for h in handles:
        h.remove()
    return {k: v.float().cuda() for k, v in store.items()}


def ridge_solve(G, d):
    """lstsq with ridge scaled to G's diagonal - the unscaled solves diverge
    (calibration overfit / ill-conditioning) as the ablation showed."""
    lam = 1e-3 * G.diagonal().mean().clamp_min(1e-12)
    return torch.linalg.lstsq(G + lam * torch.eye(G.shape[0], device=G.device),
                              d.unsqueeze(1)).solution.squeeze(1)


def fit_all(Uq, Vq, u1r, u2r, v1r, v2r, W, x, iters):
    """Alternating exact 1D/LSQ fits for u1 (rows), c (latent), v2 (cols)."""
    u1 = u1r.flatten().float() / u1r.flatten().abs().mean().clamp_min(1e-9)
    c = (u2r.flatten() * v1r.flatten()).float()
    v2 = v2r.flatten().float().abs() + 1e-6
    y_ref = x @ W.t()
    A = (u1[:, None] * Uq) * c[None, :]  # (out, r)
    for _ in range(iters):
        # u1: y_hat(t,k) = u1_k * sum_j c_j z(t,j) Uq[k,j]
        z = (x * v2[None, :]) @ Vq.t()                # (T, r)
        yh = (z * c[None, :]) @ Uq.t()
        u1 = (y_ref * yh).sum(0) / (yh * yh).sum(0).clamp_min(1e-9)
        # c: separable normal equations (z-side and Uc-side Grams)
        Uc = u1[:, None] * Uq
        G = (z.t() @ z) * (Uc.t() @ Uc)
        d = (z * (y_ref @ Uc)).sum(0)
        c = ridge_solve(G, d)
        yh = (z * c[None, :]) @ Uc.t()
        if (yh * y_ref).sum() < 0:                    # min-norm sign guard
            c = -c
        A = (u1[:, None] * Uq) * c[None, :]
        # v2: y_hat(t,k) = sum_i v2_i x[t,i] (A Vq)[k,i] -> separable Gram
        AV = A @ Vq                                   # (out, in)
        Gv = (x.t() @ x) * (AV.t() @ AV)
        dv = (x * (y_ref @ AV)).sum(0)
        v2 = ridge_solve(Gv, dv)
    return u1, c, v2


def factor(W, x, eff_bit, iters, whitening):
    out_f, in_f = W.shape
    split = split_for(out_f, in_f, eff_bit, residual=True)
    rms = x.float().square().mean(0).sqrt().clamp_min(1e-6)
    Ww = W * (rms[None, :] if whitening else 1.0)
    U, V, u1r, u2r, v1r, v2r = svd_factor(Ww, split, "cuda", use_itq=True)
    Uq = torch.where(U >= 0, 1.0, -1.0)
    Vq = torch.where(V >= 0, 1.0, -1.0)
    u1, c, v2 = fit_all(Uq, Vq, u1r, u2r, v1r, v2r, Ww, x * (rms[None, :] if whitening else 1.0), iters)
    if whitening:
        v2 = v2 * rms
    W_hat = ((u1[:, None] * Uq) * c[None, :]) @ (Vq * v2[None, :])
    # residual pair (same recipe on W - W_hat)
    W_res = W - W_hat
    U, V, u1r, u2r, v1r, v2r = svd_factor(W_res, split, "cuda", use_itq=True)
    Uq = torch.where(U >= 0, 1.0, -1.0)
    Vq = torch.where(V >= 0, 1.0, -1.0)
    u1, c, v2 = fit_all(Uq, Vq, u1r, u2r, v1r, v2r, W_res, x, iters)
    W_hat = W_hat + ((u1[:, None] * Uq) * c[None, :]) @ (Vq * v2[None, :])
    return W_hat


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
        low_cpu_mem_usage=True).eval()
    p0, _ = ppl_eval(base, tok, n_chunks=20)
    print(f"bf16 baseline: {p0:.4f}")
    sd = st.load_file(f"{MODEL}/model.safetensors")

    configs = [
        ("base_2k_ridge", 2048, 3, False),
        ("ridge_v2fit_2k", 2048, 6, False),
        ("ridge_v2fit_8k", 8192, 6, False),
    ]
    results = {"ppl_bf16": round(p0, 4)}
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
        low_cpu_mem_usage=True).eval()
    for tag, n_tok, iters, whiten in configs:
        xs = get_calib_x(model, tok, n_tok)
        model.model.layers[0].mlp.load_state_dict(
            {k: v.to(torch.bfloat16) for k, v in sd.items()
             if k.startswith("model.layers.0.mlp")}, strict=False)
        for n in MAT_NAMES:
            W_ref = sd[f"model.layers.0.mlp.{n}.weight"].float().cuda()
            W_hat = factor(W_ref, xs[f"model.layers.0.mlp.{n}"], 1.0, iters, whiten)
            getattr(model.model.layers[0].mlp, n).weight.data = W_hat.to(torch.bfloat16)
            err = ((W_hat - W_ref).norm() / W_ref.norm()).item()
            print(f"  {tag} {n}: weight rel err {err:.4f}")
        p, _ = ppl_eval(model, tok, n_chunks=20)
        d = round((p - p0) / p0 * 100, 2)
        results[tag] = {"ppl": round(p, 4), "delta_pct": d}
        print(f"{tag}: ppl {p:.4f} delta {d}%")
    out = Path(__file__).parent / "results" / "gate3b_ablation.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
