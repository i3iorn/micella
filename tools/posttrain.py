"""M3: teacher-free post-training LittleBit factorization.

Streams each nn.Linear weight from the HF checkpoint (one matrix on GPU at a
time), applies the exact LittleBit init math (svd_lowrank -> sqrt(S) split ->
rank-1 magnitude decomposition -> optional residual pair), binarizes factors,
and saves a packed checkpoint loadable by the stock LittleBitModel loader.
No QAT, no teacher - the quality question of M3.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

REPO = Path("C:/Users/micro/Code/Projects/littlebit")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))
sys.path.insert(0, str(Path(__file__).parent))

import transformers.models.qwen3.modeling_qwen3  # noqa: F401,E402
import quantization.utils.quant_util  # noqa: F401,E402  (breaks circular init)
from quantization.functions import SmoothSign  # noqa: E402
from quantization.modules import LittleBitLinear  # noqa: E402
from quantization.utils.quant_util import apply_littlebit_patch  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402


def itq_rotate(U, V, n_iter=50):
    """LittleBit-2's joint ITQ: rotate (U,V) so latent factors align with the
    binary hypercube before rank-1 magnitude decomposition (teacher-free)."""
    with torch.no_grad():
        X = torch.cat([U, V.t()], dim=0).float()
        dim = X.shape[1]
        R = torch.empty((dim, dim), device=X.device, dtype=torch.float32)
        torch.nn.init.orthogonal_(R)
        for _ in range(n_iter):
            Z = X @ R
            B = torch.sign(Z)
            B[B == 0] = 1
            M = B.t() @ X
            Up, _, Vtp = torch.linalg.svd(M, full_matrices=False)
            R = Vtp.t() @ Up.t()
        return (U @ R).contiguous(), (R.t() @ V).contiguous()


def svd_factor(W, split, device, use_itq=False):
    """LittleBit's _decompose_matrix math (fp32 on GPU). Returns U,V + scalars."""
    W = W.to(device=device, dtype=torch.float32)
    U_t, S_t, V_t = torch.svd_lowrank(W, q=split)
    sqrt_S = torch.sqrt(torch.diag(S_t))[:, :split]
    U = (U_t @ sqrt_S).contiguous()
    V = (sqrt_S.t() @ V_t.t()).contiguous()
    if use_itq:
        U, V = itq_rotate(U, V)
    u1, u2 = rank1(U.abs(), device)
    v1, v2 = rank1(V.abs(), device)
    return U, V, u1, u2, v1, v2


def rank1(X, device):
    """LittleBit's _rank_one_decompose: X (m,n) -> u (1,m), v (1,n) with X~u^T*v."""
    U, S, V = torch.svd_lowrank(X.to(device), q=1)
    s0 = torch.sqrt(S[0])
    return (U[:, :1] * s0).t().contiguous(), (s0 * V.t()[:1, :]).contiguous()


def approx_from_factors(U, V, u1, u2, v1, v2, device):
    """W_approx per littlebit.py: (sign(U)*u1^T u2) @ (sign(V)*v1^T v2)."""
    Uq = torch.where(U >= 0, 1.0, -1.0).to(device)
    Vq = torch.where(V >= 0, 1.0, -1.0).to(device)
    Ua = Uq * (u1.t().to(device) @ u2.to(device))
    Va = Vq * (v1.t().to(device) @ v2.to(device))
    return (Ua @ Va).to(U.device)


def assign(mod, U, V, u1, u2, v1, v2):
    def param(t, dtype=None):
        t = t if dtype is None else t.to(dtype)
        return torch.nn.Parameter(t, requires_grad=False)
    # factors must be SIGN-quantized to +-1 int8 (truncating the float SVD
    # factors to int8 stores mostly zeros -> garbage model)
    mod.U = param(torch.where(U >= 0, 1.0, -1.0).to(torch.int8))
    mod.V = param(torch.where(V >= 0, 1.0, -1.0).to(torch.int8))
    mod.u1, mod.u2 = param(u1.to(torch.bfloat16)), param(u2.to(torch.bfloat16))
    mod.v1, mod.v2 = param(v1.to(torch.bfloat16)), param(v2.to(torch.bfloat16))
    mod.register_parameter("weight", None)


def factorize_model(repo_id, out_dir, eff_bit, residual=True, use_itq=False):
    cfg = AutoConfig.from_pretrained(repo_id)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    model.to_empty(device="cpu")
    args = argparse.Namespace(quant_func="SmoothSign", quant_mod="LittleBitLinear",
                              residual=residual, split_dim=1024, eff_bit=eff_bit,
                              min_split_dim=8, kv_factor=1.0, use_itq=use_itq,
                              itq_n_iter=50, model_id=repo_id)
    model = apply_littlebit_patch(model, args, do_train=False)

    # weight-file map for streaming single matrices off disk
    idx_path = Path(repo_id) / "model.safetensors.index.json"
    wmap = {}
    if idx_path.exists():
        wmap = json.loads(idx_path.read_text())["weight_map"]
    else:
        wmap = {k: "model.safetensors" for k in tensor_keys(repo_id)}

    handles = {}
    def get_weight(name):
        f = wmap[name]
        if f not in handles:
            handles[f] = safe_open(Path(repo_id) / f, framework="pt", device="cpu")
        return handles[f].get_tensor(name)

    device = torch.device("cuda")
    mods = [(n, m) for n, m in model.named_modules()
            if isinstance(m, LittleBitLinear)]
    print(f"factorizing {len(mods)} linears at eff_bit={eff_bit} residual={residual}")
    for i, (name, mod) in enumerate(mods):
        W = get_weight(f"{name}.weight").to(device)
        split = int(mod._split_dim_final.item())
        U, V, u1, u2, v1, v2 = svd_factor(W, split, device, use_itq=use_itq)
        assign(mod, U, V, u1, u2, v1, v2)
        if residual:
            approx = approx_from_factors(U.float().cpu(), V.float().cpu(),
                                         u1, u2, v1, v2, device)
            W_res = (W.float() - approx.to(device))
            Ur, Vr, u1r, u2r, v1r, v2r = svd_factor(W_res, split, device,
                                                    use_itq=use_itq)
            mod.U_R = torch.nn.Parameter(Ur.to(torch.int8), requires_grad=False)
            mod.V_R = torch.nn.Parameter(Vr.to(torch.int8), requires_grad=False)
            mod.u1_R = torch.nn.Parameter(u1r.to(torch.bfloat16), requires_grad=False)
            mod.u2_R = torch.nn.Parameter(u2r.to(torch.bfloat16), requires_grad=False)
            mod.v1_R = torch.nn.Parameter(v1r.to(torch.bfloat16), requires_grad=False)
            mod.v2_R = torch.nn.Parameter(v2r.to(torch.bfloat16), requires_grad=False)
            mod._binarized = False
        del W
        if i % 20 == 0:
            print(f"  {i}/{len(mods)}")
    # load every non-factorized weight (norms, embed, lm_head, ropes)
    factorized = {n for n, _ in mods}
    sd = {}
    for k in wmap:
        if k.endswith(".weight") and k[:-len(".weight")] in factorized:
            continue
        sd[k] = get_weight(k)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    model.tie_weights()
    if getattr(cfg, "tie_word_embeddings", False):
        model.lm_head.weight = model.get_input_embeddings().weight
    print(f"loaded {len(sd)} aux tensors; missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing sample:", missing[:5])

    save_packed(model, cfg, out_dir, eff_bit, residual)
    return out_dir


def tensor_keys(repo_id):
    with safe_open(Path(repo_id) / "model.safetensors", framework="pt",
                   device="cpu") as f:
        return list(f.keys())


def save_packed(model, cfg, out_dir, eff_bit, residual):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    for k, v in dict(quant_func="SmoothSign", eff_bit=eff_bit, split_dim=1024,
                     residual=residual, kv_factor=1.0, min_split_dim=8,
                     quant_mod="LittleBitLinear", use_itq=False,
                     itq_n_iter=50).items():
        setattr(model.config, k, v)
    model.save_pretrained(out, safe_serialization=True)
    (out / "littlebit_config.json").write_text(json.dumps(dict(
        quant_func="SmoothSign", eff_bit=eff_bit, split_dim=1024,
        residual=residual, kv_factor=1.0, min_split_dim=8,
        quant_mod="LittleBitLinear", use_itq=False, itq_n_iter=50), indent=2))
    print(f"saved packed checkpoint -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id")
    ap.add_argument("out_dir")
    ap.add_argument("--eff-bit", type=float, default=0.55)
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--use-itq", action="store_true")
    args = ap.parse_args()
    factorize_model(args.repo_id, args.out_dir, args.eff_bit,
                    residual=not args.no_residual, use_itq=args.use_itq)


if __name__ == "__main__":
    main()
