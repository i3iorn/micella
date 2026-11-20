"""Compare: (a) official QAT factors' weight reconstruction error,
(b) post-training SVD init, (c) SVD init + optimal per-latent scale c_j."""
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent))
from posttrain import svd_factor  # noqa: E402


def load_packed_factors(repo, layer="model.layers.0.self_attn.q_proj"):
    """Unpack official U/V (+residual) and scalars for one module."""
    idx = Path(repo) / "model.safetensors.index.json"
    if idx.exists():
        files = sorted(set(json.loads(idx.read_text())["weight_map"].values()))
    else:
        files = ["model.safetensors"]
    sd = {}
    for f in files:
        with safe_open(Path(repo) / f, framework="pt", device="cpu") as h:
            for k in h.keys():
                if k.startswith(layer):
                    sd[k] = h.get_tensor(k)
    fac = {}
    for p in ("U", "V", "U_R", "V_R"):
        pk, sk = f"{layer}.{p}_packed", f"{layer}.{p}_shape"
        if pk in sd:
            packed, shape = sd[pk], tuple(sd[sk].tolist())
            n_rows, n_cols = shape
            wpr = (n_cols + 31) // 32
            bits = ((1 - sd[pk].int().unsqueeze(-1)) // 1)  # placeholder
            out = torch.empty(n_rows, wpr * 32, dtype=torch.int8)
            for wi in range(wpr):
                b = ((sd[pk][:, wi].int().unsqueeze(1) >>
                      torch.arange(32).unsqueeze(0)) & 1).to(torch.int8)
                out[:, wi * 32:(wi + 1) * 32] = b
            fac[p] = ((1 - 2 * out[:, :n_cols]).float().cuda())
    return fac, sd


def W_impl(Uq, Vq, u1, u2, v1, v2):
    return (Uq * (u1.t() @ u2)) @ (Vq * (v1.t() @ v2))


def optimal_c(Uq, Vq, W):
    """JOINT LSQ: min_c ||W - A diag(c) B||_F^2 with A=Uq (out,r), B=Vq (r,in).
    Normal equations G c = d; G[i,j] = <a_i,a_j><b_i,b_j> (separable Gram)."""
    A, B = Uq, Vq
    G = (A.t() @ A) * (B @ B.t())
    # d_j = <a_j b_j^T, W> = sum_{k,i} A[k,j] B[j,i] W[k,i]
    d = ((A.t() @ W) @ B.t()).diagonal()
    return torch.linalg.solve(G, d)


def main():
    W = get_w = None
    sys.path.insert(0, ".")
    from debug_factorize import get_w
    W = get_w("model.layers.0.self_attn.q_proj.weight").float().cuda()
    out_f, in_f = W.shape
    rel = lambda X: ((X - W).norm() / W.norm()).item()

    print("=== (a) official QAT factors (needs original 8B weights) ===")
    try:
        W8 = get_w("model.layers.0.self_attn.q_proj.weight",
                   Path("models/qwen3-8b")).float().cuda()
        fac, sd = load_packed_factors("models/littlebit-qwen3-8b-0.55bpw")
        if "U" in fac:
            u1 = sd["model.layers.0.self_attn.q_proj.u1"].float().cuda()
            u2 = sd["model.layers.0.self_attn.q_proj.u2"].float().cuda()
            v1 = sd["model.layers.0.self_attn.q_proj.v1"].float().cuda()
            v2 = sd["model.layers.0.self_attn.q_proj.v2"].float().cuda()
            W_official = W_impl(fac["U"], fac["V"], u1, u2, v1, v2)
            rel8 = lambda X: ((X - W8).norm() / W8.norm()).item()
            print(f"official main pair rel err: {rel8(W_official):.4f}")
            if "U_R" in fac:
                u1r = sd["model.layers.0.self_attn.q_proj.u1_R"].float().cuda()
                u2r = sd["model.layers.0.self_attn.q_proj.u2_R"].float().cuda()
                v1r = sd["model.layers.0.self_attn.q_proj.v1_R"].float().cuda()
                v2r = sd["model.layers.0.self_attn.q_proj.v2_R"].float().cuda()
                W_official += W_impl(fac["U_R"], fac["V_R"], u1r, u2r, v1r, v2r)
                print(f"official main+residual rel err: {rel8(W_official):.4f}")
    except FileNotFoundError as e:
        print(f"(skipped: {e})")

    print("=== (b/c) SVD init on 0.6B weight ===")
    base = out_f + in_f + 16
    split = int((out_f * in_f * 0.55 - 32 * (out_f + in_f)) / (2 * base) // 8 * 8)
    U, V, u1, u2, v1, v2 = svd_factor(W, split, "cuda")
    Uq = torch.where(U >= 0, 1.0, -1.0)
    Vq = torch.where(V >= 0, 1.0, -1.0)
    print(f"(b) svd-init rel err: {rel(W_impl(Uq, Vq, u1.float(), u2.float(), v1.float(), v2.float())):.4f}")
    c = optimal_c(Uq, Vq, W)
    W_cal = Uq @ (c.unsqueeze(1) * Vq)
    print(f"(c) svd + optimal c_j rel err: {rel(W_cal):.4f}")
    # sanity: what does plain rank-`split` SVD (no binarization) give?
    Wsvd = U @ V
    print(f"    (ref) plain SVD rank-{split} rel err: {rel(Wsvd):.4f}")


if __name__ == "__main__":
    main()
