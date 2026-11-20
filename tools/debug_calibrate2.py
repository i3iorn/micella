"""Calibration comparison on the 0.6B q_proj weight:
(a) rank-1 scalars as-is, (b) global Frobenius rescale,
(c) joint-LSQ per-latent c_j keeping u1/v2 row/col scales."""
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent))
from posttrain import svd_factor  # noqa: E402


def get_w(name, repo=Path("models/qwen3-0.6b")):
    with safe_open(repo / "model.safetensors", framework="pt", device="cpu") as h:
        return h.get_tensor(name)


def W_impl(Uq, Vq, u1, u2, v1, v2):
    return (Uq * (u1.t() @ u2)) @ (Vq * (v1.t() @ v2))


def lsq_c(Uq, Vq, u1, v2, W):
    """min_c ||diag(u1) Uq diag(c) Vq diag(v2) - W||_F^2, separable Gram."""
    A = Uq * u1.unsqueeze(1)  # (out, r)
    B = Vq * v2.unsqueeze(0)  # (r, in)
    G = (A.t() @ A) * (B @ B.t())
    d = ((A.t() @ W) @ B.t()).diagonal()
    return torch.linalg.lstsq(G, d.unsqueeze(1)).solution.squeeze(1)


def main():
    W = get_w("model.layers.0.self_attn.q_proj.weight").float().cuda()
    rel = lambda X: ((X - W).norm() / W.norm()).item()
    out_f, in_f = W.shape
    split = int((out_f * in_f * 0.55 - 32 * (out_f + in_f)) / (2 * (out_f + in_f + 16)) // 8 * 8)
    print(f"|W|_F = {W.norm():.3f}, split = {split}")

    U, V, u1, u2, v1, v2 = svd_factor(W, split, "cuda")
    Uq = torch.where(U >= 0, 1.0, -1.0)
    Vq = torch.where(V >= 0, 1.0, -1.0)
    u1, u2, v1, v2 = u1.float(), u2.float(), v1.float(), v2.float()

    Wa = W_impl(Uq, Vq, u1, u2, v1, v2)
    print(f"(a) rank-1 scalars:   rel={rel(Wa):.4f}  |Wa|_F={Wa.norm():.3f}")

    s = (W.norm() / Wa.norm())
    print(f"(b) global rescale:   rel={rel(Wa * s):.4f}")

    c = lsq_c(Uq, Vq, u1.flatten(), v2.flatten(), W)
    u1c, v2r = u1.flatten().unsqueeze(1), v2.flatten().unsqueeze(0)
    Wc2 = ((u1c * Uq) * c[None, :]) @ (Vq * v2r)
    print(f"(c) LSQ c_j:          rel={rel(Wc2):.4f}")

    # (e) sign refinement via iterated conditional modes (teacher-free):
    # flip Uq[k,j] / Vq[j,i] whenever it provably lowers ||W - R||_F^2.
    # R[k,i] = sum_j u1_k Uq[k,j] c_j Vq[j,i] v2_i
    Uq_s, Vq_s = Uq.clone(), Vq.clone()
    Uc = u1c * Uq_s          # (out, r) scaled U pattern
    for sweep in range(6):
        R = ((Uc) * c[None, :]) @ (Vq_s * v2r)
        Res = W - R
        # U flip (k,j): dE = 4 u1_k Uq[k,j] c_j g[k,j] + 4 (u1_k c_j)^2 n2_j
        gU = (Res * v2r) @ Vq_s.t()                       # (out, r)
        n2v = ((Vq_s * v2r) ** 2).sum(1)                  # (r,)
        dEU = 4 * Uq_s * u1c * c[None, :] * gU + \
            4 * (u1c * c[None, :]) ** 2 * n2v[None, :]
        Uq_s = Uq_s * torch.where(dEU < 0, -1.0, 1.0)
        Uc = u1c * Uq_s
        R = (Uc * c[None, :]) @ (Vq_s * v2r)
        Res = W - R
        # V flip (j,i): dE = 4 c_j Vq[j,i] v2_i g[j,i] + 4 (c_j v2_i)^2 m2_j
        gV = (Uc.t() @ Res)                               # (r, in)
        m2u = (Uc ** 2).sum(0)                            # (r,)
        dEV = 4 * Vq_s * c[:, None] * v2r * gV + \
            4 * (c[:, None] * v2r) ** 2 * m2u[:, None]
        Vq_s = Vq_s * torch.where(dEV < 0, -1.0, 1.0)
        c = lsq_c(Uq_s, Vq_s, u1.flatten(), v2.flatten(), W)
        Uc = u1c * Uq_s
        R = (Uc * c[None, :]) @ (Vq_s * v2r)
        print(f"  ICM sweep {sweep + 1}: rel={rel(R):.4f}")
    print(f"(e) SVD + ICM sign refinement: rel={rel(R):.4f}")


if __name__ == "__main__":
    main()
