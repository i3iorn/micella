"""Gate 2 (decisive): 1 bpw factorization of a real MLP block, judged in
ACTIVATION space. Methods ladder on each of gate/up/down:
  (a) SVD init, rank-1 scalars           (the M3 baseline that failed)
  (b) + activation-space u1 row fit      (per-output-row LSQ on real x)
  (c) + joint activation-space c_j fit   (iterating b/c)
  (d) ITQ before rank-1                  (LittleBit-2 init)
Reference bar: the int4 control measured cos 0.974-0.984 (gate1). Gate:
cos >= 0.95 marginal / >= 0.97 strong. FAIL at 1 and 2 bpw -> kill criterion.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from calibrate import act_cos, load_calib  # noqa: E402
from posttrain import svd_factor  # noqa: E402

MODEL = "models/qwen3-0.6b"
CACHE = "models/qwen3-0.6b/calib"
PREFIX = "model.layers.0.mlp"


def split_for(out_f, in_f, eff_bit, residual=False):
    base = out_f + in_f + 16
    s = 32 * (out_f + in_f) if residual else 16 * (out_f + in_f)
    num = out_f * in_f * eff_bit - s
    den = 2 * base if residual else base
    cap = min(out_f, in_f)  # rank can never exceed the smaller dimension
    return min(max(int(num / den // 8 * 8), 8), cap // 8 * 8)


def fit_activation_scales(Uq, Vq, u1_0, c_0, W, x, v2, iters=3):
    """Iterate: u1_k = <y_ref_k, yhat_k>/<yhat_k, yhat_k> ; c via activation
    LSQ. z must include the v2 input-side scale or the fit optimizes a
    different model than recon() reconstructs (that mismatch was producing
    negative cos)."""
    u1 = u1_0.clone()
    c = c_0.clone()
    z = (x * v2[None, :]) @ Vq.t()  # (T, r) latent activations incl. v2
    y_ref = x @ W.t()               # (T, out)
    for _ in range(iters):
        # y_hat(t,k) = u1_k * (z(t,:) * c) . Uq[k,:] -> fit u1 per row
        zk = z * c[None, :]
        yhat = zk @ Uq.t()                       # (T, out), u1=1 implied
        num = (y_ref * yhat).sum(0)
        den = (yhat * yhat).sum(0).clamp_min(1e-9)
        u1 = num / den
        # fit c: y_ref(t,k) = u1_k * sum_j c_j z(t,j) Uq[k,j] -> linear in c
        # target: A c = b with A[(t,k), j] = u1_k z(t,j) Uq[k,j]; solve via
        # normal equations on the (T*out, r) system, chunked over t.
        ATA = torch.zeros(z.shape[1], z.shape[1], device=x.device)
        ATb = torch.zeros(z.shape[1], device=x.device)
        for s in range(0, z.shape[0], 512):
            zs = z[s:s + 512] * c[None, :] * 0  # recompute properly below
            break
        # normal equations for min ||y_ref - (u1*Uq) @ diag(c) @ z^T||:
        # c_j enters as y_hat = Uc @ diag(c) @ Z^T with Uc = u1[:,None]*Uq
        Uc = u1[:, None] * Uq
        for s in range(0, z.shape[0], 1024):
            Zs = z[s:s + 1024]                    # (t, r)
            Ys = y_ref[s:s + 1024]                # (t, out)
            # y_hat(t,k) = sum_j c_j Zs(t,j) Uc(k,j) = (Zs diag(c) Uc^T)
            # d_j = <Zs[:,j]*Uc[:,j]... per-t: sum_k Uc(k,j) y_ref(t,k)
            ATb += ((Zs * (Ys @ Uc)).sum(0))
            # G[j,j'] = sum_t Zs(t,j) Zs(t,j') * sum_k Uc(k,j) Uc(k,j')
            w_jp = Uc.t() @ Uc                    # (r, r) - constant
            ATA += Zs.t() @ Zs * 0 + Zs.t() @ Zs  # placeholder, replaced below
            break
        # G[j,j'] = (Z^T Z)[j,j'] * (Uc^T Uc)[j,j']  (separable, exact)
        ZtZ = z.t() @ z
        UcUc = Uc.t() @ Uc
        G = ZtZ * UcUc
        d = ((z * (y_ref @ Uc)).sum(0))
        c = torch.linalg.lstsq(G, d.unsqueeze(1)).solution.squeeze(1)
        # lstsq is min-norm on a rank-deficient G: a global sign flip of c is
        # a valid solution - reject it (u1 was fit against the previous c).
        yh = (z * c[None, :]) @ Uc.t()
        if (yh * y_ref).sum() < 0:
            c = -c
    return u1, c


def recon(Uq, Vq, u1, c, v2, x=None):
    W = (Uq * (u1[:, None] * c[None, :])) @ (Vq * v2.flatten()[None, :])
    return W


def main():
    torch.manual_seed(0)
    Ws = {}
    import safetensors.torch as st
    sd = st.load_file("models/qwen3-0.6b/model.safetensors")
    for n in ("gate_proj", "up_proj", "down_proj"):
        Ws[n] = sd[f"model.layers.0.mlp.{n}.weight"].float().cuda()
        xs = load_calib(CACHE, f"model.layers.0.mlp.{n}")
        Ws[n + ":x"] = xs
    results = {}
    for eff_bit in (1.0, 2.0):
        for n in ("gate_proj", "up_proj", "down_proj"):
            W = Ws[n]
            x = Ws[n + ":x"]
            out_f, in_f = W.shape
            split = split_for(out_f, in_f, eff_bit, residual=False)
            row = {}
            for use_itq in (False, True):
                U, V, u1r, u2r, v1r, v2r = svd_factor(W, split, "cuda",
                                                      use_itq=use_itq)
                Uq = torch.where(U >= 0, 1.0, -1.0)
                Vq = torch.where(V >= 0, 1.0, -1.0)
                v2 = v2r.flatten()  # input-side scale from rank1(|V|)
                tag = "itq" if use_itq else "svd"
                # (a) SVD/ITQ init scalars as-is (u2*v1 product -> c)
                c0 = (u2r.flatten() * v1r.flatten()).float()
                u1_0 = u1r.flatten().float() / u1r.flatten().abs().mean()
                Wa = recon(Uq, Vq, u1_0, c0, v2)
                row[f"{tag}_a_init"] = act_cos(W, Wa, x)[0]
                # (b/c) activation-space refit
                u1, c = fit_activation_scales(Uq, Vq, u1_0, c0, W, x, v2)
                Wb = recon(Uq, Vq, u1, c, v2)
                row[f"{tag}_bc_actfit"] = act_cos(W, Wb, x)[0]
            results[f"{n}@{eff_bit}bpw(split={split})"] = {
                k: round(v, 4) for k, v in row.items()}
            print(n, eff_bit, json.dumps({k: round(v, 4) for k, v in row.items()}))
    out = Path(__file__).parent / "results" / "gate2.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
