"""Stage-by-stage numerical check of the post-training factorization."""
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent))
from posttrain import svd_factor, rank1, approx_from_factors  # noqa: E402

REPO = Path("models/qwen3-0.6b")


def get_w(name, repo=REPO):
    idx = repo / "model.safetensors.index.json"
    if idx.exists():
        f = json.loads(idx.read_text())["weight_map"][name]
    else:
        f = "model.safetensors"
    with safe_open(repo / f, framework="pt", device="cpu") as h:
        return h.get_tensor(name)


def W_from_factors(Uq, Vq, u1, u2, v1, v2):
    """Exact LittleBit forward-implied weight (fp32)."""
    return (Uq * (u1.t() @ u2)) @ (Vq * (v1.t() @ v2))


def main():
    name = "model.layers.0.self_attn.q_proj.weight"
    W = get_w(name).float().cuda()
    out_f, in_f = W.shape
    eb = 0.55
    base = out_f + in_f + 16
    num = out_f * in_f * eb - 32 * (out_f + in_f)
    split = int((num / (2 * base)) // 8 * 8)
    print(f"W {tuple(W.shape)} split={split}")

    U, V, u1, u2, v1, v2 = svd_factor(W, split, "cuda")
    Uq = torch.where(U >= 0, 1.0, -1.0)
    Vq = torch.where(V >= 0, 1.0, -1.0)
    W_approx = W_from_factors(Uq, Vq, u1.float(), u2.float(), v1.float(), v2.float())
    rel1 = (W_approx - W).norm() / W.norm()
    print(f"main pair rel err: {rel1.item():.4f}")

    Ur, Vr, u1r, u2r, v1r, v2r = svd_factor(W - W_approx, split, "cuda")
    Uqr = torch.where(Ur >= 0, 1.0, -1.0)
    Vqr = torch.where(Vr >= 0, 1.0, -1.0)
    W_res = W_from_factors(Uqr, Vqr, u1r.float(), u2r.float(), v1r.float(), v2r.float())
    total = W_approx + W_res
    rel2 = (total - W).norm() / W.norm()
    print(f"main+residual rel err: {rel2.item():.4f}")

    # bf16 roundtrip of scalars + int8 factors (what actually gets saved)
    W_rt = W_from_factors(
        Uq.to(torch.int8).float(), Vq.to(torch.int8).float(),
        u1.to(torch.bfloat16).float(), u2.to(torch.bfloat16).float(),
        v1.to(torch.bfloat16).float(), v2.to(torch.bfloat16).float())
    rel_rt = (W_rt - W).norm() / W.norm()
    print(f"main pair after int8/bf16 roundtrip: {rel_rt.item():.4f}")


if __name__ == "__main__":
    main()
