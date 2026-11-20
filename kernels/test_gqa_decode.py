"""Correctness test: M=1 GQA decode attention kernel vs sdpa reference."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "C:/Users/micro/Code/Projects/littlebit")

import transformers.models.qwen3.modeling_qwen3  # noqa: F401,E402
import torch
import torch.nn.functional as F
from kernels.fast_model import micella_attention, patch_attention


def main():
    patch_attention(None)
    torch.manual_seed(0)
    HD, KVH, GH = 128, 8, 4
    n_heads = KVH * GH
    for MAXL, L in ((256, 1), (256, 17), (256, 64), (256, 255),
                    (110, 100), (32, 8), (32, 31), (100, 99)):
        q = torch.randn(1, n_heads, 1, HD, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, KVH, MAXL, HD, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(1, KVH, MAXL, HD, dtype=torch.bfloat16, device="cuda")
        mask = torch.zeros(1, 1, 1, MAXL, dtype=torch.bfloat16, device="cuda")
        mask[..., L:] = torch.finfo(torch.bfloat16).min
        out, _ = micella_attention(None, q, k, v, attention_mask=mask,
                                   scaling=HD ** -0.5)
        ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                             enable_gqa=True)
        err = (out.float() - ref.float()).abs().max().item()
        print(f"L={L:4d}: max abs err vs sdpa = {err:.5f}")
        assert err < 0.02, err
    print("GQA decode kernel: CORRECT")


if __name__ == "__main__":
    main()
