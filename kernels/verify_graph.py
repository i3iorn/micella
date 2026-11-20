"""Verify CUDA-graph capture completeness: replay must reproduce EVERY shape's
correct output, not just the first. Also times one replay per layer-content."""
import sys
import time
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).parent))
import factored_gemm as fg  # noqa: E402

SHAPES = [("kv", 4096, 1024, 208), ("qo", 4096, 4096, 544),
          ("gate_up", 4096, 12288, 824), ("down", 12288, 4096, 824)]


def main():
    state = []
    for name, i, o, r in SHAPES:
        x, V, U, v2, v1, u2, u1 = fg.make_inputs(i, o, r)
        h = torch.empty(r, dtype=torch.bfloat16, device="cuda")
        y = torch.empty(o, dtype=torch.bfloat16, device="cuda")
        state.append((name, x, V, U, v2, v1, u1, v1 * u2, h, y, i, o, r))
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _, x, V, U, v2, _, u1, s, h, y, i, o, r in state:
            fg._scale_v_kernel[(triton.cdiv(r, 32),)](
                x, V, v2, s, h, i, r, BLOCK_IN=512, BLOCK_OUT=32, num_warps=4)
            fg._scale_u_kernel[(triton.cdiv(o, 64),)](
                h, U, u1, y, r, o, BLOCK_R=1024, BLOCK_OUT=64, num_warps=4)
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    ok = True
    for name, x, V, U, v2, v1, u1, s, h, y, i, o, r in state:
        href = ((x.float() * v2.float()) @ V.float().t()) * s.float()
        yref = (href @ U.float().t()) * u1.float()
        err = ((y.float() - yref).abs().max() / yref.abs().max()).item()
        ok &= err < 0.02
        print(f"{name:8s} replay rel err {err:.4f} {'OK' if err < 0.02 else 'FAIL'}")
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        g.replay()
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / 200 * 1e6
    layer_bytes = sum(2 * (r * i + o * r) for _, i, o, r in SHAPES) / 1e9
    print(f"\nper-layer graph time {us:.1f} us | factor bytes/layer "
          f"{layer_bytes:.3f} GB | eff {layer_bytes / (us * 1e-6):.0f} GB/s")
    print(f"x36 layers -> per-token {us * 36 / 1000:.2f} ms -> "
          f"{1e6 / (us * 36):.1f} tok/s | all_shapes_correct={ok}")


if __name__ == "__main__":
    main()
