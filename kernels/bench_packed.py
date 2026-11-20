"""Bench 1-bit packed factors vs int8 factors, eager + full-layer CUDA graph.

The graph captures one layer's 7 matrices x 2 pairs (main+residual) = 28
factor-pair evaluations, i.e. one decode token's matmul work per layer.
"""
import sys
import time
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).parent))
import factored_gemm as fg  # noqa: E402

SHAPES = [("kv", 4096, 1024, 208), ("qo", 4096, 4096, 544),
          ("gate", 4096, 12288, 824), ("up", 4096, 12288, 824),
          ("down", 12288, 4096, 824)]  # gate/up listed once each + q,o,k,v
MATRICES = SHAPES + [("o", 4096, 4096, 544), ("v", 4096, 1024, 208)]


def build_state():
    state = []
    for name, i, o, r in MATRICES:
        x, V, U, v2, v1, u2, u1 = fg.make_inputs(i, o, r)
        h = torch.empty(r, dtype=torch.bfloat16, device="cuda")
        y = torch.empty(o, dtype=torch.bfloat16, device="cuda")
        state.append(dict(name=name, x=x, V=V, U=U, v2=v2, v1=v1, u1=u1,
                          s=v1 * u2, h=h, y=y, i=i, o=o, r=r,
                          VP=fg.pack_rows(V), UP=fg.pack_rows(U)))
    return state


def check_packed(state):
    st = state[0]
    fg.factored_forward_packed(st["x"], st["VP"], st["UP"], st["v2"], st["s"],
                               st["u1"], st["i"], st["o"], st["r"], st["h"], st["y"])
    href = ((st["x"].float() * st["v2"].float()) @ st["V"].float().t()) * st["s"].float()
    yref = (href @ st["U"].float().t()) * st["u1"].float()
    err = ((st["y"].float() - yref).abs().max() / yref.abs().max()).item()
    print(f"packed correctness ({st['name']}): rel err {err:.4f}")
    assert err < 0.02


def layer_graph(mode):
    """Capture one layer; mode in {int8, packed}. Returns (graph, state)."""
    state = build_state()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for st in state:
            if mode == "int8":
                fg.factored_forward_triton(st["x"], st["V"], st["U"], st["v2"],
                                           st["s"], st["u1"], st["h"], st["y"])
            else:
                fg.factored_forward_packed(st["x"], st["VP"], st["UP"], st["v2"],
                                           st["s"], st["u1"], st["i"], st["o"],
                                           st["r"], st["h"], st["y"])
    torch.cuda.synchronize()
    return g, state


def time_graph(g, iters=200):
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def main():
    state = build_state()
    check_packed(state)

    for mode in ("int8", "packed"):
        g, _ = layer_graph(mode)
        us = time_graph(g)
        tok_us = us * 36
        mb = sum(2 * (st["r"] * st["i"] + st["o"] * st["r"]) for st in state)
        mb /= 1e9 if mode == "int8" else 8e9
        print(f"{mode:7s}: layer {us:7.1f} us | token {tok_us/1000:7.2f} ms | "
              f"{1e6/tok_us:7.1f} tok/s implied | eff {mb/(us*1e-6):5.0f} GB/s")

    # quick num_warps sweep on the packed path for the biggest matrix
    st = [s for s in state if s["name"] == "gate"][0]
    best = None
    for nwa, nwb, bino in ((2, 2, 256), (4, 4, 512), (8, 8, 1024), (4, 8, 512)):
        def pair(nwa=nwa, nwb=nwb, bino=bino):
            fg._scale_v_packed_kernel[(triton.cdiv(st["r"], 32),)](
                st["x"], st["VP"], st["v2"], st["s"], st["h"], st["i"], st["r"],
                st["VP"].shape[1], BLOCK_IN=bino, BLOCK_OUT=32, num_warps=nwa)
            fg._scale_u_packed_kernel[(triton.cdiv(st["o"], 64),)](
                st["h"], st["UP"], st["u1"], st["y"], st["r"], st["o"],
                st["UP"].shape[1], BLOCK_R=1024, BLOCK_OUT=64, num_warps=nwb)
        for _ in range(10):
            pair()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            pair()
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / 200 * 1e6
        print(f"  warps A/B=({nwa},{nwb}) BLOCK_IN={bino}: gate pair {us:.1f} us")
        best = min(best or us, us)
    print(f"best gate pair: {best:.1f} us")


if __name__ == "__main__":
    main()
