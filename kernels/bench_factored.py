"""M2 microbenchmark: Triton int8 factored GEMV vs cast-free bf16 torch.

Shapes are the real Qwen3-8B@0.55bpw layer shapes (main factor pair only;
residual doubles the launches but not the per-element cost).
Reports us/call, effective GB/s of factor traffic, and implied tok/s if every
matmul of the model ran at the measured rate (36 layers x 7 matrices).

Usage: python bench_factored.py
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from factored_gemm import check_correctness, factored_forward_torch  # noqa: E402
from factored_gemm import factored_forward_triton, make_inputs  # noqa: E402

# (name, in, out, split) - actual checkpoint values
SHAPES = [
    ("kv_proj", 4096, 1024, 208),
    ("qo_proj", 4096, 4096, 544),
    ("gate_up", 4096, 12288, 824),
    ("down", 12288, 4096, 824),
]
LAYERS, MATRICES_PER_LAYER = 36, 7


def bench(fn, args, iters=200, warmup=20):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us


def factor_bytes(in_f, out_f, r):
    return (r * in_f + out_f * r)  # int8, one factor pair


def main():
    results = []
    print("correctness (max rel err vs fp32 torch ref):")
    for name, i, o, r in SHAPES:
        rel = check_correctness(i, o, r)  # wrapper handles its own s_vu
        print(f"  {name:8s} {rel:.4f}")
        assert rel < 0.02, f"{name} wrong by {rel}"

    per_call_us = {"triton": 0.0, "torch_bf16": 0.0}
    for name, i, o, r in SHAPES:
        x, V, U, v2, v1, u2, u1 = make_inputs(i, o, r)
        s_vu = (v1 * u2)  # hoisted constant, as the real runtime would
        h = torch.empty(r, dtype=torch.bfloat16, device="cuda")
        y = torch.empty(o, dtype=torch.bfloat16, device="cuda")
        us_t = bench(factored_forward_triton, (x, V, U, v2, s_vu, u1, h, y))
        us_f = bench(factored_forward_torch, (x, V.float(), U.float(),
                                              v2.float(), v1.float(),
                                              u2.float(), u1.float()))
        fb = factor_bytes(i, o, r)
        gbps = fb / (us_t * 1e-6) / 1e9
        # 2 kernels per pair (A+B); residual = second pair, same cost
        per_call_us["triton"] += 2 * us_t
        per_call_us["torch_bf16"] += 2 * us_f
        results.append({"shape": name, "in": i, "out": o, "r": r,
                        "triton_us": round(us_t, 1), "torch_bf16_us": round(us_f, 1),
                        "speedup": round(us_f / us_t, 2),
                        "factor_GB_per_token_per_pair": round(fb / 1e9, 3),
                        "effective_GB_s": round(gbps, 1)})
        print(f"{name:8s} triton {us_t:7.1f} us | torch-bf16 {us_f:7.1f} us | "
              f"speedup {us_f / us_t:5.2f}x | {gbps:6.1f} GB/s")

    # --- CUDA graph: capture one token's full kernel sequence (7 matrices x
    # 2 factor pairs x 2 kernels = 28 launches) and time graph.replay() ---
    state = []
    for name, i, o, r in SHAPES:
        x, V, U, v2, v1, u2, u1 = make_inputs(i, o, r)
        s_vu = v1 * u2
        h = torch.empty(r, dtype=torch.bfloat16, device="cuda")
        y = torch.empty(o, dtype=torch.bfloat16, device="cuda")
        state.append((x, V, U, v2, v1, s_vu, u1, h, y, o, r))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for x, V, U, v2, v1, s_vu, u1, h, y, o, r in state:
            factored_forward_triton(x, V, U, v2, s_vu, u1, h, y)
    torch.cuda.synchronize()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        graph.replay()
    torch.cuda.synchronize()
    tok_us = (time.perf_counter() - t0) / 100 * 1e6
    # correctness of a replayed graph
    x, V, U, v2, v1, s_vu, u1, h, y, o, r = state[0]
    ref = factored_forward_torch(x, V.float(), U.float(), v2.float(),
                                 v1.float(), (s_vu / v1).float(), u1.float())
    gerr = ((y.float() - ref).abs().max() / ref.abs().max()).item()
    results.append({"summary": "cuda_graph", "per_token_us": round(tok_us, 1),
                    "implied_tok_s": round(1e6 / tok_us, 1), "replay_rel_err": gerr})
    print(f"cuda_graph : per-token {tok_us/1000:8.2f} ms -> implied "
          f"{1e6/tok_us:7.1f} tok/s (28 launches/graph, rel err {gerr:.4f})")

    for key, total in per_call_us.items():
        per_tok = total * LAYERS * MATRICES_PER_LAYER / len(SHAPES) / 2
        results.append({"summary": key, "per_layer_us": round(
            total * MATRICES_PER_LAYER / len(SHAPES), 1),
            "per_token_us": round(per_tok, 1),
            "implied_tok_s": round(1e6 / per_tok, 1)})
        print(f"{key:12s}: per-token {per_tok/1000:8.2f} ms -> "
              f"implied {1e6/per_tok:7.1f} tok/s (36 layers, decode)")

    out = Path(__file__).parent / "results" / "m2_microbench.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
