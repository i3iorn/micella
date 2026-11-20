"""Fused LittleBit factored-GEMV kernels (M2).

Op per LittleBitLinear factor pair:
    h = ((x * v2) @ V^T) * (v1 * u2)      # V: (r, in)  int8 +-1
    y = (h @ U^T) * u1                    # U: (out, r) int8 +-1
Residual variant adds a second pair; handled by the caller (two launches).

Decode path (M=1): two Triton GEMV kernels, factors loaded once per token in
int8, dequantized in registers - no materialized cast traffic.

Design notes:
- Kernel A: one program per BLOCK_OUT rows of V; loop over `in` in BLOCK_IN
  chunks; accumulate fp32; scale by (v1*u2) at the end.
- Kernel B: one program per BLOCK_OUT rows of U; reduction over r fits one or
  two chunks; scale by u1 at the end.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _scale_v_kernel(X, V, V2, S, H,
                    IN: tl.constexpr, R: tl.constexpr,
                    BLOCK_IN: tl.constexpr, BLOCK_OUT: tl.constexpr):
    """H[r] = sum_i X[i]*V2[i]*V[r,i] * S[r]   (X, V2: bf16/in; V: int8; S: bf16/r)"""
    pid = tl.program_id(0)
    rows = pid * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    rmask = rows < R
    acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)
    for start in range(0, IN, BLOCK_IN):
        cols = start + tl.arange(0, BLOCK_IN)
        xv = tl.load(X + cols).to(tl.float32) * tl.load(V2 + cols).to(tl.float32)
        vblk = tl.load(V + rows[:, None] * IN + cols[None, :], mask=rmask[:, None],
                       other=0.0).to(tl.float32)
        acc += tl.sum(vblk * xv[None, :], axis=1)
    s = tl.load(S + rows, mask=rmask, other=0.0).to(tl.float32)
    tl.store(H + rows, acc * s, mask=rmask)


@triton.jit
def _scale_u_kernel(H, U, U1, Y,
                    R: tl.constexpr, OUT: tl.constexpr,
                    BLOCK_R: tl.constexpr, BLOCK_OUT: tl.constexpr):
    """Y[o] = (sum_r H[r]*U[o,r]) * U1[o]   (H, U1: bf16; U: int8)"""
    pid = tl.program_id(0)
    rows = pid * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)
    for start in range(0, R, BLOCK_R):
        cols = start + tl.arange(0, BLOCK_R)
        h = tl.load(H + cols, mask=cols < R, other=0.0).to(tl.float32)
        ublk = tl.load(U + rows[:, None] * R + cols[None, :],
                       mask=(rows[:, None] < OUT) & (cols[None, :] < R),
                       other=0.0).to(tl.float32)
        acc += tl.sum(ublk * h[None, :], axis=1)
    u1 = tl.load(U1 + rows, mask=rows < OUT, other=0.0).to(tl.float32)
    tl.store(Y + rows, acc * u1, mask=rows < OUT)


@triton.jit
def _scale_v_packed_kernel(X, VP, V2, S, H,
                           IN: tl.constexpr, R: tl.constexpr,
                           WPR: tl.constexpr, BLOCK_IN: tl.constexpr,
                           BLOCK_OUT: tl.constexpr):
    """V row-packed 1-bit (lbit: +1->0, -1->1), WPR int32 words/row.
    H[r] = sum_i X[i]*V2[i]*V[r,i] * S[r]. BLOCK_IN must be a multiple of 32."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    rmask = rows < R
    acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)
    bits = tl.arange(0, 32)
    for start in range(0, IN, BLOCK_IN):
        cols = start + tl.arange(0, BLOCK_IN)
        words = start // 32 + tl.arange(0, BLOCK_IN // 32)
        w = tl.load(VP + rows[:, None] * WPR + words[None, :],
                    mask=rmask[:, None], other=0).to(tl.int32)
        vblk = 1 - 2 * ((w[:, :, None] >> bits[None, None, :]) & 1)
        vblk = tl.reshape(vblk, (BLOCK_OUT, BLOCK_IN)).to(tl.float32)
        xv = tl.load(X + cols, mask=cols < IN, other=0.0).to(tl.float32) * \
             tl.load(V2 + cols, mask=cols < IN, other=0.0).to(tl.float32)
        acc += tl.sum(vblk * xv[None, :], axis=1)
    s = tl.load(S + rows, mask=rmask, other=0.0).to(tl.float32)
    tl.store(H + rows, acc * s, mask=rmask)


@triton.jit
def _scale_u_packed_kernel(H, UP, U1, Y,
                           R: tl.constexpr, OUT: tl.constexpr,
                           WPR: tl.constexpr, BLOCK_R: tl.constexpr,
                           BLOCK_OUT: tl.constexpr):
    """U row-packed 1-bit. Y[o] = (sum_r H[r]*U[o,r]) * U1[o].
    BLOCK_R multiple of 32; pad bits multiply H=0 so they are harmless."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    omask = rows < OUT
    acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)
    bits = tl.arange(0, 32)
    for start in range(0, R, BLOCK_R):
        cols = start + tl.arange(0, BLOCK_R)
        words = start // 32 + tl.arange(0, BLOCK_R // 32)
        w = tl.load(UP + rows[:, None] * WPR + words[None, :],
                    mask=omask[:, None], other=0).to(tl.int32)
        ublk = 1 - 2 * ((w[:, :, None] >> bits[None, None, :]) & 1)
        ublk = tl.reshape(ublk, (BLOCK_OUT, BLOCK_R)).to(tl.float32)
        h = tl.load(H + cols, mask=cols < R, other=0.0).to(tl.float32)
        acc += tl.sum(ublk * h[None, :], axis=1)
    u1 = tl.load(U1 + rows, mask=omask, other=0.0).to(tl.float32)
    tl.store(Y + rows, acc * u1, mask=omask)


@triton.jit
def _scale_u_packed2_kernel(H, UP, U1, Y,
                            R: tl.constexpr, OUT: tl.constexpr,
                            WPR: tl.constexpr, BLOCK_R: tl.constexpr,
                            BLOCK_OUT: tl.constexpr):
    """2D per-bit loop variant of _scale_u_packed_kernel: no 3D intermediate.
    acc[o] = sum_r H[r]*U[o,r]; U row-packed 1-bit."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    omask = rows < OUT
    acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)
    for start in range(0, R, BLOCK_R):
        w = tl.load(UP + rows[:, None] * WPR + (start // 32 + tl.arange(0, BLOCK_R // 32))[None, :],
                    mask=omask[:, None], other=0)
        for b in tl.static_range(32):
            hb = tl.load(H + start + tl.arange(0, BLOCK_R // 32) * 32 + b,
                         mask=(start + tl.arange(0, BLOCK_R // 32) * 32 + b) < R,
                         other=0.0).to(tl.float32)
            sign = 1.0 - 2.0 * ((w >> b) & 1).to(tl.float32)
            acc += tl.sum(sign * hb[None, :], axis=1)
    u1 = tl.load(U1 + rows, mask=omask, other=0.0).to(tl.float32)
    tl.store(Y + rows, acc * u1, mask=omask)


@triton.jit
def _scale_v_packed2_kernel(X, VP, V2, S, H,
                            IN: tl.constexpr, R: tl.constexpr,
                            WPR: tl.constexpr, BLOCK_IN: tl.constexpr,
                            BLOCK_OUT: tl.constexpr):
    """2D per-bit loop variant of _scale_v_packed_kernel.
    H[r] = sum_i X[i]*V2[i]*V[r,i] * S[r]; V row-packed 1-bit."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    rmask = rows < R
    acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)
    for start in range(0, IN, BLOCK_IN):
        w = tl.load(VP + rows[:, None] * WPR + (start // 32 + tl.arange(0, BLOCK_IN // 32))[None, :],
                    mask=rmask[:, None], other=0)
        for b in tl.static_range(32):
            xv = tl.load(X + start + tl.arange(0, BLOCK_IN // 32) * 32 + b,
                         mask=(start + tl.arange(0, BLOCK_IN // 32) * 32 + b) < IN,
                         other=0.0).to(tl.float32) * \
                 tl.load(V2 + start + tl.arange(0, BLOCK_IN // 32) * 32 + b,
                         mask=(start + tl.arange(0, BLOCK_IN // 32) * 32 + b) < IN,
                         other=0.0).to(tl.float32)
            sign = 1.0 - 2.0 * ((w >> b) & 1).to(tl.float32)
            acc += tl.sum(sign * xv[None, :], axis=1)
    s = tl.load(S + rows, mask=rmask, other=0.0).to(tl.float32)
    tl.store(H + rows, acc * s, mask=rmask)


def pack_rows(t: torch.Tensor) -> torch.Tensor:
    """(n, d) int8 +-1 -> (n, ceil(d/32)) int32, +1->bit0, -1->bit1, pad=+1."""
    n, d = t.shape
    wpr = (d + 31) // 32
    padded = torch.ones((n, wpr * 32), dtype=torch.int8, device=t.device)
    padded[:, :d] = t
    bits = ((1 - padded) // 2).to(torch.int64).reshape(n, wpr, 32)
    wts = 2 ** torch.arange(32, dtype=torch.int64, device=t.device)
    return (bits * wts).sum(dim=2).to(torch.int32).contiguous()  # mod 2^32 wrap is intended


def factored_forward_packed(x, VP, UP, v2, s_vu, u1, in_f, out_f, r, h, y):
    """Same math as factored_forward_triton but factors are 1-bit packed
    (2D per-bit-loop kernels - the 3D broadcast variant was 18x slower)."""
    wpr_v = VP.shape[1]
    wpr_u = UP.shape[1]
    _scale_v_packed2_kernel[(triton.cdiv(r, 32),)](
        x, VP, v2, s_vu, h, in_f, r, wpr_v, BLOCK_IN=512, BLOCK_OUT=32, num_warps=4)
    # U kernel wants 8 warps: (4,4) measured 20x slower than (4,8) on gate_up
    _scale_u_packed2_kernel[(triton.cdiv(out_f, 64),)](
        h, UP, u1, y, r, out_f, wpr_u, BLOCK_R=1024, BLOCK_OUT=64, num_warps=8)
    return y


@triton.jit
def _scale_u_res_kernel(H, U, U1, Y, RES,
                        R: tl.constexpr, OUT: tl.constexpr,
                        BLOCK_R: tl.constexpr, BLOCK_OUT: tl.constexpr):
    """U-kernel with fused residual: Y[o] = (sum_r H[r]*U[o,r])*U1[o] + RES[o].
    Same semantics as _scale_u_kernel plus one free add (epilogue fusion)."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    omask = rows < OUT
    acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)
    for start in range(0, R, BLOCK_R):
        cols = start + tl.arange(0, BLOCK_R)
        h = tl.load(H + cols, mask=cols < R, other=0.0).to(tl.float32)
        ublk = tl.load(U + rows[:, None] * R + cols[None, :],
                       mask=(rows[:, None] < OUT) & (cols[None, :] < R),
                       other=0.0).to(tl.float32)
        acc += tl.sum(ublk * h[None, :], axis=1)
    u1 = tl.load(U1 + rows, mask=omask, other=0.0).to(tl.float32)
    res = tl.load(RES + rows, mask=omask, other=0.0).to(tl.float32)
    tl.store(Y + rows, acc * u1 + res, mask=omask)


@triton.jit
def _rmsnorm_kernel(X, W, Y, N: tl.constexpr, EPS: tl.constexpr,
                    BLOCK: tl.constexpr):
    """Y[row] = X[row] * rsqrt(mean(X[row]^2) + eps) * W   (fp32 accumulation)."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    acc = 0.0
    for start in range(0, N, BLOCK):
        cols = start + offs
        x = tl.load(X + row * N + cols, mask=cols < N, other=0.0).to(tl.float32)
        acc += tl.sum(x * x)
    rms = tl.math.rsqrt(acc / N + EPS)
    for start in range(0, N, BLOCK):
        cols = start + offs
        x = tl.load(X + row * N + cols, mask=cols < N, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=cols < N, other=0.0).to(tl.float32)
        tl.store(Y + row * N + cols, (x * rms * w).to(tl.bfloat16), mask=cols < N)


def rmsnorm_triton(x2d, weight, eps, out):
    """x2d: (rows, N) contiguous bf16; out: preallocated (rows, N)."""
    rows, n = x2d.shape
    _rmsnorm_kernel[(rows,)](x2d, weight, out, n, eps,
                             BLOCK=1024, num_warps=4)
    return out


def factored_forward_triton(x, V, U, v2, s_vu, u1, h=None, y=None):
    """x: (in,) bf16; V int8 (r,in); U int8 (out,r); s_vu = v1*u2 precomputed.

    Same-stream Triton launches are ordered; no sync needed between A and B.
    The caller should hoist s_vu (constant per layer) and reuse h/y buffers
    to keep per-call overhead at two launches.
    """
    R, IN = V.shape
    OUT = U.shape[0]
    if h is None:
        h = torch.empty(R, dtype=torch.bfloat16, device=x.device)
    if y is None:
        y = torch.empty(OUT, dtype=torch.bfloat16, device=x.device)
    _scale_v_kernel[(triton.cdiv(R, 32),)](
        x, V, v2, s_vu, h, IN, R, BLOCK_IN=512, BLOCK_OUT=32, num_warps=4)
    _scale_u_kernel[(triton.cdiv(OUT, 64),)](
        h, U, u1, y, R, OUT, BLOCK_R=1024, BLOCK_OUT=64, num_warps=4)
    return y


def factored_forward_torch(x, V, U, v2, v1, u2, u1):
    """Cast-free bf16 reference: the floor for any naive-PyTorch path."""
    h = (x * v2) @ V.t() * (v1 * u2)
    return (h @ U.t()) * u1


def make_inputs(in_f, out_f, r, device="cuda", seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(in_f, generator=g).to(device=device, dtype=torch.bfloat16)
    V = (torch.randint(0, 2, (r, in_f), generator=g) * 2 - 1).to(device=device, dtype=torch.int8)
    U = (torch.randint(0, 2, (out_f, r), generator=g) * 2 - 1).to(device=device, dtype=torch.int8)
    v2 = torch.randn(in_f, generator=g).to(device=device, dtype=torch.bfloat16)
    v1 = torch.randn(r, generator=g).to(device=device, dtype=torch.bfloat16)
    u2 = torch.randn(r, generator=g).to(device=device, dtype=torch.bfloat16)
    u1 = torch.randn(out_f, generator=g).to(device=device, dtype=torch.bfloat16)
    return x, V, U, v2, v1, u2, u1


def check_correctness(in_f, out_f, r, device="cuda"):
    x, V, U, v2, v1, u2, u1 = make_inputs(in_f, out_f, r, device)
    y_triton = factored_forward_triton(x, V, U, v2, v1 * u2, u1)
    y_ref = factored_forward_torch(x, V.float(), U.float(),
                                   v2.float(), v1.float(), u2.float(), u1.float())
    rel = (y_triton.float() - y_ref).abs().max() / y_ref.abs().max()
    return rel.item()
