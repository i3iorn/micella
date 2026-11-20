"""Fast-path patching for LittleBitLinear + graph-friendly decode state.

Replaces the stock forward (per-call casts, per-call scale products) with:
- decode (1 token): two Triton int8 kernels per factor pair, hoisted s_vu,
  preallocated h / residual-y buffers
- prefill (>1 token): stock torch math on pre-cast bf16 factor copies
  (cast once at patch time, not per call)
"""
import sys
import types
from pathlib import Path

import torch

REPO = Path("C:/Users/micro/Code/Projects/littlebit")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

import triton  # noqa: E402
import triton.language as tl  # noqa: E402
import quantization.utils.quant_util  # noqa: F401,E402  (must precede modules:
from quantization.modules.littlebit import LittleBitLinear  # noqa: E402  # breaks circular init)
from factored_gemm import (_scale_u_kernel, _scale_u_res_kernel,  # noqa: E402
                           _scale_v_kernel, rmsnorm_triton)


@triton.jit
def _lm_head_gemv_kernel(X, W, Y, VOCAB: tl.constexpr, HID: tl.constexpr,
                         BLOCK_V: tl.constexpr, BLOCK_K: tl.constexpr):
    """Y[v] = sum_k X[k]*W[v,k]; W bf16 (VOCAB, HID) read once, fp32 acc."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_V + tl.arange(0, BLOCK_V)
    vmask = rows < VOCAB
    acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
    for k in range(0, HID, BLOCK_K):
        cols = k + tl.arange(0, BLOCK_K)
        xv = tl.load(X + cols).to(tl.float32)
        wblk = tl.load(W + rows[:, None] * HID + cols[None, :],
                       mask=vmask[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(wblk * xv[None, :], axis=1)
    tl.store(Y + rows, acc.to(tl.bfloat16), mask=vmask)


class _FastLmHead:
    """Instance-patched lm_head: Triton GEMV into a reused logits buffer
    (cublas bf16 GEMV measured 39 GB/s here; kernel reads 1.24 GB once)."""

    def __call__(self, x):
        W = self.weight.data
        VOCAB, HID = W.shape
        if not hasattr(self, "_logits_buf"):
            self._logits_buf = torch.empty(VOCAB, dtype=torch.bfloat16,
                                           device=x.device)
        _lm_head_gemv_kernel[(triton.cdiv(VOCAB, 64),)](
            x.reshape(-1), W, self._logits_buf, VOCAB, HID,
            BLOCK_V=64, BLOCK_K=512, num_warps=4)
        return self._logits_buf.reshape(1, 1, VOCAB)


def _rope_table_forward(self, x, position_ids):
    pos = position_ids.reshape(-1)
    cos = self._cos_tab.index_select(0, pos).view(*position_ids.shape, -1)
    sin = self._sin_tab.index_select(0, pos).view(*position_ids.shape, -1)
    return cos, sin


def _rmsnorm_forward(self, hidden_states):
    """Decode (1 token): Triton fused RMSNorm into a static buffer.
    Prefill: stock torch math (fp32 accumulate differs slightly from the
    bf16-chain stock math, so prefill keeps the reference path)."""
    if hidden_states.shape[-2] == 1 and hidden_states.is_cuda:
        if not hasattr(self, "_out_buf"):
            self._out_buf = torch.empty(1, 1, self.weight.shape[0],
                                        dtype=hidden_states.dtype,
                                        device=hidden_states.device)
        rmsnorm_triton(hidden_states.reshape(1, -1), self.weight.data,
                       self.variance_epsilon, self._out_buf)
        return self._out_buf
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
    return Qwen3RMSNorm.forward(self, hidden_states)


def patch_decode_overheads(model, max_pos):
    """RoPE lookup tables (shared, one copy) + fast lm_head. Call AFTER the
    model is on its final device."""
    table_cos = table_sin = None
    n_rope = 0
    for mod in model.modules():
        if type(mod).__name__ == "Qwen3RotaryEmbedding":
            if table_cos is None:
                inv = mod.inv_freq.float()
                pos = torch.arange(max_pos, device=inv.device, dtype=torch.float32)
                freqs = pos[:, None] * inv[None, :]
                emb = torch.cat((freqs, freqs), dim=-1) * mod.attention_scaling
                table_cos = emb.cos().to(torch.bfloat16)
                table_sin = emb.sin().to(torch.bfloat16)
            mod._cos_tab = table_cos
            mod._sin_tab = table_sin
            mod.forward = types.MethodType(_rope_table_forward, mod)
            n_rope += 1
    lm_head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if lm_head is None and hasattr(model, "lm_head"):
        lm_head = model.lm_head
    if lm_head is not None:
        lm_head.forward = _FastLmHead.__call__.__get__(lm_head)
    n_norm = 0
    import os
    if os.environ.get("MICELLA_NORM", "1") == "1":
        for mod in model.modules():
            if type(mod).__name__ == "Qwen3RMSNorm":
                mod.forward = types.MethodType(_rmsnorm_forward, mod)
                n_norm += 1
    print(f"[overheads] {n_rope} rope table-ized; lm_head -> Triton GEMV; "
          f"{n_norm} RMSNorms -> fused Triton")
    return n_rope


@triton.jit
def _gqa_decode_kernel(Q, K, V, MASK, O, scale, KH_STRIDE,
                       GH: tl.constexpr, HD: tl.constexpr, MAXL: tl.constexpr,
                       BLOCK_L: tl.constexpr):
    """M=1 GQA decode attention, one program per q head, online softmax.
    Reads the FULL StaticCache K/V (beyond-L entries masked via additive mask)."""
    pid = tl.program_id(0)
    g = pid // GH
    offs_d = tl.arange(0, HD)
    q = tl.load(Q + pid * HD + offs_d).to(tl.float32)
    m_i = tl.full((1,), float("-inf"), tl.float32)
    l_i = tl.zeros((1,), dtype=tl.float32)
    acc = tl.zeros((HD,), dtype=tl.float32)
    for start in range(0, MAXL, BLOCK_L):
        js = start + tl.arange(0, BLOCK_L)
        jmask = js < MAXL
        kblk = tl.load(K + g * KH_STRIDE + js[:, None] * HD + offs_d[None, :],
                       mask=jmask[:, None], other=0.0).to(tl.float32)
        msk = tl.load(MASK + js, mask=jmask, other=0.0).to(tl.float32)
        sc = tl.sum(kblk * q[None, :], axis=1) * scale + msk
        sc = tl.where(jmask, sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, axis=0))
        p = tl.exp(sc - m_new)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        vblk = tl.load(V + g * KH_STRIDE + js[:, None] * HD + offs_d[None, :],
                       mask=jmask[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * vblk, axis=0)
        m_i = m_new
    tl.store(O + pid * HD + offs_d, (acc / l_i).to(tl.bfloat16))


class _BufHolder:
    pass


_DEFAULT_MODULE = _BufHolder()


def micella_attention(module, query, key, value, attention_mask=None,
                      scaling=None, dropout=0.0, **kwargs):
    """Dispatcher registered over 'sdpa' in ALL_ATTENTION_FUNCTIONS.
    q_len==1 (decode): custom M=1 GQA kernel. Else: stock eager attention."""
    if module is None:
        module = _DEFAULT_MODULE
    if query.shape[2] != 1 or not query.is_cuda or attention_mask is None:
        from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward
        return eager_attention_forward(module, query, key, value, attention_mask,
                                       scaling=scaling, dropout=dropout, **kwargs)
    q = query.reshape(query.shape[1], -1)
    n_heads, HD = q.shape
    kv_heads = key.shape[1]
    gh = n_heads // kv_heads
    if not hasattr(module, "_attn_out"):
        module._attn_out = torch.empty(n_heads * HD, dtype=query.dtype,
                                       device=query.device)
    _gqa_decode_kernel[(n_heads,)](
        q, key, value, attention_mask, module._attn_out,
        scaling if scaling is not None else HD ** -0.5,
        key.stride(1), GH=gh, HD=HD, MAXL=attention_mask.shape[-1],
        BLOCK_L=64, num_warps=4)
    return module._attn_out.view(1, n_heads, 1, HD), None


def patch_attention(model):
    """Route sdpa calls through the M=1 decode kernel (config stays 'sdpa' so
    HF's mask building is unchanged)."""
    import transformers.modeling_utils as mu
    mu.ALL_ATTENTION_FUNCTIONS["sdpa"] = micella_attention
    print("[attn] M=1 GQA decode kernel registered over sdpa")


def _pair_forward(mod, x, V, U, v2, u1, s_vu, h, y, res=None):
    R, IN = V.shape
    _scale_v_kernel[(triton.cdiv(R, 32),)](
        x, V, v2, s_vu, h, IN, R, BLOCK_IN=512, BLOCK_OUT=32, num_warps=4)
    if res is None:
        _scale_u_kernel[(triton.cdiv(U.shape[0], 64),)](
            h, U, u1, y, R, U.shape[0], BLOCK_R=1024, BLOCK_OUT=64, num_warps=4)
    else:
        _scale_u_res_kernel[(triton.cdiv(U.shape[0], 64),)](
            h, U, u1, y, res, R, U.shape[0], BLOCK_R=1024, BLOCK_OUT=64, num_warps=4)
    return y


def patch_model_fast(model):
    """Swap every LittleBitLinear to the kernel fast path (in place)."""
    n = 0
    for mod in model.modules():
        if type(mod) is not LittleBitLinear:
            continue
        mod.__class__ = _FastLittleBitLinear
        mod._fast_setup()
        n += 1
    return n


class _FastLittleBitLinear(LittleBitLinear):
    def _fast_setup(self):
        d = self.V.device
        self._s_vu = self.v1.data * self.u2.data
        self._h = torch.empty(self.split_dim, dtype=torch.bfloat16, device=d)
        self._y = torch.empty(self.out_features, dtype=torch.bfloat16, device=d)
        self._Vf = self.V.data  # int8, stay as-is
        self._Uf = self.U.data
        if self.residual:
            self._s_vu_R = self.v1_R.data * self.u2_R.data
            self._yR = torch.empty(self.out_features, dtype=torch.bfloat16, device=d)
        # prefill fallback casts int8->bf16 on the fly (once per prompt, not
        # per token); keeping persistent bf16 copies would double factor VRAM
        # NOTE: do NOT register aliases of parameter storages (_Vf, _v2f...)
        # as buffers - Module.to() moves params and buffers independently and
        # duplicates the storage (+3.8 GB here). Forward reads params directly.
        for n in ["_s_vu", "_h", "_y"]:
            t = getattr(self, n)
            delattr(self, n)
            self.register_buffer(n, t, persistent=False)
        if self.residual:
            for n in ["_s_vu_R", "_yR"]:
                t = getattr(self, n)
                delattr(self, n)
                self.register_buffer(n, t, persistent=False)

    def forward(self, x):
        if x.shape[-2] == 1 and x.dim() >= 2 and x.is_cuda:
            xf = x.reshape(-1)
            y = _pair_forward(self, xf, self.V.data, self.U.data, self.v2.data,
                              self.u1.data, self._s_vu, self._h, self._y)
            if self.residual:
                # residual pair accumulates y inside its U-kernel epilogue
                _pair_forward(self, xf, self.V_R.data, self.U_R.data,
                              self.v2_R.data, self.u1_R.data,
                              self._s_vu_R, self._h, self._yR, res=self._y)
                y = self._yR
            if self.bias is not None:
                y = y + self.bias
            return y.reshape(*x.shape[:-1], self.out_features)
        # prefill path (M>1): stock math, casting int8 factors on the fly
        Vb = self.V.data.to(x.dtype)
        h = (x * self.v2.data) @ Vb.t() * self._s_vu
        y = (h @ self.U.data.to(x.dtype).t()) * self.u1.data
        if self.residual:
            h2 = (x * self.v2_R.data) @ self.V_R.data.to(x.dtype).t() * self._s_vu_R
            y = y + (h2 @ self.U_R.data.to(x.dtype).t()) * self.u1_R.data
        if self.bias is not None:
            y = y + self.bias
        return y
