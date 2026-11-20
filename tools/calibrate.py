"""Gate-1 validation infrastructure: activation capture + quantization gates.

- capture: run a bf16 model over calibration text, save each nn.Linear's input
  activations (fp16, on disk). Experts share one input per MoE block -> stored
  once per parent block, many linears mapped to one file.
- act_cos(W_ref, W_hat, x): activation-space quality metric (the metric M3
  proved matters, vs weight-space rel err which does not).
- int4_channel_quant: per-output-channel int4 fake-quant (the gate-1 control:
  reproduce colibri-style int4 quality on one MLP block).
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "C:/Users/micro/Code/Projects/littlebit")
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))

import transformers.models.qwen3.modeling_qwen3  # noqa: F401,E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def calib_text(tok, path="C:/Users/micro/Code/Projects/micella/tools/calib.txt",
               n_tokens=2048):
    p = Path(path)
    if p.exists():
        ids = tok(p.read_text(encoding="utf-8"), return_tensors="pt").input_ids
    else:
        ids = tok("The theory of computation deals with the efficiency of algorithms. "
                  "Large language models compress statistical regularities of text into "
                  "weights. Quantization studies how few bits per weight preserve behavior. "
                  "Mixture of experts models route each token to specialized subnetworks. "
                  * 100, return_tensors="pt").input_ids
    return ids[:, :n_tokens]


@torch.no_grad()
def capture_linears(model, ids, cache_dir, max_linears=None):
    """Save fp16 inputs of every nn.Linear under models.<idx> modules."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    store = {}
    handles = []

    def hook(name):
        def fn(mod, inp, out):
            if name not in store:
                x = inp[0].detach()
                store[name] = x.reshape(-1, x.shape[-1]).to(torch.float16).cpu()
        return fn

    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and ".layers." in name:
            if max_linears and len(store) >= max_linears:
                break
            handles.append(mod.register_forward_hook(hook(name)))
    model(ids.cuda(), use_cache=False)
    for h in handles:
        h.remove()
    manifest = {}
    for name, x in store.items():
        fn = cache / (name.replace(".", "_") + ".pt")
        torch.save(x, fn)
        manifest[name] = {"file": fn.name, "shape": list(x.shape)}
    (cache / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"captured {len(manifest)} linear inputs -> {cache}")
    return manifest


def load_calib(cache_dir, name):
    man = json.loads((Path(cache_dir) / "manifest.json").read_text())
    x = torch.load(Path(cache_dir) / man[name]["file"], weights_only=True)
    return x.float().cuda()


@torch.no_grad()
def act_cos(W_ref, W_hat, x, chunk=8192):
    """Mean cosine between y_ref = x W_ref^T and y_hat = x W_hat^T per output
    row, averaged over rows and calibration tokens. x: (T, in) fp32."""
    W_ref = W_ref.float().cuda()
    W_hat = W_hat.float().cuda()
    num = torch.zeros(W_ref.shape[0], device="cuda")
    dref = torch.zeros_like(num)
    dhat = torch.zeros_like(num)
    for s in range(0, x.shape[0], chunk):
        xs = x[s:s + chunk]
        yr = xs @ W_ref.t()
        yh = xs @ W_hat.t()
        num += (yr * yh).sum(0)
        dref += (yr * yr).sum(0)
        dhat += (yh * yh).sum(0)
    cos = num / (dref.sqrt() * dhat.sqrt() + 1e-9)
    return cos.mean().item(), cos.min().item()


def int4_channel_quant(W):
    """Per-INPUT-channel (column) int4 symmetric fake-quant (colibri-style)."""
    s = W.abs().amax(dim=0, keepdim=True).clamp(min=1e-8) / 7.0
    q = torch.round(W / s).clamp(-8, 7)
    return q * s


def compress_block_int4(block):
    """Fake-quant every Linear in a module (in place, reversible via backup)."""
    backups = {}
    for n, m in block.named_modules():
        if isinstance(m, torch.nn.Linear):
            backups[n] = m.weight.data.clone()
            m.weight.data = int4_channel_quant(m.weight.data.float()).to(m.weight.dtype)
    return backups


@torch.no_grad()
def blockwise_cos(model, block_prefix, cache_dir):
    """Activation cos of every Linear inside one decoder block."""
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and name.startswith(block_prefix):
            W_ref = mod.weight.data.float()
            x = load_calib(cache_dir, name)
            out[name] = act_cos(W_ref, W_ref, x)  # placeholder: replaced by caller
    return out
