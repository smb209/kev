"""Run kev.train with transformers' Qwen3.5 causal conv routed to flash-linear-attention's Triton kernel.

Why: with the `causal-conv1d` package absent (no aarch64 wheel; nothing in the Spark images ships it), transformers'
Qwen3_5GatedDeltaNet falls back to a depthwise torch F.conv1d on every DeltaNet layer, forward and backward. fla's
`causal_conv1d` (already installed, pinned 0.5.2, the kernel kev.fused_qwen35 serves with) does the same arithmetic
in Triton with its own backward. Everything else in kev.train is unchanged; arguments pass through.

    python spark/train_fla_conv.py <kev.train arguments>
    python spark/train_fla_conv.py --check        # numerical check vs the torch fallback (forward + backward)
"""
import runpy, sys

import torch
from fla.modules.conv import causal_conv1d
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

reference_conv = qwen.causal_conv1d_fn


def fla_causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
    """transformers' signature: hidden_states [B, D, T], weight [D, W] -> [B, D, T]; fla works on [B, T, D]."""
    out, _ = causal_conv1d(hidden_states.transpose(1, 2).contiguous(), weight, bias, activation=activation)
    return out.transpose(1, 2)


def check():
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.bfloat16):
        x = torch.randn(2, 1536, 777, device="cuda", dtype=dtype, requires_grad=True)
        w = (torch.randn(1536, 4, device="cuda", dtype=dtype) * 0.3).requires_grad_()
        outs = []
        for fn in (reference_conv, fla_causal_conv1d_fn):
            x.grad = w.grad = None
            y = fn(x, w, None, activation="silu")
            (y.float() * torch.linspace(-1, 1, y.shape[-1], device="cuda")).sum().backward()
            outs.append((y.detach().float(), x.grad.float().clone(), w.grad.float().clone()))
        for name, a, b in zip(("out", "dx", "dw"), *outs):
            rel = ((a - b).norm() / a.norm()).item()
            print(f"{dtype} {name}: rel err {rel:.2e}")
            assert rel < (1e-4 if dtype == torch.float32 else 2e-2), name


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        check(); sys.exit(0)
    qwen.causal_conv1d_fn = fla_causal_conv1d_fn   # the forward looks the name up at call time
    print("train_fla_conv: transformers causal_conv1d_fn -> fla.modules.conv.causal_conv1d", flush=True)
    sys.argv = ["kev.train", *sys.argv[1:]]
    runpy.run_module("kev.train", run_name="__main__", alter_sys=True)
