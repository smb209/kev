"""Post-training FP8 / NVFP4 quantization of a loaded Kev model, for serving and benchmarking on a DGX Spark (GB10, sm_121).

Weights are quantized once at load from the official bf16 checkpoint; activations are quantized per call on the device (no
host sync, so the passes stay capturable as CUDA graphs). FP8 activations take a per-token scale (row-independent). NVFP4
activations take a per-layer static global scale calibrated once on the calibration partition of decision-v7 (--act-scale
static, the default; a question's answer then does not depend on what shares its forward pass), or the old per-call
dynamic one from the amax of the whole flattened batch (--act-scale dynamic, kept for comparison). Nothing in kev/ is edited: the CLI wraps
`kev.checkpoint.Checkpoint._load_torch` and then runs kev.serve's or kev.benchmark's own main with the remaining argv.

    python spark/kev_quant.py --scheme nvfp4 serve --run jaredpalmer/kev-4b --host 0.0.0.0 --port 8019
    python spark/kev_quant.py --scheme fp8 benchmark --run jaredpalmer/kev-0.8b --suite evals/external/semif-v1 --out runs/spark/q-...
    python spark/kev_quant.py --scheme nvfp4 isolation --run jaredpalmer/kev-4b
    python spark/kev_quant.py layercheck --run jaredpalmer/kev-0.8b --suite evals/external/semif-v1

Schemes (decoder-layer projections only; embeddings, norms, the DeltaNet conv / gates and kev's PointerHead are untouched):
  bf16       nothing quantized: the same load path (bf16 activations, same fused / graph settings), the same-environment baseline
  fp8        every eligible projection Fp8Linear (e4m3 weight, per-output-channel scale; e4m3 activation, per-token scale)
  nvfp4      every eligible projection Nvfp4Linear (FlashInfer fp4_quantize + mm_fp4: e2m1 values, e4m3 scale per 16, fp32 global)
  nvfp4-mlp  MLP gate / up / down NVFP4, attention and DeltaNet projections FP8

Eligibility: a projection with either dimension under MIN_DIM (1024) stays bf16 (DeltaNet's per-head a / b gate projections;
0.8B's attention k / v), as does one whose shape a kernel cannot take (logged with the reason). kev.fused_qwen35 concatenates
projections that share an input (DeltaNet q/k/v + z + b + a, attention q + k + v, MLP gate + up) into one weight; the rule is
applied per original projection, so a fused weight with small members becomes a SplitLinear: the leading large members
quantized, the small tail rows one bf16 GEMM, the outputs concatenated.

Load path under every scheme (including bf16): the backbone loads in bf16 (LoadOptions.dtype is forced; kev.benchmark's
LocalPredictor would otherwise load the small checkpoints in fp32 for its exact path), the LoRA is merged (fp32 math, one
rounding), kev.fused_qwen35 rewrites the layers when fused is on, then the projections are quantized layer by layer (each bf16
weight freed as soon as its replacement exists, so the peak stays the bf16 model), and only then are CUDA graphs attached
(kev.cuda_graphs captures lazily, after this). Fused and graphs default to what the entry point would do (kev.serve: both on
for CUDA; kev.benchmark: both off, as LocalPredictor's reference path); --fused / --graphs override.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import types
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the repo root, for `kev` when run as a script

SCHEMES = ("bf16", "fp8", "nvfp4", "nvfp4-mlp")
MIN_DIM = 1024          # either dimension under this stays bf16
FP8_MAX = 448.0         # e4m3 largest normal
FP4_MAX = 6.0           # e2m1 largest
CALIB_SUITE = "evals/v7/decision-v7"   # its calibration partition drives the activation calibration; never development / test
NVFP4_BLOCK = 16
MLP_ROLES = ("gate_up", "gate_proj", "up_proj", "down_proj")


# --- activation quantization helpers (device-side, no host sync) ---------------------------------------------------------

def _fp8_quant_rows(x2):
    """bf16 [M, K] -> (e4m3 [M, K], fp32 scale [M, 1]) with a dynamic per-token scale amax / 448."""
    amax = torch.linalg.vector_norm(x2, float("inf"), dim=-1, keepdim=True).float().clamp_(min=1e-12)
    scale = amax / FP8_MAX
    return (x2 / scale).to(torch.float8_e4m3fn), scale


_fp8_quant = _fp8_quant_rows   # replaced by a torch.compile'd version with --act-quant compile


def set_act_quant(mode):
    """'torch': eager (4 kernels: amax, divide, two casts); 'compile': torch.compile fuses them into ~2 Triton kernels."""
    global _fp8_quant
    _fp8_quant = torch.compile(_fp8_quant_rows, dynamic=True, fullgraph=True) if mode == "compile" else _fp8_quant_rows


class Fp8Linear(nn.Module):
    """y = x W^T (+ b): W e4m3 with a per-output-channel scale (amax / 448), x e4m3 with a dynamic per-token scale,
    torch._scaled_mm rowwise (verified on sm_121, torch 2.13), bf16 out. Any leading dims."""
    precision = "fp8"

    def __init__(self, weight, bias=None):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        w = weight.detach().float()
        scale = w.abs().amax(1, keepdim=True).clamp_(min=1e-12) / FP8_MAX       # [N, 1]
        self.register_buffer("qweight", (w / scale).to(torch.float8_e4m3fn))   # [N, K] row-major = [K, N] column-major as .t()
        self.register_buffer("wscale", scale.t().contiguous())                 # [1, N]
        self.register_buffer("bias", None if bias is None else bias.detach().to(torch.bfloat16))
        del w

    def forward(self, x):
        shape = x.shape
        xq, xs = _fp8_quant(x.reshape(-1, self.in_features))
        y = torch._scaled_mm(xq, self.qweight.t(), scale_a=xs, scale_b=self.wscale, out_dtype=torch.bfloat16)
        if self.bias is not None: y = y + self.bias
        return y.view(*shape[:-1], self.out_features)

    @staticmethod
    def supports(n, k):
        return n % 16 == 0 and k % 16 == 0, "fp8: _scaled_mm needs N and K multiples of 16"


class Nvfp4Linear(nn.Module):
    """y = x W^T (+ b) in NVFP4: W quantized once (fp4_quantize, global scale 448*6 / amax(W), swizzled e4m3 scales per 16),
    x quantized per call with a global scale 448*6 / amax, FlashInfer mm_fp4 (backend 'cutlass' by default; 'trtllm' does
    not support sm_121), bf16 out. Any leading dims. act_mode:
      'static'     amax fixed by calibrate_model (times a margin), a device buffer: a row's result does not depend on the
                   other rows of the pass, no host sync, graph capturable
      'calibrate'  dynamic scale for the output, and act_amax = running max of amax(|x|) over every input
      'dynamic'    amax over the whole flattened batch, padding rows included, on every call (rows interact)"""
    precision = "nvfp4"
    backend = "cutlass"

    def __init__(self, weight, bias=None):
        super().__init__()
        from flashinfer import fp4_quantize
        self.out_features, self.in_features = weight.shape
        w = weight.detach().to(torch.bfloat16).contiguous()
        gw = (FP8_MAX * FP4_MAX / w.float().abs().amax().clamp(min=1e-12)).reshape(1)
        q, sf = fp4_quantize(w, gw, NVFP4_BLOCK, False, True)
        self.register_buffer("qweight", q)        # [N, K/2] uint8 (two e2m1 per byte)
        self.register_buffer("wsf", sf)           # swizzled e4m3 block scales
        self.register_buffer("wgs", gw.float())   # [1] fp32 global scale
        self.register_buffer("bias", None if bias is None else bias.detach().to(torch.bfloat16))
        self.act_mode = "dynamic"
        self.register_buffer("act_amax", torch.zeros(1, device=w.device))   # calibration running max
        self.register_buffer("ga", torch.ones(1, device=w.device))          # static activation global scale

    def forward(self, x):
        from flashinfer import fp4_quantize, mm_fp4
        shape = x.shape
        x2 = x.reshape(-1, self.in_features)
        if x2.dtype != torch.bfloat16: x2 = x2.to(torch.bfloat16)
        x2 = x2.contiguous()
        if self.act_mode == "static":
            ga = self.ga
        else:
            amax = torch.linalg.vector_norm(x2, float("inf")).float().reshape(1)
            if self.act_mode == "calibrate": torch.maximum(self.act_amax, amax, out=self.act_amax)   # device op, no sync
            ga = (FP8_MAX * FP4_MAX) / amax.clamp(min=1e-12)
        aq, asf = fp4_quantize(x2, ga, NVFP4_BLOCK, False, True)
        y = mm_fp4(aq, self.qweight.T, asf, self.wsf.T, 1.0 / (ga * self.wgs), torch.bfloat16, backend=self.backend)
        if self.bias is not None: y = y + self.bias
        return y.view(*shape[:-1], self.out_features)

    @staticmethod
    def supports(n, k):
        return n % 32 == 0 and k % 32 == 0, "nvfp4: mm_fp4 needs N and K multiples of 32"


class SplitLinear(nn.Module):
    """A fused projection whose leading rows are quantized and whose trailing small members stay bf16: two GEMMs, outputs
    concatenated in the original row order (kev.fused_qwen35 splits the result as before)."""

    def __init__(self, head, tail_weight):
        super().__init__()
        self.head = head
        self.register_buffer("tail", tail_weight.detach().to(torch.bfloat16).clone())   # a copy: a view would keep the whole fused bf16 weight alive
        self.in_features, self.out_features = head.in_features, head.out_features + tail_weight.shape[0]
        self.precision = head.precision

    def forward(self, x):
        return torch.cat([self.head(x), F.linear(x, self.tail)], -1)


QUANT = {"fp8": Fp8Linear, "nvfp4": Nvfp4Linear}


# --- kev.fused_qwen35 interplay ------------------------------------------------------------------------------------------

class _Functional:
    """Stand-in for torch.nn.functional inside kev.fused_qwen35: its fused forwards call F.linear(x, self.in_proj | self.qkv
    | self.gate_up) on plain weight tensors; when such an attribute is a quantized module instead, call it. Everything else
    is torch.nn.functional. This keeps kev's fused forwards (and their rounding) as written, without copying them."""

    def __getattr__(self, name):
        return getattr(F, name)

    @staticmethod
    def linear(x, weight, bias=None):
        return weight(x) if isinstance(weight, nn.Module) else F.linear(x, weight, bias)


def _install_fused_shim():
    import kev.fused_qwen35 as fq
    if not isinstance(fq.F, _Functional): fq.F = _Functional()


# --- quantize_model -------------------------------------------------------------------------------------------------------

def _targets(lm):
    """Every projection of the decoder layers -> (qualified name, owner module, attribute, role, members). members is None
    for an nn.Linear; for a fused weight tensor (kev.fused_qwen35) the [(original name, rows)] it concatenates."""
    for i, layer in enumerate(lm.layers):
        p = f"layers.{i}"
        if layer.block_type == "linear_attention":
            m = layer.linear_attn
            if isinstance(getattr(m, "in_proj", None), (torch.Tensor, nn.Module)) and not hasattr(m, "in_proj_qkv"):
                yield f"{p}.linear_attn.in_proj", m, "in_proj", "in_proj", list(zip(("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"), m.splits))
            else:
                for n in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"): yield f"{p}.linear_attn.{n}", m, n, n, None
            yield f"{p}.linear_attn.out_proj", m, "out_proj", "out_proj", None
        else:
            m = layer.self_attn
            if hasattr(m, "qkv"):
                yield f"{p}.self_attn.qkv", m, "qkv", "qkv", list(zip(("q_proj", "k_proj", "v_proj"), m.splits))
            else:
                for n in ("q_proj", "k_proj", "v_proj"): yield f"{p}.self_attn.{n}", m, n, n, None
            yield f"{p}.self_attn.o_proj", m, "o_proj", "o_proj", None
        mlp = layer.mlp
        if hasattr(mlp, "gate_up"):
            half = mlp.gate_up.shape[0] // 2 if isinstance(mlp.gate_up, torch.Tensor) else None
            yield f"{p}.mlp.gate_up", mlp, "gate_up", "gate_up", [("gate_proj", half), ("up_proj", half)]
        else:
            for n in ("gate_proj", "up_proj"): yield f"{p}.mlp.{n}", mlp, n, n, None
        yield f"{p}.mlp.down_proj", mlp, "down_proj", "down_proj", None


def precision_for(scheme, role):
    if scheme == "bf16": return "bf16"
    if scheme == "nvfp4-mlp": return "nvfp4" if role in MLP_ROLES else "fp8"
    return scheme


@torch.no_grad()
def quantize_model(model, scheme, exclude=(), min_dim=MIN_DIM, verbose=True):
    """Replace the projections of a loaded torch DecisionModel's decoder layers in place (see the module docstring).
    exclude: regexes on qualified names (layers.3.mlp.down_proj, ...) to keep bf16. -> report dict (coverage, fallbacks,
    memory). The model must be merged (no peft wrappers) and bf16."""
    if scheme not in SCHEMES: raise ValueError(f"scheme must be one of {SCHEMES}")
    lm = model.lm
    if any(type(m).__module__.startswith("peft") for m in lm.modules()):
        raise ValueError("quantization needs merged weights; load with merge on (KEV_MERGE unset)")
    if any(hasattr(l.mlp, "gate_up") for l in lm.layers): _install_fused_shim()   # kev.fused_qwen35 ran
    exclude = [re.compile(e) for e in exclude]
    dev = next(lm.parameters()).device
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    entries, t0 = [], time.time()

    def build(name, weight, bias, role):
        """-> (module or None for bf16, precision, reason)."""
        n, k = weight.shape
        want = precision_for(scheme, role)
        if want == "bf16": return None, "bf16", "scheme"
        if any(e.search(name) for e in exclude): return None, "bf16", "excluded"
        if min(n, k) < min_dim: return None, "bf16", f"small ({n}x{k} < {min_dim})"
        ok, why = QUANT[want].supports(n, k)
        if not ok: return None, "bf16", why
        return QUANT[want](weight, bias), want, ""

    for i, layer in enumerate(lm.layers):
        for name, owner, attr, role, members in [t for t in _targets(lm) if t[0].startswith(f"layers.{i}.")]:
            obj = getattr(owner, attr)
            if isinstance(obj, nn.Module) and not isinstance(obj, nn.Linear):   # already quantized
                continue
            weight, bias = (obj.weight, obj.bias) if isinstance(obj, nn.Linear) else (obj, None)
            if weight.dtype != torch.bfloat16: raise ValueError(f"{name}: expected bf16 weights, got {weight.dtype} (load with dtype=bf16)")
            if members is None:
                mod, prec, why = build(name, weight, bias, role)
                entries.append({"name": name, "shape": list(weight.shape), "params": weight.numel(), "precision": prec, "reason": why})
                if mod is not None: setattr(owner, attr, mod)
                continue
            # fused weight: decide per original member; small members must form a contiguous tail
            sizes = [r for _, r in members]
            small = [min(r, weight.shape[1]) < min_dim for r in sizes]
            lead = small.index(True) if True in small else len(small)
            if not all(small[lead:]):
                mod, prec, why = None, "bf16", "fused weight with small members not at its tail"
                rows = weight.shape[0]
            else:
                rows = sum(sizes[:lead])
                mod, prec, why = build(name, weight[:rows], None, role) if rows else (None, "bf16", "all members small")
            for (mname, r), s in zip(members, small):
                entries.append({"name": f"{name}[{mname}]", "shape": [r, weight.shape[1]], "params": r * weight.shape[1],
                                "precision": "bf16" if s or mod is None else prec,
                                "reason": f"small ({r}x{weight.shape[1]} < {min_dim}), split off" if s and mod is not None else why})
            if mod is not None:
                setattr(owner, attr, SplitLinear(mod, weight[rows:]) if rows < weight.shape[0] else mod)
            del weight, obj
        torch.cuda.empty_cache()

    torch.cuda.synchronize(); torch.cuda.empty_cache()
    report = coverage(entries)
    report.update(scheme=scheme, min_dim=min_dim, exclude=[e.pattern for e in exclude], seconds=round(time.time() - t0, 1),
                  fused=any(hasattr(l.mlp, "gate_up") for l in lm.layers),
                  nvfp4_backend=Nvfp4Linear.backend, fp8_act_quant="compile" if _fp8_quant is not _fp8_quant_rows else "torch",
                  memory={"allocated_before_gib": round(before / 2**30, 3), "allocated_after_gib": round(torch.cuda.memory_allocated(dev) / 2**30, 3),
                          "peak_during_gib": round(torch.cuda.max_memory_allocated(dev) / 2**30, 3)},
                  layers=entries)
    if verbose: print_report(report)
    return report


def coverage(entries):
    total = sum(e["params"] for e in entries) or 1
    table = {}
    for e in entries:
        t = table.setdefault(e["precision"], {"count": 0, "params": 0})
        t["count"] += 1; t["params"] += e["params"]
    for t in table.values(): t["share"] = round(t["params"] / total, 4)
    fallbacks = {}
    for e in entries:
        if e["precision"] == "bf16" and e["reason"] != "scheme":
            key = re.sub(r"layers\.\d+\.", "layers.*.", e["name"]) + f"  ({e['reason']})"
            fallbacks[key] = fallbacks.get(key, 0) + 1
    return {"coverage": table, "projection_params": total, "fallbacks": fallbacks}


def print_report(r):
    print(f"[kev_quant] scheme={r['scheme']} fused={r['fused']} nvfp4_backend={r['nvfp4_backend']} fp8_act_quant={r['fp8_act_quant']} ({r['seconds']} s)")
    print(f"[kev_quant] {'precision':10s} {'count':>6s} {'params':>14s} {'share':>7s}")
    for p, t in sorted(r["coverage"].items()):
        print(f"[kev_quant] {p:10s} {t['count']:6d} {t['params']:14,d} {t['share']:7.1%}")
    for k, n in r["fallbacks"].items(): print(f"[kev_quant] bf16 fallback x{n}: {k}")
    m = r["memory"]
    print(f"[kev_quant] GPU allocated {m['allocated_before_gib']} -> {m['allocated_after_gib']} GiB (peak during {m['peak_during_gib']} GiB)", flush=True)


# --- static activation scales ---------------------------------------------------------------------------------------------

def _nvfp4_modules(model):
    return {n: m for n, m in model.lm.named_modules() if isinstance(m, Nvfp4Linear)}


def _set_act_mode(mods, mode):
    for m in mods.values(): m.act_mode = mode


@torch.no_grad()
def calibrate_model(model, tok, records, margin, path, run, scheme, fused, suite=CALIB_SUITE):
    """Static NVFP4 activation scales: run `records` (the calibration partition of `suite`, never development or test)
    through model.probs (the serving path) with every Nvfp4Linear in 'calibrate' mode, then freeze
    ga = 448*6 / (margin * calibrated amax) as a device buffer. The amaxes + meta are saved to `path` (JSON keyed by module
    name); a file whose (run, scheme, fused, record count) match is loaded instead, and the margin is applied to its amaxes.
    -> info dict for quant.json."""
    from kev.data import materialize
    mods = _nvfp4_modules(model)
    info = {"mode": "static", "margin": margin, "calib_records": len(records), "calib_suite": suite, "calib_partition": "calibration",
            "scales_path": str(path), "modules": len(mods)}
    if not mods:
        info.update(sha256=None, loaded=False); return info
    meta = {"run": run, "scheme": scheme, "fused": fused, "calib_records": len(records), "suite": suite, "partition": "calibration"}
    saved = None
    if Path(path).exists():
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("meta") == meta and set(d.get("amax", {})) == set(mods): saved = d
        else: print(f"[kev_quant] {path} does not match {meta} / these modules; recalibrating", flush=True)
    t0 = time.time()
    if saved is None:
        for m in mods.values(): m.act_amax.zero_()
        _set_act_mode(mods, "calibrate")
        for r in records:
            model.probs(model.encode(tok, materialize(r), max_state=65536, max_branch=73728))
        torch.cuda.synchronize()
        amax = {n: m.act_amax.item() for n, m in mods.items()}
        body = {"meta": meta, "margin": margin, "amax": amax,
                "ga": {n: FP8_MAX * FP4_MAX / (margin * max(a, 1e-12)) for n, a in amax.items()}}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")
    else:
        amax = saved["amax"]
    for n, m in mods.items():
        m.ga.copy_(torch.tensor([FP8_MAX * FP4_MAX / (margin * max(amax[n], 1e-12))], device=m.ga.device))
    _set_act_mode(mods, "static")
    info.update(sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), loaded=saved is not None, seconds=round(time.time() - t0, 1))
    print(f"[kev_quant] static activation scales: {len(mods)} modules, {len(records)} records, margin {margin}, "
          f"{'loaded from' if saved else 'calibrated, saved to'} {path} (sha256 {info['sha256'][:12]}, {info['seconds']} s)", flush=True)
    return info


def default_scales_path(requested, scheme):
    return f"runs/spark/quant/act-scales-{re.sub(r'[^A-Za-z0-9._-]+', '_', str(requested))}-{scheme}.json"


# --- wrapping kev's loader ------------------------------------------------------------------------------------------------

STATE = {"report": None, "load_seconds": None, "opts": None, "act_scale": None}


def install(scheme, exclude=(), fused=None, graphs=None, act_scale="static", act_margin=4.0, calib_records=64, act_scales=None):
    """Patch kev.checkpoint.Checkpoint._load_torch: bf16 backbone, CUDA graphs attached only after quantization.
    fused / graphs: None = whatever the caller's LoadOptions say; True / False to override. act_scale: NVFP4 activation scale,
    'static' (calibrated right after quantize_model, before CUDA graphs) or 'dynamic'; act_scales: the JSON path."""
    from kev.checkpoint import Checkpoint
    original = Checkpoint._load_torch

    def _load_torch(self, tok, device, opts):
        t0 = time.time()
        opts = replace(opts, dtype=torch.bfloat16, merge=True,
                       fused=opts.fused if fused is None else fused, cuda_graphs=opts.cuda_graphs if graphs is None else graphs)
        STATE["opts"] = {"dtype": "bfloat16", "fused": bool(opts.fused), "cuda_graphs": bool(opts.cuda_graphs), "merge": True}
        m = original(self, tok, device, replace(opts, cuda_graphs=False))
        if not str(device).startswith("cuda"): raise ValueError("kev_quant needs CUDA")
        STATE["report"] = quantize_model(m, scheme, exclude)
        mods = _nvfp4_modules(m)
        if act_scale == "static" and mods:
            from kev.suite import load_split
            records = load_split(CALIB_SUITE, "calibration")[:calib_records]
            STATE["act_scale"] = calibrate_model(m, tok, records, act_margin, act_scales or default_scales_path(self.requested, scheme + ("-fused" if opts.fused else "")),
                                                 self.requested, scheme, bool(opts.fused))
        else:
            _set_act_mode(mods, "dynamic")
            STATE["act_scale"] = {"mode": "dynamic" if mods else "not applicable (no NVFP4 layers)"}
        if opts.cuda_graphs and m.hybrid:
            from kev.cuda_graphs import CudaGraphs
            m.graphs = CudaGraphs(m.lm, m.pad_id)   # captures lazily (kev.serve's model thread), so always after quantization
        STATE["load_seconds"] = round(time.time() - t0, 1)
        return m

    Checkpoint._load_torch = _load_torch


def environment():
    def v(name):
        try: return version(name)
        except PackageNotFoundError: return None
    return {"packages": {"torch": torch.__version__, **{n: v(n) for n in ("transformers", "peft", "flash-linear-attention", "flashinfer-python", "triton")}},
            "gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()), "cuda": torch.version.cuda}


def meminfo_gib():
    try:
        d = dict(l.split(":", 1) for l in Path("/proc/meminfo").read_text().splitlines())
        return {k: round(int(d[k].split()[0]) / 2**20, 2) for k in ("MemTotal", "MemAvailable")}
    except Exception:
        return None


def write_quant_json(path, scheme, extra):
    r = STATE["report"] or {}
    body = {"scheme": scheme, "load_options": STATE["opts"], "load_seconds": STATE["load_seconds"],
            **{k: r.get(k) for k in ("coverage", "projection_params", "fallbacks", "fused", "nvfp4_backend", "fp8_act_quant", "min_dim", "exclude", "memory")},
            "act_scale": STATE["act_scale"], "environment": environment(), **extra, "layers": r.get("layers")}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    print(f"[kev_quant] wrote {path}")


# --- layercheck: quantized vs bf16 output of real decoder projections on real activations ----------------------------------

@torch.no_grad()
def layercheck(run, suite, names, records=1, out=None):
    from kev.checkpoint import Checkpoint, LoadOptions
    from kev.data import materialize
    from kev.suite import load_split
    tok, model = Checkpoint(run).load("cuda", LoadOptions(dtype=torch.bfloat16))
    recs = load_split(suite, "development")[:records]
    mods = dict(model.lm.named_modules())
    captured = {n: [] for n in names}
    hooks = [mods[n].register_forward_hook(lambda m, i, o, n=n: captured[n].append(i[0].detach().reshape(-1, i[0].shape[-1]).clone())) for n in names]
    for r in recs:
        model.forward(model.encode(tok, materialize(r), max_state=65536, max_branch=73728))
    for h in hooks: h.remove()
    results = []
    for n in names:
        lin = mods[n]
        x = torch.cat(captured[n])
        ref = F.linear(x, lin.weight, lin.bias).float()
        row = {"name": n, "shape": list(lin.weight.shape), "tokens": x.shape[0], "act_absmax": x.float().abs().max().item(),
               "act_absmax_over_rms": (x.float().abs().max() / x.float().pow(2).mean().sqrt()).item()}
        for prec, cls in QUANT.items():
            ok, why = cls.supports(*lin.weight.shape)
            if not ok or min(lin.weight.shape) < MIN_DIM:
                row[prec] = None; continue
            q = cls(lin.weight, lin.bias)
            y = q(x).float()
            row[prec] = {"rel_err": ((y - ref).norm() / ref.norm()).item(),
                         "cos": F.cosine_similarity(y.flatten(), ref.flatten(), 0).item()}
            # weight-only error (bf16 activations through the dequantized weight) separates the two sources for fp8
            if prec == "fp8":
                wd = q.qweight.float() * q.wscale.t()
                yw = F.linear(x.float(), wd).float()
                row[prec]["rel_err_weight_only"] = ((yw - ref).norm() / ref.norm()).item()
        results.append(row)
        print(json.dumps(row), flush=True)
    if out: Path(out).write_text(json.dumps({"run": run, "suite": suite, "records": records, "environment": environment(), "layers": results}, indent=2) + "\n")
    return results


# --- isolation: a question's probabilities alone vs inside its full record ------------------------------------------------

@torch.no_grad()
def isolation(run, records=5, out=None):
    """For `records` development records of decision-v7 with >= 2 questions: score the full record, then each question alone
    (a record with just that question), through model.probs. -> max |dp| over every question (0.0 = bit-identical)."""
    from kev.checkpoint import Checkpoint, LoadOptions
    from kev.data import materialize
    from kev.suite import load_split
    tok, model = Checkpoint(run).load("cuda", LoadOptions(dtype=torch.bfloat16))
    recs = [r for r in load_split(CALIB_SUITE, "development") if len(r["questions"]) >= 2][:records]
    rows, worst = [], 0.0
    for r in recs:
        enc = lambda rec: model.encode(tok, materialize(rec), max_state=65536, max_branch=73728)
        full = model.probs(enc(r))
        for k, qid in enumerate(r["questions"]):
            alone = model.probs(enc({**r, "questions": {qid: r["questions"][qid]}}))[0]
            d = (full[k].float() - alone.float()).abs().max().item()
            worst = max(worst, d)
            rows.append({"qid": qid, "questions_in_record": len(r["questions"]), "max_abs_dp": d, "bit_identical": bool(torch.equal(full[k], alone))})
            print(f"[kev_quant] isolation {qid} ({len(r['questions'])} questions): max|dp|={d:.3e}", flush=True)
    res = {"run": run, "scheme": STATE["report"]["scheme"] if STATE["report"] else None, "act_scale": STATE["act_scale"],
           "records": len(recs), "questions": len(rows), "max_abs_dp": worst, "all_bit_identical": all(x["bit_identical"] for x in rows), "rows": rows}
    print(f"[kev_quant] ISOLATION scheme={res['scheme']} act_scale={(STATE['act_scale'] or {}).get('mode')} records={len(recs)} "
          f"questions={len(rows)} max|dp|={worst:.3e} bit_identical={res['all_bit_identical']}", flush=True)
    if out: Path(out).parent.mkdir(parents=True, exist_ok=True); Path(out).write_text(json.dumps(res, indent=2) + "\n")
    return res


# --- CLI ------------------------------------------------------------------------------------------------------------------

def _flag(v):
    return None if v == "auto" else v == "1"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmds = ("serve", "benchmark", "layercheck", "isolation")
    at = next((i for i, a in enumerate(argv) if a in cmds), None)
    if at is None: sys.exit(f"usage: kev_quant.py [--scheme S] [--fused auto|0|1] [--graphs auto|0|1] [--exclude RE ...] {{{','.join(cmds)}}} ...")
    ap = argparse.ArgumentParser(prog="kev_quant.py")
    ap.add_argument("--scheme", choices=SCHEMES, default="bf16")
    ap.add_argument("--fused", choices=["auto", "0", "1"], default="auto", help="kev.fused_qwen35 (auto = the entry point's default)")
    ap.add_argument("--graphs", choices=["auto", "0", "1"], default="auto", help="kev.cuda_graphs (auto = the entry point's default)")
    ap.add_argument("--exclude", nargs="*", default=[], help="regexes on projection names to keep bf16")
    ap.add_argument("--nvfp4-backend", default="cutlass", choices=["cutlass", "b12x", "cudnn"])
    ap.add_argument("--act-quant", default="torch", choices=["torch", "compile"], help="FP8 per-token activation quantization: eager or torch.compile'd")
    ap.add_argument("--act-scale", choices=["static", "dynamic"], default="static",
                    help="NVFP4 activation global scale: calibrated per layer and fixed (default), or per call from the batch's amax")
    ap.add_argument("--act-margin", type=float, default=4.0, help="static: headroom over the calibrated amax (e4m3 block scales have range to spare)")
    ap.add_argument("--calib-records", type=int, default=64, help="static: records of decision-v7's calibration partition to calibrate on")
    ap.add_argument("--act-scales", help="static: scales JSON, loaded if present (matching run / scheme / records) else written "
                                         "(default runs/spark/quant/act-scales-<run>-<scheme>.json)")
    ap.add_argument("--quant-json", help="serve: also write the quantization report here")
    a = ap.parse_args(argv[:at])
    cmd, rest = argv[at], argv[at + 1:]
    Nvfp4Linear.backend = a.nvfp4_backend
    set_act_quant(a.act_quant)

    if cmd == "layercheck":
        lp = argparse.ArgumentParser(prog="kev_quant.py layercheck")
        lp.add_argument("--run", required=True); lp.add_argument("--suite", required=True)
        lp.add_argument("--records", type=int, default=1); lp.add_argument("--out")
        lp.add_argument("--names", nargs="*")
        b = lp.parse_args(rest)
        names = b.names or ["layers.0.linear_attn.in_proj_qkv", "layers.0.linear_attn.in_proj_z", "layers.0.linear_attn.out_proj",
                            "layers.3.self_attn.q_proj", "layers.3.self_attn.o_proj", "layers.0.mlp.gate_proj", "layers.0.mlp.down_proj",
                            "layers.11.mlp.up_proj", "layers.11.mlp.down_proj", "layers.23.mlp.down_proj"]
        return layercheck(b.run, b.suite, names, b.records, b.out)

    install(a.scheme, a.exclude, _flag(a.fused), _flag(a.graphs), a.act_scale, a.act_margin, a.calib_records, a.act_scales)
    if cmd == "isolation":
        ip = argparse.ArgumentParser(prog="kev_quant.py isolation")
        ip.add_argument("--run", required=True); ip.add_argument("--records", type=int, default=5); ip.add_argument("--out")
        b = ip.parse_args(rest)
        return isolation(b.run, b.records, b.out)
    if cmd == "serve":
        import kev.serve
        sys.argv = ["kev.serve", *rest]
        if a.quant_json:   # written once the model is loaded, before uvicorn blocks
            import uvicorn
            run = uvicorn.run
            def run_after_report(*args, **kw):
                write_quant_json(a.quant_json, a.scheme, {"entry": "serve", "argv": rest, "meminfo_after_load": meminfo_gib(),
                                                          "gpu_allocated_after_load_gib": round(torch.cuda.memory_allocated() / 2**30, 3)})
                return run(*args, **kw)
            uvicorn.run = run_after_report
        return kev.serve.main()

    import kev.benchmark
    bp = argparse.ArgumentParser(add_help=False); bp.add_argument("--out", required=True)
    out = bp.parse_known_args(rest)[0].out
    sys.argv = ["kev.benchmark", *rest]
    t0 = time.time()
    kev.benchmark.main()
    wall = round(time.time() - t0, 1)
    report = json.loads((Path(out) / "report.json").read_text())
    write_quant_json(Path(out) / "quant.json", a.scheme, {
        "entry": "benchmark", "argv": rest, "wall_seconds": wall,
        "gpu_peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "gpu_allocated_end_gib": round(torch.cuda.memory_allocated() / 2**30, 3), "meminfo_end": meminfo_gib(),
        "clean_accuracy": report["clean"].get("acc"), "latency_ms": report.get("latency_ms")})
    print(f"[kev_quant] scheme={a.scheme} clean accuracy={report['clean'].get('acc')} wall={wall}s "
          f"gpu peak={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
