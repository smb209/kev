"""Pre-quantized FP8 export of Cloudflare's Clef-Flash, and a loader that runs it on a 16 GB Ada GPU (sm_89).

E10 (spark/LAB_NOTEBOOK.md). The language model's decoder-layer projections become PortableFp8Linear: e4m3 weights with an
fp32 per-output-channel scale, e4m3 activations with a dynamic per-token scale, and torch._scaled_mm with tensorwise unit
scales and an fp32 output, rescaled per token x per channel in fp32. kev_quant.Fp8Linear's rowwise _scaled_mm runs on sm_121
only in this torch; the tensorwise form runs on sm_89 and sm_121 alike, so the Spark computes what the Ada card computes
(the kernel cuBLASLt picks may still differ). Everything else stays bf16: embeddings, norms, lm_head (Clef's head reads its
weight), the whole vision tower and its merger, projections with a dimension under 1,024 (Gated DeltaNet's in_proj_a /
in_proj_b) and the JointSchemaHead.

The export (a directory) holds safetensors shards (float8_e4m3fn weights, fp32 `<name>.weight_scale`, bf16 everything else),
`clef_fp8.safetensors.index.json`, `clef_fp8_config.json`, the release's config / tokenizer / processor / head files and a copy
of its joint_schema_model.py (export dirs live under runs/, which is gitignored; the copy is imported only if its sha256 is
the hand-reviewed one). load_fp8 builds the backbone with its parameters on the meta device and assigns each tensor of the
shards straight to the GPU, so loading never holds bf16 decoder weights.

    python spark/clef_fp8.py selftest
    python spark/clef_fp8.py quantize --out runs/spark/c10-clef-flash-fp8
    python spark/clef_fp8.py check    --export runs/spark/c10-clef-flash-fp8 --out runs/spark/c10/check.json
    python spark/clef_fp8.py memtest  --export runs/spark/c10-clef-flash-fp8 --cap-gib 15.0 --out runs/spark/c10/memtest.json

Serving needs only this file, spark/clef_server.py and the export (no `kev` package): clef_server.py --fp8-export DIR.
check / memtest / selftest import kev (suites, kev_quant) and are meant for the Spark.
"""
import argparse
import contextlib
import gc
import hashlib
import importlib.util
import json
import re
import shutil
import sys
import time
from pathlib import Path

import os
import torch
import torch.nn as nn

REPO = "Cloudflare/clef-flash"
REVISION = "17f0b0ad64efb65d273590632833508766b2aae6"
REVIEWED_SHA256 = "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"   # joint_schema_model.py (clef_server.py)
FORMAT = "clef-fp8/1"
CONFIG_NAME = "clef_fp8_config.json"
INDEX_NAME = "clef_fp8.safetensors.index.json"
CODE_NAME = "joint_schema_model.py"
COPY_FILES = ("config.json", "generation_config.json", "chat_template.jinja", "tokenizer.json", "tokenizer_config.json",
              "processor_config.json", "joint_head.safetensors", "joint_head_config.json", "LICENSE", "README.md")
FP8_MAX = 448.0
MIN_DIM = 1024          # either dimension under this stays bf16 (kev_quant.MIN_DIM)
ALIGN = 16              # torch._scaled_mm: K and N multiples of 16
SHARD_BYTES = 4 * 2**30
QUANT_CHUNK = 2048      # rows per activation-quantization chunk (memory only; the result is the same)


def gib(x):
    return round(x / 2**30, 3)


# --- release code (sha256-gated) ------------------------------------------------------------------------------------------

def import_release_code(code_path):
    """Import Clef's joint_schema_model.py only if its sha256 is the hand-reviewed one; one module per process."""
    code_path = Path(code_path)
    digest = hashlib.sha256(code_path.read_bytes()).hexdigest()
    if digest != REVIEWED_SHA256:
        raise SystemExit(f"{code_path} sha256 {digest} is not the reviewed {REVIEWED_SHA256}; review it before running")
    if "joint_schema_model" in sys.modules:
        return sys.modules["joint_schema_model"]
    spec = importlib.util.spec_from_file_location("joint_schema_model", code_path)
    jsm = importlib.util.module_from_spec(spec); sys.modules["joint_schema_model"] = jsm; spec.loader.exec_module(jsm)
    return jsm


def release_path(repo=REPO, revision=REVISION):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo, revision=revision))


# --- PortableFp8Linear ----------------------------------------------------------------------------------------------------

_ONES = {}


def _one(device):
    """fp32 scalar 1 on `device` for _scaled_mm's tensorwise scales (created once per device)."""
    t = _ONES.get(device)
    if t is None:
        t = _ONES[device] = torch.ones((), dtype=torch.float32, device=device)
    return t


_OUT = {}


def gemm_out_dtype(device):
    """The _scaled_mm output dtype this GPU accepts for e4m3 x e4m3 with tensorwise scales, probed once per device:
    float32 (the arithmetic every E10 read used: the per-token x per-channel rescale happens before any rounding) or,
    where cuBLASLt refuses fp32 output (possible on sm_89; unverified), bfloat16 (one bf16 rounding before the rescale,
    like rowwise kernels with bf16 out). FP8_GEMM_OUT=float32|bfloat16 forces one. Reported by clef_server at start."""
    key = str(device)
    if key not in _OUT:
        forced = os.environ.get("FP8_GEMM_OUT")
        if forced: _OUT[key] = getattr(torch, forced)
        else:
            a = torch.zeros(16, 16, device=device, dtype=torch.float8_e4m3fn)
            try:
                torch._scaled_mm(a, a.t(), scale_a=_one(device), scale_b=_one(device), out_dtype=torch.float32)
                _OUT[key] = torch.float32
            except RuntimeError:
                _OUT[key] = torch.bfloat16
    return _OUT[key]


def _quant_rows(x2):
    amax = torch.linalg.vector_norm(x2, float("inf"), dim=-1, keepdim=True).float().clamp_(min=1e-12)
    scale = amax / FP8_MAX
    return (x2 / scale).to(torch.float8_e4m3fn), scale


def quant_rows(x2, chunk=QUANT_CHUNK):
    """bf16 [M, K] -> (e4m3 [M, K], fp32 [M, 1]): dynamic per-token scale amax / 448, on device (kev_quant._fp8_quant_rows).
    Above `chunk` rows it runs per chunk of rows into preallocated outputs: the same bits (elementwise ops and a per-row
    max), without the fp32 [M, K] temporary of the whole input (768 MiB for a 16k-token down_proj input)."""
    m = x2.shape[0]
    if m <= chunk: return _quant_rows(x2)
    xq = torch.empty(x2.shape, dtype=torch.float8_e4m3fn, device=x2.device)
    xs = torch.empty((m, 1), dtype=torch.float32, device=x2.device)
    for i in range(0, m, chunk):
        xq[i:i + chunk], xs[i:i + chunk] = _quant_rows(x2[i:i + chunk])
    return xq, xs


def quant_weight(w):
    """[N, K] -> (e4m3 [N, K], fp32 [N]): per-output-channel scale amax / 448 (kev_quant.Fp8Linear's weight rule)."""
    w = w.detach().float()
    scale = w.abs().amax(1, keepdim=True).clamp_(min=1e-12) / FP8_MAX
    return (w / scale).to(torch.float8_e4m3fn), scale.squeeze(1).contiguous()


class PortableFp8Linear(nn.Module):
    """y = x W^T (+ b). W: `weight` float8_e4m3fn [N, K] and `weight_scale` fp32 [N] (W ~ weight * weight_scale[:, None]);
    x: e4m3 with a dynamic per-token scale. torch._scaled_mm(xq, W^T, scale_a=1, scale_b=1, out_dtype=fp32) (tensorwise,
    which sm_89 supports), then y = out * x_scale[:, None] * weight_scale[None, :] in fp32, + bias, -> bf16. Any leading dims."""
    precision = "fp8"

    def __init__(self, out_features, in_features, bias=False, device=None):
        super().__init__()
        self.out_features, self.in_features = out_features, in_features
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn, device=device))
        self.register_buffer("weight_scale", torch.empty(out_features, dtype=torch.float32, device=device))
        self.register_buffer("bias", torch.empty(out_features, dtype=torch.bfloat16, device=device) if bias else None)

    @classmethod
    def from_linear(cls, linear):
        n, k = linear.weight.shape
        m = cls(n, k, bias=linear.bias is not None, device="meta")
        q, s = quant_weight(linear.weight)
        m.weight, m.weight_scale = q, s
        if linear.bias is not None: m.bias = linear.bias.detach().to(torch.bfloat16).clone()
        return m

    def forward(self, x):
        shape = x.shape
        x2 = x.reshape(-1, self.in_features)
        if x2.dtype != torch.bfloat16: x2 = x2.to(torch.bfloat16)
        x2 = x2.contiguous()
        xq, xs = quant_rows(x2)
        one = _one(x2.device)
        y = torch._scaled_mm(xq, self.weight.t(), scale_a=one, scale_b=one, out_dtype=gemm_out_dtype(x2.device))
        if y.dtype != torch.float32: y = y.float()          # bf16-out fallback (gemm_out_dtype): rescale in fp32 after one rounding
        y.mul_(xs).mul_(self.weight_scale.unsqueeze(0))     # in place: the bits of out * xs * w_scale, one fp32 [M, N] buffer
        if self.bias is not None: y.add_(self.bias.float())
        return y.to(torch.bfloat16).view(*shape[:-1], self.out_features)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, fp8 e4m3"


def eligibility(n, k, min_dim=MIN_DIM):
    """-> reason the [N, K] projection stays bf16, or '' when it can be FP8."""
    if min(n, k) < min_dim: return f"small ({n}x{k} < {min_dim})"
    if n % ALIGN or k % ALIGN: return f"alignment ({n}x{k}: _scaled_mm needs N and K multiples of {ALIGN})"
    return ""


# --- quantize -------------------------------------------------------------------------------------------------------------

def _backbone(model):
    lm = model.language_model
    return lm.get_base_model() if hasattr(lm, "get_base_model") else lm


def _text_model(backbone):
    return backbone.model.language_model     # Qwen3_5ForConditionalGeneration -> Qwen3_5Model -> Qwen3_5TextModel


def _decoder_linears(text_model):
    """-> [(name under the text model, parent, attribute, module)] for every nn.Linear / PortableFp8Linear of its decoder layers."""
    out = []
    for i, layer in enumerate(text_model.layers):
        for name, mod in layer.named_modules():
            if isinstance(mod, (nn.Linear, PortableFp8Linear)):
                parent_name, _, attr = name.rpartition(".")
                out.append((f"layers.{i}.{name}", layer.get_submodule(parent_name) if parent_name else layer, attr, mod))
    return out


def _sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()


@torch.no_grad()
def quantize_clef(model, min_dim=MIN_DIM, verbose=True):
    """Replace the eligible nn.Linear modules of the language model's decoder layers with PortableFp8Linear, layer by layer,
    freeing each bf16 weight as its replacement is built. -> report (coverage, fallbacks, memory, fp8 module names)."""
    backbone = _backbone(model)
    tm = _text_model(backbone)
    dev = next(tm.parameters()).device
    _sync(); torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    t0, entries, prefix = time.time(), [], _prefix(backbone, tm)
    for i in range(len(tm.layers)):
        for name, parent, attr, mod in _decoder_linears(tm):
            if not name.startswith(f"layers.{i}."): continue
            n, k = mod.weight.shape
            why = eligibility(n, k, min_dim) if isinstance(mod, nn.Linear) else ""
            if isinstance(mod, nn.Linear):
                if mod.weight.dtype != torch.bfloat16: raise ValueError(f"{name}: expected bf16 weights, got {mod.weight.dtype}")
                if not why:
                    setattr(parent, attr, PortableFp8Linear.from_linear(mod))
            entries.append({"name": prefix + name, "shape": [n, k], "params": n * k, "precision": "bf16" if why else "fp8", "reason": why})
            del mod
        gc.collect(); torch.cuda.empty_cache()
    _sync(); torch.cuda.empty_cache()
    report = coverage(entries, model)
    report.update(min_dim=min_dim, seconds=round(time.time() - t0, 1),
                  memory={"allocated_before_gib": gib(before), "allocated_after_gib": gib(torch.cuda.memory_allocated(dev)),
                          "peak_during_gib": gib(torch.cuda.max_memory_allocated(dev))},
                  fp8_modules=[e["name"] for e in entries if e["precision"] == "fp8"],
                  layers=entries)
    if verbose: print_report(report)
    return report


def _prefix(backbone, tm):
    for name, mod in backbone.named_modules():
        if mod is tm: return name + "."
    raise ValueError("text model not found in the backbone")


def coverage(entries, model):
    """Decoder-projection table (count, params, share per precision) plus whole-model parameter counts by storage precision."""
    total = sum(e["params"] for e in entries) or 1
    table = {}
    for e in entries:
        t = table.setdefault(e["precision"], {"count": 0, "params": 0})
        t["count"] += 1; t["params"] += e["params"]
    for t in table.values(): t["share"] = round(t["params"] / total, 4)
    fallbacks = {}
    for e in entries:
        if e["precision"] == "bf16":
            key = re.sub(r"layers\.\d+\.", "layers.*.", e["name"]) + f"  ({e['reason']})"
            fallbacks[key] = fallbacks.get(key, 0) + 1
    return {"coverage": table, "projection_params": total, "fallbacks": fallbacks, "model_params": model_params(model)}


def model_params(model):
    """Weight elements by where they live and how they are stored (tied tensors counted once; scales counted as fp32)."""
    backbone = _backbone(model)
    seen, out = set(), {}

    def add(group, t):
        if t is None or t.data_ptr() in seen and t.device.type != "meta": return
        seen.add(t.data_ptr())
        key = f"{group}/{str(t.dtype).replace('torch.', '')}"
        out[key] = out.get(key, 0) + t.numel()

    tm = _text_model(backbone)
    for name, t in list(backbone.named_parameters()) + list(backbone.named_buffers()):
        if t is None or (t.dtype != torch.float8_e4m3fn and not t.is_floating_point()): continue
        if name.endswith("inv_freq"): continue
        group = ("decoder" if ".layers." in name and name.startswith(_prefix(backbone, tm)) else
                 "vision" if ".visual." in name or name.startswith("model.visual") else
                 "embed/lm_head" if "embed_tokens" in name or name.startswith("lm_head") else "other")
        add(group, t)
    for t in model.head.parameters(): add("joint_head", t)
    return dict(sorted(out.items()))


def print_report(r):
    print(f"[clef_fp8] decoder projections ({r['seconds']} s)")
    print(f"[clef_fp8] {'precision':10s} {'count':>6s} {'params':>14s} {'share':>7s}")
    for p, t in sorted(r["coverage"].items()):
        print(f"[clef_fp8] {p:10s} {t['count']:6d} {t['params']:14,d} {t['share']:7.1%}")
    for k, n in r["fallbacks"].items(): print(f"[clef_fp8] bf16 fallback x{n}: {k}")
    print("[clef_fp8] whole model (elements by group/storage dtype):")
    for k, n in r["model_params"].items(): print(f"[clef_fp8]   {k:32s} {n:14,d}")
    m = r["memory"]
    print(f"[clef_fp8] GPU allocated {m['allocated_before_gib']} -> {m['allocated_after_gib']} GiB (peak during {m['peak_during_gib']} GiB)", flush=True)


# --- export ---------------------------------------------------------------------------------------------------------------

def _unique_state(backbone):
    """Backbone state dict with tied tensors kept once (lm_head.weight when tied to embed_tokens) -> (state, tied names)."""
    state, owner, tied = {}, {}, {}
    for name, t in backbone.state_dict().items():
        key = (t.data_ptr(), t.dtype, tuple(t.shape))
        if key in owner and t.data_ptr() != 0:
            tied[name] = owner[key]; continue
        owner[key] = name; state[name] = t
    return state, tied


@torch.no_grad()
def export(model, out_dir, source_path, report, repo=REPO, revision=REVISION):
    """Write the quantized model (see the module docstring). out_dir must not exist."""
    from safetensors.torch import save_file
    out = Path(out_dir)
    if out.exists(): raise SystemExit(f"{out} exists; refusing to overwrite an export")
    out.mkdir(parents=True)
    source_path = Path(source_path)
    backbone = _backbone(model)
    state, tied = _unique_state(backbone)
    shards, cur, size = [], {}, 0
    for name, t in state.items():
        nbytes = t.numel() * t.element_size()
        if cur and size + nbytes > SHARD_BYTES:
            shards.append(cur); cur, size = {}, 0
        cur[name] = t; size += nbytes
    if cur: shards.append(cur)
    weight_map, total = {}, 0
    for i, shard in enumerate(shards):
        fname = f"clef-fp8-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        cpu = {k: v.detach().to("cpu").contiguous() for k, v in shard.items()}
        save_file(cpu, str(out / fname), metadata={"format": "pt", "clef_fp8": FORMAT})
        total += sum(v.numel() * v.element_size() for v in cpu.values())
        for k in shard: weight_map[k] = fname
        del cpu
        print(f"[clef_fp8] wrote {fname} ({len(shard)} tensors)", flush=True)
    (out / INDEX_NAME).write_text(json.dumps({"metadata": {"total_size": total, "format": FORMAT}, "weight_map": weight_map}, indent=1) + "\n")
    copied = []
    for f in COPY_FILES:
        if (source_path / f).exists():
            shutil.copyfile(source_path / f, out / f); copied.append(f)
    shutil.copyfile(source_path / CODE_NAME, out / CODE_NAME)
    dtypes = {}
    for t in state.values(): dtypes[str(t.dtype).replace("torch.", "")] = dtypes.get(str(t.dtype).replace("torch.", ""), 0) + t.numel()
    cfg = {
        "format": FORMAT,
        "source": {"repo": repo, "revision": revision, "joint_schema_model_sha256": REVIEWED_SHA256},
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": environment(),
        "scale_layout": {
            "weight": "float8_e4m3fn [out_features, in_features] = round_e4m3(W / weight_scale[:, None])",
            "weight_scale": "float32 [out_features] = max(|W[o, :]|, 1e-12) / 448 (per output channel)",
            "activation": "dynamic per token: e4m3(x / s), s = max(|x[t, :]|, 1e-12) / 448, computed on device",
            "gemm": "torch._scaled_mm(xq, weight.t(), scale_a=1, scale_b=1, out_dtype=float32) * s[:, None] * weight_scale[None, :], + bias, -> bf16",
        },
        "min_dim": report["min_dim"], "align": ALIGN,
        "fp8_modules": report["fp8_modules"],
        "bf16_fallbacks": report["fallbacks"],
        "coverage": report["coverage"],
        "model_params": report["model_params"],
        "tensor_elements_by_dtype": dtypes,
        "tied": tied,
        "shards": sorted(set(weight_map.values())), "index": INDEX_NAME,
        "copied_files": copied, "release_code": CODE_NAME,
    }
    (out / CONFIG_NAME).write_text(json.dumps(cfg, indent=1) + "\n")
    disk = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    print(f"[clef_fp8] export {out}: {len(shards)} shards, {gib(total)} GiB of tensors, {gib(disk)} GiB on disk", flush=True)
    return {"dir": str(out), "shards": len(shards), "tensor_gib": gib(total), "disk_gib": gib(disk), "tied": tied}


def environment():
    import transformers
    try:
        import safetensors; st = safetensors.__version__
    except Exception:
        st = None
    return {"torch": torch.__version__, "transformers": transformers.__version__, "safetensors": st,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "capability": ".".join(map(str, torch.cuda.get_device_capability(0))) if torch.cuda.is_available() else None}


# --- load -----------------------------------------------------------------------------------------------------------------

@contextlib.contextmanager
def params_on_meta():
    """Parameters created inside go to the meta device; buffers (rotary inv_freq, ...) stay real on the CPU."""
    original = nn.Module.register_parameter

    def register(self, name, param):
        original(self, name, param)
        if param is not None and param.device.type != "meta":
            self._parameters[name] = nn.Parameter(param.to("meta"), requires_grad=param.requires_grad)

    nn.Module.register_parameter = register
    try:
        yield
    finally:
        nn.Module.register_parameter = original


def _assign(root, name, tensor):
    parent_name, _, attr = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    if attr in parent._parameters:
        parent._parameters[attr] = nn.Parameter(tensor, requires_grad=False)
    elif attr in parent._buffers:
        parent._buffers[attr] = tensor
    else:
        raise KeyError(f"{name}: no such parameter or buffer in the built model")


@torch.no_grad()
def load_fp8(export_dir, device="cuda", verbose=True):
    """Build Clef from an export: backbone on meta, target Linears swapped for PortableFp8Linear shells, every tensor of the
    shards assigned straight onto `device`, JointSchemaHead from the release's class. -> (ClefModel, processor), as
    load_release_model returns. Peak GPU memory during the load stays at the final size (one tensor in flight)."""
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration
    export_dir = Path(export_dir)
    cfg = json.loads((export_dir / CONFIG_NAME).read_text())
    if cfg.get("format") != FORMAT: raise SystemExit(f"{export_dir}: format {cfg.get('format')!r}, this loader reads {FORMAT}")
    jsm = import_release_code(export_dir / cfg.get("release_code", CODE_NAME))
    dev = torch.device(device)
    if dev.type == "cuda":
        dev = torch.device("cuda", torch.cuda.current_device() if dev.index is None else dev.index)
        _sync(); torch.cuda.reset_peak_memory_stats(dev)
    before = torch.cuda.memory_allocated(dev) if dev.type == "cuda" else 0
    t0 = time.time()
    config = AutoConfig.from_pretrained(export_dir)
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)     # as from_pretrained(dtype=bf16) builds the model
    try:
        with params_on_meta():
            backbone = Qwen3_5ForConditionalGeneration(config)
    finally:
        torch.set_default_dtype(default_dtype)
    backbone.config.use_cache = False
    tm = _text_model(backbone)
    prefix = _prefix(backbone, tm)
    fp8 = set(cfg["fp8_modules"])
    for name, parent, attr, mod in _decoder_linears(tm):
        if prefix + name in fp8:
            setattr(parent, attr, PortableFp8Linear(mod.out_features, mod.in_features, bias=mod.bias is not None, device="meta"))
    index = json.loads((export_dir / cfg["index"]).read_text())
    files = {}
    for k, f in index["weight_map"].items(): files.setdefault(f, []).append(k)
    expected = {k for k, _ in backbone.state_dict().items()}
    loaded = set()
    for f, keys in files.items():
        with safe_open(str(export_dir / f), framework="pt", device=str(dev)) as sf:
            for k in keys:
                _assign(backbone, k, sf.get_tensor(k)); loaded.add(k)
    for name, src in cfg.get("tied", {}).items():
        _assign(backbone, name, backbone.get_parameter(src) if src in dict(backbone.named_parameters()) else backbone.get_buffer(src))
        loaded.add(name)
    missing = expected - loaded
    if missing: raise SystemExit(f"{len(missing)} tensors missing from the export, e.g. {sorted(missing)[:5]}")
    backbone.to(dev)       # the CPU buffers (rotary inv_freq); every weight is already there
    meta = [n for n, t in list(backbone.named_parameters()) + list(backbone.named_buffers()) if t is not None and t.device.type == "meta"]
    if meta: raise SystemExit(f"tensors left on meta after the load: {meta[:5]}")
    if getattr(config, "tie_word_embeddings", False) or getattr(getattr(config, "text_config", None), "tie_word_embeddings", False):
        backbone.tie_weights()
    head_config = json.loads((export_dir / "joint_head_config.json").read_text())
    with params_on_meta():
        head = jsm.JointSchemaHead(**head_config)
    head.load_state_dict(load_file(str(export_dir / "joint_head.safetensors"), device=str(dev)), strict=True, assign=True)
    head = head.to(device=dev, dtype=torch.bfloat16)
    for p in head.parameters(): p.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(export_dir)
    model = jsm.ClefModel(backbone, head).eval()
    if dev.type == "cuda":
        _sync()
        stats = {"allocated_gib": gib(torch.cuda.memory_allocated(dev) - before), "peak_gib": gib(torch.cuda.max_memory_allocated(dev) - before)}
    else:
        stats = {}
    model.load_stats = {"seconds": round(time.time() - t0, 1), **stats}
    if verbose: print(f"[clef_fp8] loaded {export_dir} in {model.load_stats['seconds']} s: {stats}", flush=True)
    return model, processor


# --- inference helper (clef_server.py's path) ----------------------------------------------------------------------------

def probabilities(jsm, model, processor, body, max_length=16384):
    """-> (EncodedRecord, {question id: {option id: probability}}), exactly as clef_server.py computes them."""
    enc = jsm.encode_record(processor.tokenizer, body, max_length=max_length, processor=processor)
    with torch.inference_mode():
        logits = model(jsm.collate_records([enc], processor.tokenizer.pad_token_id, torch.device("cuda")))[0]
    return enc, {q.question_id: dict(zip(q.option_ids, z.float().softmax(-1).tolist())) for q, z in zip(enc.questions, logits)}


def load_bf16(path):
    jsm = import_release_code(Path(path) / CODE_NAME)
    return jsm.load_release_model(path, device="cuda")


def free(*objs):
    for o in objs: del o
    gc.collect(); torch.cuda.empty_cache(); _sync()


# --- CLI ------------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def cmd_selftest(a):
    """PortableFp8Linear vs kev_quant.Fp8Linear (rowwise _scaled_mm, bf16 out) and vs an fp64 dequantized reference."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from kev_quant import Fp8Linear
    torch.manual_seed(0)
    rows = []
    xc = (torch.randn(5000, 12288, device="cuda") * 2).to(torch.bfloat16)
    q1, s1 = _quant_rows(xc); q2, s2 = quant_rows(xc, chunk=1024)
    chunk_equal = bool(torch.equal(q1.view(torch.uint8), q2.view(torch.uint8)) and torch.equal(s1, s2))
    print(f"[selftest] chunked activation quantization equals one-shot: {chunk_equal}", flush=True)
    del xc, q1, q2
    for (m, k, n, bias) in [(1, 4096, 4096, False), (7, 4096, 12288, False), (300, 12288, 4096, True), (2048, 4096, 8192, False),
                            (33, 1024, 2048, True), (5, 4096, 6144, False), (2600, 12288, 4096, False)]:
        w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
        b = (torch.randn(n, device="cuda") * 0.1).to(torch.bfloat16) if bias else None
        x = (torch.randn(2, m, k, device="cuda") * torch.rand(2, m, 1, device="cuda") * 3).to(torch.bfloat16)
        x[0, 0, :8] *= 40     # an outlier token
        lin = nn.Linear(k, n, bias=bias, device="cuda", dtype=torch.bfloat16)
        lin.weight.copy_(w)
        if bias: lin.bias.copy_(b)
        port, ref = PortableFp8Linear.from_linear(lin), Fp8Linear(w, b)
        assert torch.equal(port.weight.view(torch.uint8), ref.qweight.view(torch.uint8)) and torch.equal(port.weight_scale, ref.wscale.squeeze(0))
        with torch.no_grad():
            yp, yr = port(x), ref(x)
            # fp32 output of the portable path before the bf16 cast, vs an fp64 dequantized matmul of the same e4m3 operands
            x2 = x.reshape(-1, k)
            xq, xs = quant_rows(x2)
            out = torch._scaled_mm(xq, port.weight.t(), scale_a=_one(x2.device), scale_b=_one(x2.device), out_dtype=torch.float32)
            y32 = out * xs * port.weight_scale.unsqueeze(0) + (b.float() if bias else 0)
            y64 = (xq.double() * xs.double()) @ (port.weight.double() * port.weight_scale.double()[:, None]).t() + (b.double() if bias else 0)
        denom = y64.abs().max().item()
        rel_bf = ((yp.float() - yr.float()).abs().max() / yr.float().abs().max()).item()
        rel_32 = ((y32.double() - y64).abs().max() / denom).item()
        rel_ref64 = ((yr.reshape(-1, n).double() - y64).abs().max() / denom).item()
        rows.append({"m": 2 * m, "k": k, "n": n, "bias": bias, "portable_vs_rowwise_bf16_max_rel": rel_bf,
                     "bitwise_equal_bf16": bool(torch.equal(yp, yr)), "frac_bf16_equal": float((yp == yr).float().mean()),
                     "portable_fp32_vs_fp64_dequant_max_rel": rel_32, "rowwise_bf16_vs_fp64_dequant_max_rel": rel_ref64})
        print(f"[selftest] M={2 * m:5d} K={k:5d} N={n:5d} bias={bias!s:5s} portable vs rowwise (bf16 out) max rel {rel_bf:.2e}, "
              f"equal {rows[-1]['frac_bf16_equal']:.4f}; portable fp32 vs fp64 dequant {rel_32:.2e}; rowwise vs fp64 {rel_ref64:.2e}", flush=True)
    res = {"environment": environment(), "cases": rows, "chunked_quant_equal": chunk_equal,
           "max_portable_vs_rowwise_rel": max(r["portable_vs_rowwise_bf16_max_rel"] for r in rows),
           "max_portable_fp32_vs_fp64_rel": max(r["portable_fp32_vs_fp64_dequant_max_rel"] for r in rows)}
    _write(a.out, res)


def cmd_quantize(a):
    src = release_path(a.repo, a.revision)
    t0 = time.time()
    model, processor = load_bf16(src)
    _sync()
    print(f"[clef_fp8] bf16 release loaded in {time.time() - t0:.0f} s, GPU {gib(torch.cuda.memory_allocated())} GiB", flush=True)
    report = quantize_clef(model)
    exp = export(model, a.out, src, report, a.repo, a.revision)
    res = {"quantize": {k: v for k, v in report.items() if k not in ("layers", "fp8_modules")}, "export": exp}
    if a.verify_reload:
        # the reloaded model must equal the in-memory quantized one bit for bit, buffers included
        model2, _ = load_fp8(a.out)
        t1 = dict(list(model.named_parameters()) + list(model.named_buffers()))
        t2 = dict(list(model2.named_parameters()) + list(model2.named_buffers()))
        diff = [n for n in t1 if n not in t2 or t1[n].dtype != t2[n].dtype or t1[n].shape != t2[n].shape
                or not torch.equal(t1[n].view(torch.uint8) if t1[n].dtype == torch.float8_e4m3fn else t1[n],
                                   t2[n].view(torch.uint8) if t2[n].dtype == torch.float8_e4m3fn else t2[n])]
        extra = [n for n in t2 if n not in t1]
        res["verify_reload"] = {"tensors": len(t1), "differ": diff[:20], "n_differ": len(diff), "extra": extra[:20], "load": model2.load_stats}
        print(f"[clef_fp8] reload check: {len(t1)} tensors, {len(diff)} differ, {len(extra)} extra", flush=True)
    _write(Path(a.out) / "quantize_report.json", {**res, "layers": report["layers"]})
    if a.report: _write(a.report, res)


def _transfer_records(n):
    from kev.data import api_request
    from kev.suite import load_split
    return [{**api_request(r), "model": REPO} for r in load_split("evals/v9/transfer-v9", "development")[:n]]


def cmd_check(a):
    bodies = _transfer_records(a.n)
    src = release_path(a.repo, a.revision)

    def run(load):
        model, processor = load()
        jsm = sys.modules["joint_schema_model"]
        out, lat = [], []
        for body in bodies:
            _sync(); t = time.perf_counter()
            _, p = probabilities(jsm, model, processor, body)
            _sync(); lat.append(time.perf_counter() - t)
            out.append(p)
        stats = {"allocated_gib": gib(torch.cuda.memory_allocated()), "load": getattr(model, "load_stats", None),
                 "median_ms": round(1000 * sorted(lat)[len(lat) // 2], 1)}
        del model, processor
        gc.collect(); torch.cuda.empty_cache(); _sync()
        return out, stats

    bf, bf_stats = run(lambda: load_bf16(src))
    print(f"[check] bf16 done {bf_stats}", flush=True)
    q8, q8_stats = run(lambda: load_fp8(a.export))
    print(f"[check] fp8 done {q8_stats}", flush=True)
    rows, agree, n = [], 0, 0
    for i, (pb, pq) in enumerate(zip(bf, q8)):
        for qid in pb:
            dp = max(abs(pb[qid][o] - pq[qid][o]) for o in pb[qid])
            ab, aq = max(pb[qid], key=pb[qid].get), max(pq[qid], key=pq[qid].get)
            agree += ab == aq; n += 1
            rows.append({"record": i, "question": qid, "options": len(pb[qid]), "max_abs_dp": dp, "argmax_bf16": ab, "argmax_fp8": aq,
                         "p_bf16": pb[qid], "p_fp8": pq[qid]})
    dps = sorted(r["max_abs_dp"] for r in rows)
    res = {"records": len(bodies), "questions": n, "argmax_agree": agree, "argmax_agreement": round(agree / n, 4),
           "max_abs_dp": dps[-1], "median_abs_dp": dps[len(dps) // 2], "mean_abs_dp": sum(dps) / n,
           "bf16": bf_stats, "fp8": q8_stats, "rows": rows}
    print(f"[check] {n} questions over {len(bodies)} records: argmax agree {agree}/{n}, max |dp| {dps[-1]:.4f}, "
          f"median {dps[len(dps) // 2]:.4f}, mean {sum(dps) / n:.4f}", flush=True)
    _write(a.out, res)


def _synthetic_state(tokenizer, target):
    """A deterministic support-ticket log of at least `target` tokens (state only), plus 3 questions."""
    parts, i = [], 0
    teams = ("billing", "shipping", "accounts", "security", "returns")
    while True:
        chunk = []
        for _ in range(64):
            chunk.append(f"Ticket {i:05d} ({teams[i % 5]}): customer {1000 + 37 * i % 9000} wrote on day {i % 28 + 1} that order "
                         f"#{(7919 * i) % 100000:05d} {'arrived damaged' if i % 3 == 0 else 'was charged twice' if i % 3 == 1 else 'cannot log in'}; "
                         f"agent noted priority {i % 4} and status {'open' if i % 2 else 'closed'}.")
            i += 1
        parts.append("\n".join(chunk))
        text = "\n".join(parts)
        if len(tokenizer(text, add_special_tokens=False).input_ids) >= target: return text


def _memrun(jsm, model, processor, body, label):
    dev = torch.device("cuda", torch.cuda.current_device())
    _sync(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
    base = torch.cuda.memory_allocated(dev)
    t = time.perf_counter()
    try:
        enc, p = probabilities(jsm, model, processor, body)
        _sync()
        res = {"fit": True, "input_tokens": len(enc.input_ids), "answers": {q: max(v, key=v.get) for q, v in p.items()}}
    except torch.OutOfMemoryError as e:
        res = {"fit": False, "error": str(e).splitlines()[0]}
    res.update(seconds=round(time.perf_counter() - t, 2), allocated_before_gib=gib(base),
               peak_allocated_gib=gib(torch.cuda.max_memory_allocated(dev)), peak_reserved_gib=gib(torch.cuda.max_memory_reserved(dev)))
    print(f"[memtest] {label}: {res}", flush=True)
    gc.collect(); torch.cuda.empty_cache()
    return res


def cmd_memtest(a):
    dev = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(dev).total_memory
    frac = a.cap_gib * 2**30 / total
    torch.cuda.set_per_process_memory_fraction(frac, dev)
    print(f"[memtest] cap {a.cap_gib} GiB of {gib(total)} GiB (fraction {frac:.4f})", flush=True)
    res = {"cap_gib": a.cap_gib, "device_total_gib": gib(total), "fraction": frac, "environment": environment()}
    try:
        model, processor = load_fp8(a.export)
    except torch.OutOfMemoryError as e:
        res["load"] = {"fit": False, "error": str(e).splitlines()[0]}
        _write(a.out, res); return
    res["load"] = {"fit": True, **model.load_stats}
    res["weights_gib"] = gib(torch.cuda.memory_allocated(dev))
    jsm = sys.modules["joint_schema_model"]
    tok = processor.tokenizer
    # (a) the longest breadth-v1 development record by encoded tokens (licence-restricted rows: read here, never copied)
    from kev.data import api_request
    try:
        from kev.suite import load_split
        records = load_split("evals/breadth-v1", "development")
        src = "load_split"
    except Exception as e:      # the rebuilt partition may not match the private mirror's hash
        from kev.suite import read_jsonl
        records = read_jsonl(Path("evals/breadth-v1/development.jsonl"))
        src = f"read_jsonl (load_split failed: {type(e).__name__}: {str(e)[:120]})"
    lengths = []
    for i, r in enumerate(records):
        body = {**api_request(r), "model": REPO}
        lengths.append((len(jsm.encode_record(tok, body, max_length=10**9, processor=processor).input_ids), i))
    n_tok, idx = max(lengths)
    res["a_breadth_longest"] = {"source": src, "records": len(records), "record_index": idx, "record_id": records[idx].get("id"),
                                "encoded_tokens": n_tok, **_memrun(jsm, model, processor, {**api_request(records[idx]), "model": REPO}, "a breadth-v1 longest")}
    # (b) synthetic: a 16,384-token state and 3 questions; encode_record trims the state to the server's 16,384-token input
    state = _synthetic_state(tok, a.state_tokens)
    body = {"model": REPO, "state": state, "questions": {
        "damaged": {"type": "noul", "instructions": "Does any ticket report a damaged order?"},
        "team": {"type": "choice", "instructions": "Which team has the most tickets?",
                 "criteria": {"billing": "Payments", "shipping": "Deliveries", "accounts": "Account access"}},
        "load": {"type": "score", "instructions": "How heavy is the support load?", "criteria": ["Light", "Moderate", "Heavy"]}}}
    res["b_synthetic"] = {"state_tokens": len(tok(state, add_special_tokens=False).input_ids),
                          **_memrun(jsm, model, processor, body, "b synthetic 16k")}
    res["fit"] = all(res[k]["fit"] for k in ("a_breadth_longest", "b_synthetic"))
    _write(a.out, res)


def _write(path, obj):
    if not path: return
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n")
    print(f"[clef_fp8] wrote {path}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selftest"); s.add_argument("--out")
    s = sub.add_parser("quantize"); s.add_argument("--out", required=True); s.add_argument("--report")
    s.add_argument("--repo", default=REPO); s.add_argument("--revision", default=REVISION)
    s.add_argument("--verify-reload", type=int, default=1)
    s = sub.add_parser("check"); s.add_argument("--export", required=True); s.add_argument("--out"); s.add_argument("--n", type=int, default=20)
    s.add_argument("--repo", default=REPO); s.add_argument("--revision", default=REVISION)
    s = sub.add_parser("memtest"); s.add_argument("--export", required=True); s.add_argument("--out")
    s.add_argument("--cap-gib", type=float, default=15.0); s.add_argument("--state-tokens", type=int, default=16384)
    a = ap.parse_args(argv)
    {"selftest": cmd_selftest, "quantize": cmd_quantize, "check": cmd_check, "memtest": cmd_memtest}[a.cmd](a)


if __name__ == "__main__":
    main()
