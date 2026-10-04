"""Per-scheme latency of one projection at Kev-27B (Qwen3.8-27B) shapes, as kev.fused_qwen35 serves them, on this GPU.

Shapes come from the base's config.json (Qwen/Qwen3.8-27B rev 1d4bf0f2, only that file downloaded). For every shape and
M (tokens in the pass) it times, as CUDA-graph replays (the served path: no launch overhead; GPU time only):
  bf16                 F.linear
  fp8 gemm / fp8       torch._scaled_mm rowwise alone / with the per-token activation quantization (eager ops, or
                       torch.compile'd with --compile)
  nvfp4 gemm / nvfp4   FlashInfer mm_fp4 alone / with the activation amax + fp4_quantize, per backend
and sums them over the 64 layers (48 DeltaNet, 16 attention, 64 MLP) into the projection time of one forward pass.

    spark/quant_run.sh mb python3 spark/quant_microbench.py --out runs/spark/quant/microbench.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kev_quant as kq   # noqa: E402

FP8_MAX, FP4_MAX = 448.0, 6.0


def shapes_27b(config_path):
    c = json.loads(Path(config_path).read_text())
    t = c.get("text_config", c)
    H, I = t["hidden_size"], t["intermediate_size"]
    nk, dk, nv, dv = t["linear_num_key_heads"], t["linear_key_head_dim"], t["linear_num_value_heads"], t["linear_value_head_dim"]
    heads, kv, hd = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
    layers = t["num_hidden_layers"]; attn = layers // t["full_attention_interval"]; delta = layers - attn
    # (name, N, K, how many per forward pass); the DeltaNet a/b rows (nv each) stay bf16 (SplitLinear tail), timed separately
    return t, [("deltanet.in_proj[qkv+z]", 2 * nk * dk + 2 * nv * dv, H, delta),
               ("deltanet.in_proj[b+a] (bf16 tail)", 2 * nv, H, delta),
               ("deltanet.out_proj", H, nv * dv, delta),
               ("attn.qkv", 2 * heads * hd + 2 * kv * hd, H, attn),
               ("attn.o_proj", H, heads * hd, attn),
               ("mlp.gate_up", 2 * I, H, layers),
               ("mlp.down_proj", H, I, layers)]


def graph_time(fn, iters=20, reps=5):
    """Median ms per call of fn() replayed from a CUDA graph holding `iters` calls."""
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for _ in range(iters): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g.replay(); torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize()
        times.append(a.elapsed_time(b) / iters)
    del g
    return sorted(times)[len(times) // 2]


@torch.no_grad()
def bench(N, K, M, backends, compiled):
    from flashinfer import fp4_quantize, mm_fp4
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    flop = 2 * M * N * K
    row = {"N": N, "K": K, "M": M}
    row["bf16"] = graph_time(lambda: F.linear(x, w))
    if min(N, K) < kq.MIN_DIM: return row
    f8 = kq.Fp8Linear(w)
    xq, xs = kq._fp8_quant_rows(x)
    row["fp8_gemm"] = graph_time(lambda: torch._scaled_mm(xq, f8.qweight.t(), scale_a=xs, scale_b=f8.wscale, out_dtype=torch.bfloat16))
    row["fp8_quant"] = graph_time(lambda: kq._fp8_quant_rows(x))
    row["fp8"] = graph_time(lambda: f8(x))
    if compiled is not None:
        kq._fp8_quant = compiled
        row["fp8_quant_compiled"] = graph_time(lambda: compiled(x))
        row["fp8_compiled"] = graph_time(lambda: f8(x))
        kq._fp8_quant = kq._fp8_quant_rows
    f4 = kq.Nvfp4Linear(w)
    ga = (FP8_MAX * FP4_MAX / x.float().abs().amax()).reshape(1)
    aq, asf = fp4_quantize(x, ga, 16, False, True)
    alpha = 1.0 / (ga * f4.wgs)
    row["nvfp4_quant"] = graph_time(lambda: fp4_quantize(x, (FP8_MAX * FP4_MAX) / torch.linalg.vector_norm(x, float("inf")).float().reshape(1), 16, False, True))
    for be in backends:
        try:
            row[f"nvfp4_gemm[{be}]"] = graph_time(lambda: mm_fp4(aq, f4.qweight.T, asf, f4.wsf.T, alpha, torch.bfloat16, backend=be))
            kq.Nvfp4Linear.backend = be
            row[f"nvfp4[{be}]"] = graph_time(lambda: f4(x))
        except Exception as e:
            row[f"nvfp4[{be}]"] = f"{type(e).__name__}: {str(e)[:120]}"
    kq.Nvfp4Linear.backend = "cutlass"
    row["tflops"] = {k: round(flop / (v * 1e-3) / 1e12, 1) for k, v in row.items() if isinstance(v, float)}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Qwen3.8-27B config.json (default: the HF cache snapshot of rev 1d4bf0f2)")
    ap.add_argument("--M", default="16,128,512,2048")
    ap.add_argument("--backends", default="cutlass,b12x")
    ap.add_argument("--compile", action="store_true", help="also time the torch.compile'd FP8 activation quantization")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.config is None:
        from huggingface_hub import hf_hub_download
        a.config = hf_hub_download("Qwen/Qwen3.8-27B", "config.json", revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
    text, shapes = shapes_27b(a.config)
    compiled = torch.compile(kq._fp8_quant_rows, dynamic=True, fullgraph=True) if a.compile else None
    backends = a.backends.split(",")
    rows = []
    for M in map(int, a.M.split(",")):
        for name, N, K, count in shapes:
            r = bench(N, K, M, backends, compiled)
            r.update(name=name, per_pass=count)
            rows.append(r)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    # one forward pass's projection time per scheme: quantized shapes use the scheme's kernel, the bf16 tail stays bf16
    schemes = {"bf16": lambda r, mlp: r["bf16"],
               "fp8": lambda r, mlp: r.get("fp8", r["bf16"]),
               **{f"nvfp4[{be}]": (lambda be: lambda r, mlp: r.get(f"nvfp4[{be}]", r["bf16"]) if isinstance(r.get(f"nvfp4[{be}]"), float) else r["bf16"])(be) for be in backends},
               **{f"nvfp4-mlp[{be}]": (lambda be: lambda r, mlp: (r.get(f"nvfp4[{be}]") if isinstance(r.get(f"nvfp4[{be}]"), float) else r["bf16"]) if mlp else r.get("fp8", r["bf16"]))(be) for be in backends}}
    totals = {}
    for M in map(int, a.M.split(",")):
        at = [r for r in rows if r["M"] == M]
        totals[M] = {s: round(sum(f(r, r["name"].startswith("mlp")) * r["per_pass"] for r in at), 3) for s, f in schemes.items()}
    print("\nprojection GPU ms per Kev-27B forward pass (64 layers), CUDA-graph replay:")
    print(f"{'M':>6s} " + " ".join(f"{s:>18s}" for s in schemes))
    for M, t in totals.items(): print(f"{M:6d} " + " ".join(f"{t[s]:18.3f}" for s in schemes))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"config": a.config, "environment": kq.environment(), "shapes": shapes, "rows": rows, "per_pass_ms": totals}, indent=2) + "\n")


if __name__ == "__main__":
    main()
