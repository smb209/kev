"""E4: one quantization arm of Kev-27B v2 read on every E4 suite, the model loaded and quantized once.

Pre-registered read conditions: spark/LAB_NOTEBOOK.md, "E4 quantization". Each arm (bf16 | fp8 | nvfp4-mlp | nvfp4) runs
in the kev-spark-quant image through kev_quant's loader wrap on kev.benchmark's own path (unfused, eager: the benchmark
defaults), so the bf16 arm is the same-environment baseline the others are paired against (spark/parity.py). NVFP4 arms
use static activation scales calibrated on decision-v7's calibration partition (never the development rows read here).

    python spark/e4_reads.py <scheme> [--run jaredpalmer/kev-27b] [--out_root runs/spark/e4]
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import kev_quant as kq
import kev.benchmark
from kev.checkpoint import Checkpoint

SUITES = ("evals/external/semif-v1", "evals/v4/transfer-v4", "evals/v7/decision-v7")

ap = argparse.ArgumentParser()
ap.add_argument("scheme", choices=kq.SCHEMES)
ap.add_argument("--run", default="jaredpalmer/kev-27b")
ap.add_argument("--out_root", default="runs/spark/e4")
ap.add_argument("--fused", type=int, default=0, help="1 = kev.fused_qwen35 kernels (the serving path, E4b); CUDA graphs stay off (serve-only)")
a = ap.parse_args()

kq.Nvfp4Linear.backend = "cutlass"
kq.set_act_quant("torch")
kq.install(a.scheme, (), bool(a.fused), False, "static", 4.0, 64, None)   # E4: unfused benchmark path; E4b: fused
loaded = {}
wrapped = Checkpoint._load_torch


def load_once(self, tok, device, opts):
    key = (str(self.path), str(device))
    if key not in loaded: loaded[key] = wrapped(self, tok, device, opts)
    return loaded[key]


Checkpoint._load_torch = load_once
name = a.run.split("/")[-1]
for suite in SUITES:
    out = Path(a.out_root) / f"{name}-{a.scheme}{'-fused' if a.fused else ''}-{Path(suite).name}"
    sys.argv = ["kev.benchmark", "--run", a.run, "--suite", suite, "--out", str(out)]
    t0 = time.time()
    kev.benchmark.main()
    wall = round(time.time() - t0, 1)
    report = json.loads((out / "report.json").read_text())
    kq.write_quant_json(out / "quant.json", a.scheme, {
        "entry": "spark/e4_reads.py", "suite": suite, "wall_seconds": wall,
        "gpu_peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3), "meminfo_end": kq.meminfo_gib(),
        "clean_accuracy": report["clean"].get("acc"), "latency_ms": report.get("latency_ms")})
    print(f"E4 {a.scheme} {Path(suite).name}: clean acc {report['clean'].get('acc'):.4f} wall {wall}s", flush=True)
assert len(loaded) == 1, f"expected one load, got {len(loaded)}"
print(f"E4DONE {a.scheme}", flush=True)
