"""E4 read-out: each quantized arm of Kev-27B v2 against the same-environment bf16 arm, pooled over the E4 suites.

Pre-registered (spark/LAB_NOTEBOOK.md, "E4 quantization"): verdict = pooled accuracy delta vs same-env bf16 (all
variants, as E2). Serving-grade = point >= -1.0 pp AND 95 % CI lower bound >= -2.0 pp AND pooled ECE increase <= 0.01.
Context: per-suite deltas, flips, |dp| (spark/parity.py). CI = record-clustered paired bootstrap, 2,000 resamples, seed 0,
clusters keyed by (suite, group). Probabilities as saved (both arms served at the checkpoint's own T, asserted equal).

    python spark/e4_readout.py [--root runs/spark/e4] [--json runs/spark/e4/readout.json]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kev.metrics import ece
from parity import parity

SUFFIX = ""   # "-fused" for E4b
SUITES = ("semif-v1", "transfer-v4", "decision-v7")
ARMS = ("fp8", "nvfp4-mlp", "nvfp4")


def rows(root, arm, suite):
    path = Path(root) / f"kev-27b-{arm}{SUFFIX}-{suite}" / "rows.json"
    data = json.load(open(path, encoding="utf-8"))
    assert data, f"empty {path}"
    return {(suite, r["id"], r["question"]): r for r in data}


def pooled(root, arm, samples=2000, seed=0):
    ref, cand = {}, {}
    for s in SUITES:
        ref.update(rows(root, "bf16", s)); cand.update(rows(root, arm, s))
    keys = sorted(ref.keys() & cand.keys())
    assert len(keys) == len(ref) == len(cand), f"{arm}: unpaired rows ({len(ref)} vs {len(cand)}, {len(keys)} paired)"
    ok_r, ok_c, conf_r, conf_c, groups = [], [], [], [], []
    for k in keys:
        a, b = ref[k], cand[k]
        assert a["keys"] == b["keys"] and a["label"] == b["label"] and a["inference_temperature"] == b["inference_temperature"], k
        pa, pb = np.asarray(a["p"]), np.asarray(b["p"])
        ok_r.append(int(pa.argmax() == a["label"])); ok_c.append(int(pb.argmax() == b["label"]))
        conf_r.append(float(pa.max())); conf_c.append(float(pb.max()))
        groups.append((k[0], a["group"]))
    ok_r, ok_c = np.asarray(ok_r), np.asarray(ok_c)
    uniq = {g: i for i, g in enumerate(sorted(set(groups)))}
    gid = np.array([uniq[g] for g in groups])
    d = np.bincount(gid, weights=ok_c - ok_r, minlength=len(uniq)); n = np.bincount(gid, minlength=len(uniq))
    rng = np.random.default_rng(seed)
    boots = [(d[p].sum() / n[p].sum()) for p in (rng.integers(0, len(uniq), len(uniq)) for _ in range(samples))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    delta = float(ok_c.mean() - ok_r.mean())
    ece_r, ece_c = ece(np.asarray(conf_r), ok_r), ece(np.asarray(conf_c), ok_c)
    verdict = delta >= -0.010 and lo >= -0.020 and (ece_c - ece_r) <= 0.010
    return {"arm": arm, "questions": len(keys), "clusters": len(uniq),
            "acc_bf16": float(ok_r.mean()), "acc_arm": float(ok_c.mean()), "delta": delta, "ci95": [float(lo), float(hi)],
            "ece_bf16": float(ece_r), "ece_arm": float(ece_c), "ece_delta": float(ece_c - ece_r),
            "discordant": {"arm_right_bf16_wrong": int(((ok_c == 1) & (ok_r == 0)).sum()), "arm_wrong_bf16_right": int(((ok_c == 0) & (ok_r == 1)).sum())},
            "serving_grade": bool(verdict)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/spark/e4"); ap.add_argument("--json"); ap.add_argument("--fused", action="store_true")
    a = ap.parse_args()
    if a.fused: SUFFIX = "-fused"; ARMS = ("nvfp4",)
    out = {"pooled": [], "per_suite": []}
    for arm in ARMS:
        if not all((Path(a.root) / f"kev-27b-{arm}{SUFFIX}-{s}" / "rows.json").exists() for s in SUITES):
            print(f"{arm}: not complete, skipped"); continue
        p = pooled(a.root, arm); out["pooled"].append(p)
        print(f"{arm:10s} pooled n={p['questions']} acc {p['acc_bf16']:.4f} -> {p['acc_arm']:.4f} delta {p['delta']*100:+.2f} pp "
              f"CI [{p['ci95'][0]*100:+.2f}, {p['ci95'][1]*100:+.2f}]  ECE {p['ece_bf16']:.4f} -> {p['ece_arm']:.4f}  "
              f"discordant +{p['discordant']['arm_right_bf16_wrong']}/-{p['discordant']['arm_wrong_bf16_right']}  "
              f"SERVING-GRADE={p['serving_grade']}")
        for s in SUITES:
            r = parity(str(Path(a.root) / f"kev-27b-bf16{SUFFIX}-{s}" / "rows.json"), str(Path(a.root) / f"kev-27b-{arm}{SUFFIX}-{s}" / "rows.json"))
            out["per_suite"].append({"arm": arm, "suite": s, **{k: r[k] for k in ("paired", "flips", "flip_rate", "dp_max", "dp_p99", "acc_delta", "acc_delta_ci95")}})
            print(f"    {s:12s} flips {r['flips']:3d} ({r['flip_rate']:.2%}) dp p99 {r['dp_p99']:.3f} max {r['dp_max']:.3f} "
                  f"delta {r['acc_delta']*100:+.2f} pp CI [{r['acc_delta_ci95'][0]*100:+.2f}, {r['acc_delta_ci95'][1]*100:+.2f}]")
    if a.json: json.dump(out, open(a.json, "w"), indent=1)
