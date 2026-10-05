"""E10 read-out: FP8 Clef-Flash (export) vs bf16 Clef-Flash, pooled over transfer-v9 / breadth-v1 / decision-v7 dev.

Pre-registered (spark/LAB_NOTEBOOK.md, "E10 FP8 Clef-Flash"): primary = pooled accuracy delta FP8 - bf16,
record-clustered bootstrap (2,000 resamples, seed 0; clusters keyed by (suite, group)); claim = "no loss beyond X pp",
X = -(CI lower bound). transfer-v9's accuracy set excludes `unknowable` rows (kev/benchmark.py; the E8 correction).
Also: the bf16 control (transfer-v9 re-read on the same machine) must reproduce the E8 rows.

    python spark/e10_readout.py [--json runs/spark/e10/readout.json]
"""
import argparse, json
import numpy as np

BF16 = {"transfer-v9": "runs/spark/e8/clef-flash-transfer-v9", "breadth-v1": "runs/spark/e8/clef-flash-breadth-v1",
        "decision-v7": "runs/spark/e8/clef-flash-decision-v7"}
FP8 = {s: f"runs/spark/e10/clef-flash-fp8-{s}" for s in BF16}
CONTROL = ("runs/spark/e8/clef-flash-transfer-v9", "runs/spark/e10/clef-flash-bf16ctl-transfer-v9")


def rows(d):
    data = json.load(open(f"{d}/rows.json", encoding="utf-8"))
    assert data, f"empty {d}"
    return {(r["id"], r["question"]): r for r in data if r["source"] != "unknowable"}


def pick(r): return int(np.argmax(r["p"]))


def compare(pairs, samples=2000, seed=0):
    ok_a, ok_b, flips, dps, groups, conf_a, conf_b = [], [], [], [], [], [], []
    for suite, a, b in pairs:
        assert a.keys() == b.keys(), f"{suite}: row sets differ ({len(a)} vs {len(b)})"
        for k in a:
            x, y = a[k], b[k]
            assert x["keys"] == y["keys"] and x["label"] == y["label"], (suite, k)
            pa, pb = np.asarray(x["p"]), np.asarray(y["p"])
            ok_a.append(int(pick(x) == x["label"])); ok_b.append(int(pick(y) == y["label"]))
            flips.append(int(pick(x) != pick(y))); dps.append(float(np.abs(pa - pb).max()))
            groups.append((suite, x["group"])); conf_a.append(pa.max()); conf_b.append(pb.max())
    ok_a, ok_b = np.array(ok_a), np.array(ok_b)
    idx = {g: i for i, g in enumerate(sorted(set(groups)))}; gid = np.array([idx[g] for g in groups])
    d = np.bincount(gid, weights=ok_b - ok_a, minlength=len(idx)); n = np.bincount(gid, minlength=len(idx))
    rng = np.random.default_rng(seed)
    boots = [d[p].sum() / n[p].sum() for p in (rng.integers(0, len(idx), len(idx)) for _ in range(samples))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": len(ok_a), "clusters": len(idx), "acc_bf16": float(ok_a.mean()), "acc_fp8": float(ok_b.mean()),
            "delta": float(ok_b.mean() - ok_a.mean()), "ci95": [float(lo), float(hi)],
            "discordant": [int(((ok_b == 1) & (ok_a == 0)).sum()), int(((ok_b == 0) & (ok_a == 1)).sum())],
            "flips": int(sum(flips)), "dp_max": float(max(dps)), "dp_p99": float(np.percentile(dps, 99)),
            "dp_median": float(np.median(dps))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); a = ap.parse_args()
    ctl = compare([("transfer-v9", rows(CONTROL[0]), rows(CONTROL[1]))])
    print(f"bf16 control (E8 rows vs re-read): flips {ctl['flips']} / {ctl['n']}, dp max {ctl['dp_max']:.2e}")
    per = {s: compare([(s, rows(BF16[s]), rows(FP8[s]))]) for s in BF16}
    pooled = compare([(s, rows(BF16[s]), rows(FP8[s])) for s in BF16])
    for s, r in [*per.items(), ("POOLED", pooled)]:
        print(f"{s:12s} n={r['n']:5d} acc {r['acc_bf16']:.4f} -> {r['acc_fp8']:.4f} delta {r['delta']*100:+.2f} pp "
              f"CI [{r['ci95'][0]*100:+.2f}, {r['ci95'][1]*100:+.2f}] flips {r['flips']} ({r['flips']/r['n']:.2%}) "
              f"disc +{r['discordant'][0]}/-{r['discordant'][1]} dp p99 {r['dp_p99']:.3f} max {r['dp_max']:.3f}")
    print(f"claim: no loss beyond {max(0.0, -pooled['ci95'][0]) * 100:.2f} pp (pooled lower bound)")
    if a.json: json.dump({"control": ctl, "per_suite": per, "pooled": pooled}, open(a.json, "w"), indent=1)
