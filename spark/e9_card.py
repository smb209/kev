"""E9: does Cloudflare's Clef card reproduce, per dataset and in direction, on breadth-v1 development?

Pre-registered in spark/LAB_NOTEBOOK.md ("E9 breadth-v1 card comparison"). For each of breadth-v1's 14 datasets and each
card pair (Clef vs Kev 9B, Clef-Flash vs Kev 9B): the card's sign of the gap, then our paired delta on the dataset's own
metric (question accuracy, or case-exact over records where the manifest says so), record-clustered bootstrap (2,000
resamples, seed 0). reproduces = our sign matches the card's and the 95 % CI excludes 0; contradicted = CI excludes 0 with
the opposite sign; unresolved otherwise. Card gaps under 2 points are "ties": tested for no difference only.
Scores are never compared to the card's absolute numbers (breadth-v1 restricts candidate sets).

    python spark/e9_card.py --root runs/spark/e8 [--kev9b kev-9b] [--json runs/spark/e9/card.json]
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Cloudflare/clef README (revision 2f3de3dd), Decision Index 0.2.1 table: Clef, Clef-Flash, Jev, Kev 9B (percent).
CARD = {"bfcl": (98.5, 98.8, 95.8, 94.5), "toolret": (69.2, 66.4, 65.3, 64.3), "apibank": (91.9, 93.1, 88.2, 56.3),
        "clinc150": (97.4, 66.8, 89.3, 79.0), "routerbench": (79.7, 79.9, 79.9, 80.0), "sgd": (43.8, 34.2, 43.0, 64.0),
        "contractnli": (81.4, 84.3, 71.7, 57.8), "humicroedit": (66.7, 75.1, 61.9, 55.8), "cfcolor": (66.0, 65.8, 64.7, 56.3),
        "hellaswag": (98.2, 98.6, 94.5, 81.9), "chessbench": (24.7, 23.0, 17.2, 11.2), "musr": (83.5, 86.0, 66.1, 57.9),
        "sata_bench": (33.8, 36.7, 26.4, 26.7), "bright": (45.9, 39.3, 47.5, 38.5)}
TIE = 2.0


def rows(path):
    data = json.load(open(path, encoding="utf-8"))
    assert data, f"empty {path}"
    return {(r["id"], r["question"]): r for r in data}


def unit_scores(rs, metric):
    """-> {unit: (group, correct)}: per question for accuracy, per record (all its questions right) for case_exact."""
    if metric != "case_exact":
        return {k: (r["group"], int(int(np.argmax(r["p"])) == r["label"])) for k, r in rs.items()}
    by_rec = defaultdict(list)
    for (rid, _), r in rs.items(): by_rec[rid].append((r["group"], int(int(np.argmax(r["p"])) == r["label"])))
    return {rid: (v[0][0], int(all(c for _, c in v))) for rid, v in by_rec.items()}


def paired(a, b, samples=2000, seed=0):
    keys = sorted(a.keys() & b.keys())
    assert keys and len(keys) == len(a) == len(b), f"unpaired units {len(a)} / {len(b)} / {len(keys)}"
    ca = np.array([a[k][1] for k in keys]); cb = np.array([b[k][1] for k in keys])
    g = [a[k][0] for k in keys]; idx = {v: i for i, v in enumerate(sorted(set(g)))}; gid = np.array([idx[v] for v in g])
    d = np.bincount(gid, weights=cb - ca, minlength=len(idx)); n = np.bincount(gid, minlength=len(idx))
    rng = np.random.default_rng(seed)
    boots = [d[p].sum() / n[p].sum() for p in (rng.integers(0, len(idx), len(idx)) for _ in range(samples))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": len(keys), "a": float(ca.mean()), "b": float(cb.mean()), "delta": float(cb.mean() - ca.mean()), "ci": [float(lo), float(hi)]}


def verdict(card_gap, r):
    lo, hi = r["ci"]
    if abs(card_gap) < TIE: return "tie: no difference" if lo <= 0 <= hi else "tie: DIFFERENT"
    if lo > 0 or hi < 0:
        return "reproduces" if (lo > 0) == (card_gap > 0) else "contradicted"
    return "unresolved"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/spark/e8"); ap.add_argument("--kev9b", default="kev-9b"); ap.add_argument("--json")
    a = ap.parse_args()
    manifest = json.load(open(ROOT / "evals/breadth-v1/manifest.json"))
    metric = {k: v["metric"] for k, v in manifest["datasets"].items()}
    systems = {n: rows(Path(a.root) / f"{n}-breadth-v1" / "rows.json") for n in ("clef", "clef-flash", a.kev9b, "kev-27b")}
    out, tally = [], defaultdict(lambda: defaultdict(int))
    print(f"{'dataset':12s} {'metric':10s} | {'pair':22s} {'card gap':>8s} | {'ours (pp)':>9s} {'95% CI':>16s} n    verdict")
    for ds in CARD:
        sub = {n: {k: r for k, r in s.items() if r["source"] == ds} for n, s in systems.items()}
        for col, name in ((0, "clef"), (1, "clef-flash")):
            gap = CARD[ds][col] - CARD[ds][3]
            r = paired(unit_scores(sub[a.kev9b], metric[ds]), unit_scores(sub[name], metric[ds]))
            v = verdict(gap, r); tally[name][v] += 1
            out.append({"dataset": ds, "metric": metric[ds], "pair": f"{name} vs {a.kev9b}", "card_gap": gap, **r, "verdict": v})
            print(f"{ds:12s} {metric[ds]:10s} | {name+' vs '+a.kev9b:22s} {gap:+8.1f} | {r['delta']*100:+9.1f} [{r['ci'][0]*100:+6.1f}, {r['ci'][1]*100:+6.1f}] {r['n']:4d} {v}")
        r = paired(unit_scores(sub["kev-27b"], metric[ds]), unit_scores(sub["clef"], metric[ds]))
        out.append({"dataset": ds, "metric": metric[ds], "pair": "clef vs kev-27b", **r})
    print("tally:", {k: dict(v) for k, v in tally.items()})
    if a.json: json.dump({"rows": out, "tally": {k: dict(v) for k, v in tally.items()}}, open(a.json, "w"), indent=1)
