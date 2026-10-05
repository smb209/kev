"""Row-by-row parity of two kev.benchmark reads (rows.json): argmax flips, |dp| and the paired accuracy delta.

Read conditions (pre-registered in spark/LAB_NOTEBOOK.md, E2 / E4): flips <= 1 % and p99 |dp| <= 0.05 per suite ->
equivalent within known kernel drift; 1-2 % -> investigate; > 2 % or an accuracy-delta CI excluding 0 -> not equivalent.

Saved logits are already divided by the row's inference_temperature (softmax(logits) == p). Probabilities are recomputed
at one common temperature (the reference's): raw = logits * T_row, p = softmax(raw / T_ref), so a difference in served
temperature cannot masquerade as drift. When either read has no logits (kev.benchmark --remote saves p only), both
sides are compared on their served probabilities instead and the output says so. Pairs on (id, question); refuses empty or misaligned pairings.

usage: python spark/parity.py --reference runs/a/rows.json --candidate runs/b/rows.json [--json out.json]
"""
import argparse, json
import numpy as np


def load(path, clean_only):
    rows = json.load(open(path, encoding="utf-8"))
    return {(r["id"], r["question"]): r for r in rows if not clean_only or r.get("variant", "clean") == "clean"}


def softmax(z, t):
    z = np.asarray(z, dtype=np.float64) / t
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def parity(ref_path, cand_path, samples=2000, seed=0, clean_only=False):
    ref, cand = load(ref_path, clean_only), load(cand_path, clean_only)
    keys = sorted(ref.keys() & cand.keys())
    assert keys, f"no paired rows between {ref_path} and {cand_path}"
    served = any("logits" not in r for r in (*ref.values(), *cand.values()))   # a --remote read saves p only
    temps = {r.get("inference_temperature") for r in ref.values()}
    assert served or len(temps) == 1, f"reference has several temperatures {temps}"
    t = None if served else temps.pop()
    flips, dps, groups, acc_r, acc_c = [], [], [], [], []
    for k in keys:
        a, b = ref[k], cand[k]
        assert a["keys"] == b["keys"] and a["label"] == b["label"], f"misaligned row {k}"
        if served:   # different models (or a remote read): each side's served probabilities, at its own temperature
            pa, pb = np.asarray(a["p"], dtype=np.float64), np.asarray(b["p"], dtype=np.float64)
        else:
            pa = softmax(np.asarray(a["logits"]) * a["inference_temperature"], t)
            pb = softmax(np.asarray(b["logits"]) * b["inference_temperature"], t)
        flips.append(int(pa.argmax() != pb.argmax()))
        dps.append(float(np.abs(pa - pb).max()))
        groups.append(a["group"])
        acc_r.append(int(pa.argmax() == a["label"]))
        acc_c.append(int(pb.argmax() == b["label"]))
    flips, dps, acc_r, acc_c = map(np.asarray, (flips, dps, acc_r, acc_c))
    # record-clustered paired bootstrap of the accuracy delta (candidate - reference)
    rng = np.random.default_rng(seed)
    uniq = sorted(set(groups))
    index = {g: i for i, g in enumerate(uniq)}
    gid = np.array([index[g] for g in groups])
    diff_by_group = np.bincount(gid, weights=acc_c - acc_r, minlength=len(uniq))
    n_by_group = np.bincount(gid, minlength=len(uniq))
    boots = []
    for _ in range(samples):
        pick = rng.integers(0, len(uniq), len(uniq))
        boots.append(diff_by_group[pick].sum() / n_by_group[pick].sum())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"reference": ref_path, "candidate": cand_path, "temperature": t,
            "paired": len(keys), "unpaired_reference": len(ref) - len(keys), "unpaired_candidate": len(cand) - len(keys),
            "flips": int(flips.sum()), "flip_rate": float(flips.mean()),
            "dp_max": float(dps.max()), "dp_p99": float(np.percentile(dps, 99)), "dp_median": float(np.median(dps)),
            "acc_reference": float(acc_r.mean()), "acc_candidate": float(acc_c.mean()),
            "acc_delta": float(acc_c.mean() - acc_r.mean()), "acc_delta_ci95": [float(lo), float(hi)],
            "groups": len(uniq), "probabilities": "served (each side at its own T)" if served else f"recomputed at T={t}", "rows": "clean only" if clean_only else "all variants"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--json")
    ap.add_argument("--clean_only", action="store_true", help="only variant == clean rows (the accuracy kev.benchmark headlines)")
    a = ap.parse_args()
    out = parity(a.reference, a.candidate, clean_only=a.clean_only)
    print(json.dumps(out, indent=1))
    if a.json: json.dump(out, open(a.json, "w"), indent=1)
