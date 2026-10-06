"""E14 read: Clef-Flash micro-batching correctness and throughput against a running clef_server.

Pre-registered in spark/LAB_NOTEBOOK.md ("E14 Clef-Flash micro-batching").
- correct: the same records (transfer-v9 development, unknowable excluded) sent at concurrency 1 (every batch_size must
  be 1) and at concurrency 8 (most batch_size must be > 1, else the run is invalid and nothing is scored). Pass =
  argmax agreement >= 99 % and max |dp| <= 0.05 over all questions.
- speed: requests/s and p50 / p95 latency at concurrency 1 / 4 / 8 / 16 on the same short states.

    python spark/clef_batch_check.py correct --url http://127.0.0.1:8035 --records 200 --out runs/spark/e14/correct.json
    python spark/clef_batch_check.py speed --url http://127.0.0.1:8035 --records 100 --label batched --out runs/spark/e14/speed-batched.json
"""
import argparse, json, sys, threading, time, urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.data import api_request
from kev.suite import load_split


def requests_from_suite(n):
    recs = [r for r in load_split("evals/v9/transfer-v9", "development") if r.get("source") != "unknowable"][:n]
    assert len(recs) == n, f"only {len(recs)} records"
    return [{**api_request(r), "model": "clef"} for r in recs]


def post(url, body):
    req = urllib.request.Request(f"{url}/v1/systemone", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as f:
        out = json.loads(f.read())
    out["_wall_s"] = time.time() - t0
    return out


def run(url, bodies, conc):
    results = [None] * len(bodies)
    nxt = iter(range(len(bodies))); lock = threading.Lock()

    def worker():
        while True:
            with lock:
                i = next(nxt, None)
            if i is None: return
            results[i] = post(url, bodies[i])

    t0 = time.time()
    ths = [threading.Thread(target=worker) for _ in range(conc)]
    for t in ths: t.start()
    for t in ths: t.join()
    return results, time.time() - t0


def probs(answer):
    if answer["type"] == "noul": return np.array([answer["noul"], 1 - answer["noul"]])
    p = answer["probabilities"]; return np.array([p[k] for k in sorted(p)])


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
c = sub.add_parser("correct"); c.add_argument("--url", required=True); c.add_argument("--records", type=int, default=200); c.add_argument("--out", required=True)
s = sub.add_parser("speed"); s.add_argument("--url", required=True); s.add_argument("--records", type=int, default=100)
s.add_argument("--label", required=True); s.add_argument("--out", required=True)
a = ap.parse_args()
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
bodies = requests_from_suite(a.records)
post(a.url, bodies[0])   # warm (first request compiles kernels)

if a.cmd == "correct":
    single, _ = run(a.url, bodies, 1)
    batched, _ = run(a.url, bodies, 8)
    sizes1 = [r["batch_size"] for r in single]; sizes8 = [r["batch_size"] for r in batched]
    assert set(sizes1) == {1}, f"concurrency 1 produced batch sizes {sorted(set(sizes1))}"
    frac = np.mean([b > 1 for b in sizes8])
    assert frac >= 0.5, f"only {frac:.0%} of concurrency-8 responses were batched: run invalid"
    flips = dps = 0; dp_all = []; agree = []
    for r1, r8 in zip(single, batched):
        assert r1["answers"].keys() == r8["answers"].keys()
        for q in r1["answers"]:
            p1, p8 = probs(r1["answers"][q]), probs(r8["answers"][q])
            agree.append(int(p1.argmax() == p8.argmax())); dp_all.append(float(np.abs(p1 - p8).max()))
    agree, dp_all = np.array(agree), np.array(dp_all)
    res = {"records": len(bodies), "questions": len(agree), "batched_fraction": float(frac),
           "batch_size_hist": {str(k): int(v) for k, v in zip(*np.unique(sizes8, return_counts=True))},
           "argmax_agreement": float(agree.mean()), "flips": int((1 - agree).sum()),
           "dp_max": float(dp_all.max()), "dp_p99": float(np.percentile(dp_all, 99)), "dp_median": float(np.median(dp_all))}
    res["pass"] = res["argmax_agreement"] >= 0.99 and res["dp_max"] <= 0.05
else:
    res = {"label": a.label, "records": len(bodies), "by_concurrency": {}}
    for conc in (1, 4, 8, 16):
        out, el = run(a.url, bodies, conc)
        lat = np.array([r["_wall_s"] for r in out]) * 1000
        res["by_concurrency"][str(conc)] = {"req_per_s": round(len(out) / el, 2), "p50_ms": round(float(np.percentile(lat, 50)), 1),
                                           "p95_ms": round(float(np.percentile(lat, 95)), 1),
                                           "mean_batch_size": round(float(np.mean([r.get("batch_size", 1) for r in out])), 2),
                                           "mean_input_tokens": round(float(np.mean([r["usage"]["input_tokens"] for r in out])), 0)}
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps(res, indent=1))
