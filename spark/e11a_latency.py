"""E11a: time warm /v1/systemone requests against a clef_server (server-side latency_ms: median / p95).

    python spark/e11a_latency.py --url http://127.0.0.1:8031 --req spark/req1.json -n 30
    python spark/e11a_latency.py --url ... --state-tokens 4096 --tokenizer runs/spark/c10-clef-flash-fp8 -n 30
"""
import argparse, json, statistics, sys, urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8031")
ap.add_argument("--req"); ap.add_argument("--state-tokens", type=int); ap.add_argument("--tokenizer")
ap.add_argument("-n", type=int, default=30); ap.add_argument("--warmup", type=int, default=3); ap.add_argument("--out")
a = ap.parse_args()
if a.state_tokens:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from transformers import AutoTokenizer
    import clef_fp8
    state = clef_fp8._synthetic_state(AutoTokenizer.from_pretrained(a.tokenizer), a.state_tokens)
    body = {"model": clef_fp8.REPO, "state": state, "questions": {
        "damaged": {"type": "noul", "instructions": "Does any ticket report a damaged order?"},
        "team": {"type": "choice", "instructions": "Which team has the most tickets?",
                 "criteria": {"billing": "Payments", "shipping": "Deliveries", "accounts": "Account access"}}}}
else:
    body = json.loads(Path(a.req).read_text())


def post(b):
    r = urllib.request.Request(a.url + "/v1/systemone", json.dumps(b).encode(), {"content-type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=600).read())


for _ in range(a.warmup): post(body)
lat, tokens = [], None
for _ in range(a.n):
    r = post(body); lat.append(r["latency_ms"]); tokens = r["usage"]["input_tokens"]
lat.sort()
res = {"input_tokens": tokens, "n": a.n, "median_ms": round(statistics.median(lat), 1), "p95_ms": round(lat[int(0.95 * (a.n - 1) + 0.5)], 1),
       "min_ms": round(lat[0], 1), "max_ms": round(lat[-1], 1)}
print(json.dumps(res))
if a.out: Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
