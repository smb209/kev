"""Warm latency of a running kev.serve (spark/kev_quant.py serve) on spark/req1.json: 10 repeats of the same request (the
state prefix cached after the first) and 10 requests whose state differs (a new state every time: state pass + rows).
Warms both patterns first and waits for the lazily captured CUDA graphs. Stdlib only.
    python3 spark/quant_serve_client.py --url http://127.0.0.1:8019 --out runs/spark/quant/serve-....json"""
import argparse, json, statistics, time, urllib.request
from pathlib import Path


def post(url, body):
    req = urllib.request.Request(f"{url}/v1/systemone", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read())
    return out, 1000 * (time.perf_counter() - t)


def models(url):
    with urllib.request.urlopen(f"{url}/v1/models", timeout=30) as r:
        return json.loads(r.read())["models"][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8019")
    ap.add_argument("--req", default="spark/req1.json")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out")
    a = ap.parse_args()
    base = json.loads(Path(a.req).read_text())
    for _ in range(600):   # wait for the server
        try: models(a.url); break
        except Exception: time.sleep(2)
    first, first_ms = post(a.url, base)
    def varied(i): return {**base, "state": base["state"] + f" Reference {i:04d}."}
    for i in range(6): post(a.url, base); post(a.url, varied(9000 + i))   # warm-up: both patterns, buckets captured when idle
    time.sleep(8)
    hit = [post(a.url, base) for _ in range(a.repeats)]
    time.sleep(3)
    miss = [post(a.url, varied(i)) for i in range(a.repeats)]
    card = models(a.url)
    def stats(rs):
        wall = [ms for _, ms in rs]; model = [r["latency_ms"] for r, _ in rs]
        return {"wall_ms_median": round(statistics.median(wall), 2), "wall_ms_min": round(min(wall), 2), "wall_ms_max": round(max(wall), 2),
                "model_ms_median": round(statistics.median(model), 2), "wall_ms": [round(x, 2) for x in wall]}
    res = {"first_request_ms": round(first_ms, 1), "answers": hit[-1][0]["answers"], "usage": hit[-1][0]["usage"],
           "same_request": stats(hit), "new_state": stats(miss), "server": {k: card.get(k) for k in ("run", "dtype", "backend", "cuda_graphs", "prefix_cache", "batches")}}
    print(json.dumps(res, indent=2))
    if a.out: Path(a.out).write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
