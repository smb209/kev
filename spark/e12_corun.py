"""E12: Qwen3-Embedding-4B (vLLM) and text-only FP8 Clef-Flash (clef_server) loaded together; peak GPU memory under load.

Pre-registered in spark/LAB_NOTEBOOK.md ("E12 co-residency"). Pass = peak of the summed per-process GPU memory
(nvidia-smi used_memory, CUDA context included) <= 15.0 GiB. Both servers must already be running; this script drives
interleaved 8k-token load against both for --seconds while sampling nvidia-smi every second, then reports per-process
and summed peaks. Standard library only.

    python3 spark/e12_corun.py --embed http://127.0.0.1:8051 --clef http://127.0.0.1:8031 --seconds 180 --out runs/spark/e12/corun.json
"""
import argparse, json, subprocess, threading, time, urllib.request

PARA = ("The quarterly report covers revenue, customer retention and infrastructure spending across all regions. "
        "Support tickets about billing errors rose after the pricing change, while shipping delays fell. ")


def text_of(tokens):   # ~4.3 characters per token for this prose; long enough that the server truncates to its limit
    return (PARA * (tokens * 43 // (10 * len(PARA)) + 1))


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def sample(stop, out):
    while not stop.is_set():
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True).stdout
        procs = {int(p): int(m) for p, m in (l.split(", ") for l in q.strip().splitlines() if l.strip())}
        out.append({"t": time.time(), "procs": procs, "sum_mib": sum(procs.values())})
        time.sleep(1)


def loop(name, fn, stop, stats):
    while not stop.is_set():
        t0 = time.time()
        try:
            fn(); stats[name]["ok"] += 1; stats[name]["lat"].append(time.time() - t0)
        except Exception as e:
            stats[name]["err"] += 1; stats[name]["last_error"] = str(e)[:300]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", required=True); ap.add_argument("--clef", required=True)
    ap.add_argument("--embed-model", default="Qwen/Qwen3-Embedding-4B")
    ap.add_argument("--tokens", type=int, default=8192); ap.add_argument("--seconds", type=int, default=180)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base = text_of(a.tokens + 800)
    seq = iter(range(10**9)); seen = {"embed_prompt_tokens": []}
    def unique():   # a distinct document each call: a leading counter changes every token's context, so no prefix cache can reuse a prefill
        n = next(seq); return f"Document {n}: case {n * 7919 % 104729} of batch {n % 97}. " + base
    def embed():
        r = post(f"{a.embed}/v1/embeddings", {"model": a.embed_model, "input": [unique()], "truncate_prompt_tokens": a.tokens})
        seen["embed_prompt_tokens"].append((r.get("usage") or {}).get("prompt_tokens"))
    doc = base
    clef = lambda: post(f"{a.clef}/v1/systemone", {"model": "clef", "state": unique(), "questions": {
        "billing": {"type": "noul", "instructions": "Is this about a billing problem?"},
        "team": {"type": "choice", "instructions": "Which team should handle it?", "criteria": {"billing": "Payments", "shipping": "Deliveries", "infra": "Infrastructure"}},
        "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["Routine", "Urgent", "Emergency"]}}})
    idle = []; ev = threading.Event(); t = threading.Thread(target=sample, args=(ev, idle)); t.start(); time.sleep(5); ev.set(); t.join()
    embed(); clef()   # warm both once (compiles, first-shape allocations) before the measured window
    stop = threading.Event(); samples = []; stats = {n: {"ok": 0, "err": 0, "lat": []} for n in ("embed", "clef")}
    threads = [threading.Thread(target=sample, args=(stop, samples)),
               threading.Thread(target=loop, args=("embed", embed, stop, stats)),
               threading.Thread(target=loop, args=("clef", clef, stop, stats))]
    for th in threads: th.start()
    time.sleep(a.seconds); stop.set()
    for th in threads: th.join()
    assert samples and stats["embed"]["ok"] and stats["clef"]["ok"], f"no load ran: {stats}"
    pids = sorted({p for s in samples for p in s["procs"]})
    peak = {str(p): max(s["procs"].get(p, 0) for s in samples) for p in pids}
    peak_sum = max(s["sum_mib"] for s in samples)
    for n in stats:
        lat = sorted(stats[n].pop("lat")); stats[n]["median_s"] = lat[len(lat) // 2]; stats[n]["p95_s"] = lat[int(len(lat) * 0.95) - 1]
    pt = [t for t in seen["embed_prompt_tokens"] if t]
    assert pt, "embedding responses carried no usage.prompt_tokens"
    out = {"tokens": a.tokens, "seconds": a.seconds, "embed_prompt_tokens": {"min": min(pt), "max": max(pt), "n": len(pt)},
           "embed_tokens_per_s": round(sum(pt) / a.seconds), "idle_sum_mib": max(s["sum_mib"] for s in idle),
           "idle_procs": idle[-1]["procs"], "peak_per_pid_mib": peak, "peak_sum_mib": peak_sum,
           "peak_sum_gib": round(peak_sum / 1024, 2), "pass_15_gib": peak_sum / 1024 <= 15.0, "load": stats}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))
