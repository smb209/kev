"""Short-document embedding throughput against a live OpenAI-compatible embeddings server (fooddb-shaped load).

Each input is a unique ~460-character food document (a leading counter keeps vLLM's prefix cache from reusing work).
Runs three patterns for a fixed time each and prints docs/s, tokens/s (from usage.prompt_tokens) and request latency.
Standard library only.

    python3 embed_throughput.py --url http://127.0.0.1:8001/v1/embeddings --model qwen3-embedding-4b --dims 2560
"""
import argparse, json, random, threading, time, urllib.request

WORDS = ("apple oat milk almond tofu lentil spinach yogurt honey rice barley wheat gluten soy peanut sesame egg "
         "cheddar salmon tuna chicken beef pork turkey tomato basil garlic onion pepper vanilla cocoa sugar salt "
         "protein fiber calcium iron vitamin sodium organic vegan kosher halal dairy-free nut-free roasted raw").split()

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True); ap.add_argument("--model", required=True); ap.add_argument("--dims", type=int, required=True)
ap.add_argument("--seconds", type=int, default=20)
a = ap.parse_args()
random.seed(0)
counter = [0]
counter_lock = threading.Lock()


def doc():
    with counter_lock:
        counter[0] += 1
        i = counter[0]
    return ("Food %d: " % i + " ".join(random.choice(WORDS) for _ in range(70)))[:460]


def post(inputs):
    req = urllib.request.Request(a.url, data=json.dumps({"model": a.model, "input": inputs}).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as f:
        d = json.loads(f.read())
    assert len(d["data"]) == len(inputs) and len(d["data"][0]["embedding"]) == a.dims, "bad response shape"
    return d["usage"]["prompt_tokens"]


def run(label, batch, conc):
    stop = time.time() + a.seconds
    tot = {"docs": 0, "tok": 0, "req": 0, "lat": []}
    lock = threading.Lock()

    def worker():
        while time.time() < stop:
            inputs = [doc() for _ in range(batch)]
            t0 = time.time(); tok = post(inputs); dt = time.time() - t0
            with lock:
                tot["docs"] += batch; tot["tok"] += tok; tot["req"] += 1; tot["lat"].append(dt)

    t0 = time.time()
    threads = [threading.Thread(target=worker) for _ in range(conc)]
    for t in threads: t.start()
    for t in threads: t.join()
    el = time.time() - t0
    lat = sorted(tot["lat"])
    print("%-30s %7.1f docs/s %8.0f tok/s %5.0f tok/doc   request p50 %6.0f ms   (%d requests, %d docs, %.0f s)" % (
        label, tot["docs"] / el, tot["tok"] / el, tot["tok"] / max(tot["docs"], 1), lat[len(lat) // 2] * 1000,
        tot["req"], tot["docs"], el), flush=True)


post([doc()])   # warm
run("1 doc/request, serial", 1, 1)
run("32 docs/request, serial", 32, 1)
run("32 docs/request, 4 parallel", 32, 4)
