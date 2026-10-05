"""E12 context item: does FP8 (weights + KV cache, unit KV scale) change Qwen3-Embedding-4B's embeddings vs bf16?

Embeds the same documents through two vLLM servers (bf16 reference; FP8 weights + FP8 KV) and reports per-document
cosine(bf16, fp8) and nearest-neighbour agreement: for each query (the document's first ~300 characters), whether the
top-1 retrieved document under FP8 equals the one under bf16. Standard library + numpy only.

    python3 spark/e12_quality.py dump --url http://127.0.0.1:8051 --docs docs.json --out bf16.npz     # one server at a time
    python3 spark/e12_quality.py compare --ref bf16.npz --cand fp8.npz --out q.json
"""
import argparse, json, urllib.request
import numpy as np

def embed(url, model, texts):
    out = []
    for i in range(0, len(texts), 8):
        body = json.dumps({"model": model, "input": texts[i:i + 8]}).encode()
        req = urllib.request.Request(f"{url}/v1/embeddings", data=body, headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r: out += [d["embedding"] for d in json.loads(r.read())["data"]]
    v = np.array(out, dtype=np.float64); return v / np.linalg.norm(v, axis=1, keepdims=True)

ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
d = sub.add_parser("dump"); d.add_argument("--url", required=True); d.add_argument("--docs", required=True); d.add_argument("--out", required=True)
d.add_argument("--model", default="Qwen/Qwen3-Embedding-4B")
c = sub.add_parser("compare"); c.add_argument("--ref", required=True); c.add_argument("--cand", required=True); c.add_argument("--out", required=True)
a = ap.parse_args()
if a.cmd == "dump":
    docs = json.load(open(a.docs)); assert len(docs) >= 50, "too few documents"
    queries = ["Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: " + x[:300] for x in docs]
    np.savez(a.out, docs=embed(a.url, a.model, docs), queries=embed(a.url, a.model, queries)); print("dumped", a.out, len(docs))
else:
    r, k = np.load(a.ref), np.load(a.cand)
    R, C, Rq, Cq = r["docs"], k["docs"], r["queries"], k["queries"]
    assert R.shape == C.shape and len(R) >= 50, (R.shape, C.shape)
    cos = (R * C).sum(1)
    sr, sc = Rq @ R.T, Cq @ C.T
    top_ref, top_cand = sr.argmax(1), sc.argmax(1)
    rk_r, rk_c = np.argsort(-sr, 1)[:, :10], np.argsort(-sc, 1)[:, :10]
    res = {"documents": len(R), "cosine_min": float(cos.min()), "cosine_median": float(np.median(cos)), "cosine_mean": float(cos.mean()),
           "top1_agreement": float((top_ref == top_cand).mean()),
           "top10_overlap": float(np.mean([len(set(x) & set(y)) / 10 for x, y in zip(rk_r, rk_c)])),
           "self_retrieval_ref": float((top_ref == np.arange(len(R))).mean()), "self_retrieval_cand": float((top_cand == np.arange(len(R))).mean())}
    res["harmless_by_rule"] = res["cosine_median"] >= 0.99 and res["top1_agreement"] >= 0.95
    json.dump(res, open(a.out, "w"), indent=1); print(json.dumps(res, indent=1))
