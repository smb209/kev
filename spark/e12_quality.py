"""E12 context item: does FP8 (weights + KV cache, unit KV scale) change Qwen3-Embedding-4B's embeddings vs bf16?

Embeds the same documents through two vLLM servers (bf16 reference; FP8 weights + FP8 KV) and reports per-document
cosine(bf16, fp8) and nearest-neighbour agreement: for each query (the document's first ~300 characters), whether the
top-1 retrieved document under FP8 equals the one under bf16. Standard library + numpy only.

    python3 spark/e12_quality.py --ref http://127.0.0.1:8052 --cand http://127.0.0.1:8051 --docs docs.json --out q.json
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
ap.add_argument("--ref", required=True); ap.add_argument("--cand", required=True); ap.add_argument("--docs", required=True)
ap.add_argument("--model", default="Qwen/Qwen3-Embedding-4B"); ap.add_argument("--out", required=True)
a = ap.parse_args()
docs = json.load(open(a.docs)); assert len(docs) >= 50, "too few documents"
queries = ["Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: " + d[:300] for d in docs]
R, C = embed(a.ref, a.model, docs), embed(a.cand, a.model, docs)
Rq, Cq = embed(a.ref, a.model, queries), embed(a.cand, a.model, queries)
cos = (R * C).sum(1)
top_ref, top_cand = (Rq @ R.T).argmax(1), (Cq @ C.T).argmax(1)
rank_ref = np.argsort(-(Rq @ R.T), 1)[:, :10]; rank_cand = np.argsort(-(Cq @ C.T), 1)[:, :10]
overlap10 = np.mean([len(set(x) & set(y)) / 10 for x, y in zip(rank_ref, rank_cand)])
res = {"documents": len(docs), "cosine_min": float(cos.min()), "cosine_median": float(np.median(cos)), "cosine_mean": float(cos.mean()),
       "top1_agreement": float((top_ref == top_cand).mean()), "top10_overlap": float(overlap10),
       "self_retrieval_ref": float((top_ref == np.arange(len(docs))).mean()), "self_retrieval_cand": float((top_cand == np.arange(len(docs))).mean())}
json.dump(res, open(a.out, "w"), indent=1); print(json.dumps(res, indent=1))
