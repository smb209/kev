"""A /v1/systemone server for Cloudflare's Clef / Clef-Flash, so kev.benchmark --remote scores them through Kev's harness.

E8 (spark/LAB_NOTEBOOK.md). Uses the release's own joint_schema_model.py (encode_record, collate_records,
load_release_model), refusing to import it unless its sha256 is the hand-reviewed one. Unlike the release's
`systemone()`, probabilities are returned at full precision (it rounds to 4 decimals, which would put zeros into
kev.benchmark's NLL), and one request runs at a time.

    python spark/clef_server.py --repo Cloudflare/clef --revision 2f3de3dd85f379784083b0814d997ab627200f0c --port 8031

E10: --fp8-export DIR serves a spark/clef_fp8.py export (FP8 decoder projections) through spark/clef_fp8.load_fp8 instead of
load_release_model; the release code comes from the export's copy (same sha256 gate), --repo / --revision default to the
export's source and must match it when given. Everything after the load is the same.

    python spark/clef_server.py --fp8-export runs/spark/c10-clef-flash-fp8 --port 8031
"""
import argparse, hashlib, importlib.util, sys, threading, time
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from huggingface_hub import snapshot_download

REVIEWED_SHA256 = "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"   # joint_schema_model.py, both repos

ap = argparse.ArgumentParser()
ap.add_argument("--repo")
ap.add_argument("--revision")
ap.add_argument("--fp8-export", help="a spark/clef_fp8.py export directory: serve it instead of the bf16 release")
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=8031)
ap.add_argument("--max_length", type=int, default=16384)
a = ap.parse_args()
dtype = "bfloat16"

if a.fp8_export:
    import json
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import clef_fp8
    source = json.loads((Path(a.fp8_export) / clef_fp8.CONFIG_NAME).read_text())["source"]
    for k in ("repo", "revision"):
        if getattr(a, k) is None: setattr(a, k, source[k])
        elif getattr(a, k) != source[k]: raise SystemExit(f"--{k} {getattr(a, k)} is not the export's source {source[k]}")
    path = Path(a.fp8_export)
    code = path / "joint_schema_model.py"
    dtype = "fp8-e4m3 decoder projections + bfloat16"
elif not (a.repo and a.revision):
    raise SystemExit("--repo and --revision are required without --fp8-export")
else:
    path = Path(snapshot_download(a.repo, revision=a.revision))
    code = path / "joint_schema_model.py"
digest = hashlib.sha256(code.read_bytes()).hexdigest()
if digest != REVIEWED_SHA256:
    raise SystemExit(f"{code} sha256 {digest} is not the reviewed {REVIEWED_SHA256}; review it before running")
spec = importlib.util.spec_from_file_location("joint_schema_model", code)
jsm = importlib.util.module_from_spec(spec); sys.modules["joint_schema_model"] = jsm; spec.loader.exec_module(jsm)

t0 = time.time()
if a.fp8_export:
    model, processor = clef_fp8.load_fp8(path, device="cuda")
    dtype += f", fp8 GEMM output {str(clef_fp8.gemm_out_dtype(torch.device('cuda'))).replace('torch.', '')}"   # float32 = the E10-read arithmetic
else:
    model, processor = jsm.load_release_model(path, device="cuda")
print(f"loaded {a.repo}@{a.revision[:8]} ({dtype}) in {time.time() - t0:.0f}s, GPU {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
lock = threading.Lock()
app = FastAPI()


@app.get("/v1/models")
def models():
    return {"models": [{"name": a.repo, "revision": a.revision, "device": torch.cuda.get_device_name(0), "dtype": dtype}]}


@app.post("/v1/systemone")
async def systemone(request: Request):
    body = await request.json()
    questions = body.get("questions")
    if "state" not in body or not isinstance(questions, dict) or not questions:
        raise HTTPException(422, "state and at least one question are required")
    with lock:
        start = time.perf_counter()
        try:
            enc = jsm.encode_record(processor.tokenizer, body, max_length=a.max_length, processor=processor)
        except ValueError as e:   # over max_length and similar: the harness treats 422 as a context overflow
            raise HTTPException(422, str(e))
        with torch.inference_mode():
            logits = model(jsm.collate_records([enc], processor.tokenizer.pad_token_id, torch.device("cuda")))[0]
        torch.cuda.synchronize()
        latency = 1000 * (time.perf_counter() - start)
    answers = {}
    for q, z in zip(enc.questions, logits):
        p = dict(zip(q.option_ids, z.float().softmax(-1).tolist()))
        spec_q = questions[q.question_id]
        if spec_q["type"] == "noul":
            answers[q.question_id] = {"type": "noul", "noul": p["true"]}
        elif spec_q["type"] == "choice":
            choice = max(p, key=p.get)
            answers[q.question_id] = {"type": "choice", "choice": choice, "confidence": p[choice], "probabilities": p}
        else:
            answers[q.question_id] = {"type": "score", "score": sum(int(k) * v for k, v in p.items()), "confidence": max(p.values()),
                                      "probabilities": p}
    return {"model": body.get("model", a.repo), "answers": answers, "latency_ms": latency,
            "usage": {"input_tokens": len(enc.input_ids), "output_tokens": 0}}


uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
