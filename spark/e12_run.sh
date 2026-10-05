#!/bin/bash
# E12: Qwen3-Embedding-4B (vLLM, FP8 weights + FP8 KV, 8k, KV pinned) + text-only FP8 Clef-Flash (8k), together on spark-1.
cd ~/kev; mkdir -p runs/spark/e12
until grep -q DLDONE runs/spark/dl4b.log; do sleep 15; done
docker run -d --rm --name e12-embed --gpus all --network host --ipc=host \
  -v $HOME/.cache/huggingface:/hf -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  --entrypoint vllm sparkrun-eugr-vllm-tf5:latest serve Qwen/Qwen3-Embedding-4B \
  --runner pooling --quantization fp8 --kv-cache-dtype fp8 --max-model-len 8192 --max-num-seqs 2 \
  --kv-cache-memory-bytes 1342177280 --host 127.0.0.1 --port 8051 > /dev/null
until curl -s -m 3 localhost:8051/v1/models >/dev/null; do docker ps --format "{{.Names}}" | grep -q e12-embed || { echo "embed died"; exit 1; }; sleep 10; done
docker logs e12-embed 2>&1 | grep -iE "memory|KV cache|weights" | tail -8 > runs/spark/e12/embed-startup.log
# probe: an over-length input (~14.7k tokens > max_model_len 8192) with truncate_prompt_tokens; v3 saw it hang 600 s
python3 - > runs/spark/e12/overlength-probe.txt 2>&1 <<"PY"
import json, time, urllib.request
p = ("The quarterly report covers revenue, customer retention and infrastructure spending across all regions. "
     "Support tickets about billing errors rose after the pricing change, while shipping delays fell. ")
for name, body in (("over-length, truncate_prompt_tokens=8192", {"input": [p * 475], "truncate_prompt_tokens": 8192}),
                   ("over-length, no truncation", {"input": [p * 475]})):
    t0 = time.time()
    try:
        req = urllib.request.Request("http://127.0.0.1:8051/v1/embeddings", data=json.dumps({"model": "Qwen/Qwen3-Embedding-4B", **body}).encode(), headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as r: print(name, "->", r.status, json.loads(r.read()).get("usage"), f"{time.time()-t0:.1f}s")
    except Exception as e: print(name, "->", type(e).__name__, str(e)[:300], f"{time.time()-t0:.1f}s")
PY
docker logs e12-embed 2>&1 | tail -30 > runs/spark/e12/embed-after-probe.log
curl -s -m 5 localhost:8051/health -o /dev/null -w "health after probe: %{http_code}\n" >> runs/spark/e12/overlength-probe.txt
docker ps --format "{{.Names}}" | grep -q e12-embed || { echo "embed died after probe"; exit 1; }
QUANT_MEM=60g QUANT_DOCKER_FLAGS="-d --network host -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" bash spark/quant_run.sh e12-clef \
  python3 spark/clef_server.py --port 8031 --max_length 8192 --fp8-export runs/spark/c10-clef-flash-fp8 --host-embeddings --text-only > /dev/null
until curl -s -m 3 localhost:8031/v1/models >/dev/null; do docker ps --format "{{.Names}}" | grep -q q-e12-clef || { echo "clef died"; exit 1; }; sleep 10; done
python3 spark/e12_corun.py --embed http://127.0.0.1:8051 --clef http://127.0.0.1:8031 --seconds 180 --out runs/spark/e12/corun-v4.json > runs/spark/e12/corun-v4.log 2>&1 || echo "corun failed"
docker logs e12-embed > runs/spark/e12/embed-full.log 2>&1; docker logs q-e12-clef > runs/spark/e12/clef-full.log 2>&1
docker stop e12-embed q-e12-clef > /dev/null
echo E12DONE
