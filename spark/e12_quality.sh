#!/bin/bash
# bf16 reference server (8052) vs FP8 + FP8 KV server (8051), one after the other, on 300 documents-v1 states (<= 8k tokens).
cd ~/kev; out=runs/spark/e12/quality; mkdir -p $out
. .venv-spark/bin/activate
python - <<"PY"
import json
from kev.suite import load_split
docs = [r["state"] if isinstance(r["state"], str) else json.dumps(r["state"]) for r in load_split("evals/documents-v1", "development")]
docs = [d[:30000] for d in docs if len(d) > 400][:300]
json.dump(docs, open("runs/spark/e12/quality/docs.json", "w")); print(len(docs), "documents")
PY
deactivate
serve() { docker run -d --rm --name $1 --gpus all --network host --ipc=host -v $HOME/.cache/huggingface:/hf -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  --entrypoint vllm sparkrun-eugr-vllm-tf5:latest serve Qwen/Qwen3-Embedding-4B --runner pooling --max-model-len 8192 --max-num-seqs 8 \
  --kv-cache-memory-bytes 4294967296 --host 127.0.0.1 --port $2 "${@:3}" > /dev/null
  until curl -s -m 3 localhost:$2/v1/models >/dev/null; do docker ps --format "{{.Names}}" | grep -q $1 || { echo "$1 died"; exit 1; }; sleep 5; done; }
serve q-bf16 8052
serve q-fp8 8051 --quantization fp8 --kv-cache-dtype fp8
python3 spark/e12_quality.py --ref http://127.0.0.1:8052 --cand http://127.0.0.1:8051 --docs $out/docs.json --out $out/quality.json > $out/quality.log 2>&1 || echo "quality failed"
docker stop q-bf16 q-fp8 > /dev/null
echo QDONE
