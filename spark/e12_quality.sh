#!/bin/bash
# bf16 vs FP8 + FP8 KV Qwen3-Embedding-4B, ONE server at a time (two vLLM instances at once killed the second); containers
# are not --rm so their logs survive; embeddings dumped to disk and compared offline.
cd ~/kev; out=runs/spark/e12/quality; mkdir -p $out
test -s $out/docs.json || { echo "no docs.json"; exit 1; }
run() {  # name port extra-args...
  local name=$1 port=$2; shift 2
  docker rm -f $name >/dev/null 2>&1
  docker run -d --name $name --gpus all --network host --ipc=host -v $HOME/.cache/huggingface:/hf -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
    --entrypoint vllm sparkrun-eugr-vllm-tf5:latest serve Qwen/Qwen3-Embedding-4B --runner pooling --max-model-len 8192 --max-num-seqs 8 \
    --kv-cache-memory-bytes 4294967296 --host 127.0.0.1 --port $port "$@" > /dev/null
  until curl -s -m 3 localhost:$port/v1/models >/dev/null; do
    if ! docker ps --format "{{.Names}}" | grep -q "^$name$"; then echo "$name died"; docker logs $name > $out/$name.log 2>&1; docker rm $name >/dev/null; return 1; fi; sleep 5; done
  python3 spark/e12_quality.py dump --url http://127.0.0.1:$port --docs $out/docs.json --out $out/$name.npz >> $out/quality.log 2>&1 || echo "dump $name failed"
  docker logs $name > $out/$name.log 2>&1; docker stop $name >/dev/null; docker rm $name >/dev/null
}
run q-bf16 8052 || exit 1
run q-fp8 8051 --quantization fp8 --kv-cache-dtype fp8 || exit 1
python3 spark/e12_quality.py compare --ref $out/q-bf16.npz --cand $out/q-fp8.npz --out $out/quality.json >> $out/quality.log 2>&1 || echo "compare failed"
echo QDONE
