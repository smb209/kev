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
QUANT_MEM=60g QUANT_DOCKER_FLAGS="-d --network host -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" bash spark/quant_run.sh e12-clef \
  python3 spark/clef_server.py --port 8031 --max_length 8192 --fp8-export runs/spark/c10-clef-flash-fp8 --host-embeddings --text-only > /dev/null
until curl -s -m 3 localhost:8031/v1/models >/dev/null; do docker ps --format "{{.Names}}" | grep -q q-e12-clef || { echo "clef died"; exit 1; }; sleep 10; done
python3 spark/e12_corun.py --embed http://127.0.0.1:8051 --clef http://127.0.0.1:8031 --seconds 180 --out runs/spark/e12/corun.json > runs/spark/e12/corun.log 2>&1 || echo "corun failed"
docker stop e12-embed q-e12-clef > /dev/null
echo E12DONE
