#!/bin/bash
# E12 v5 (after review): GPU memory sampled from BEFORE either server starts (startup / load peaks), true cold idle (no
# probe), two concurrent embedding streams, Clef alternating 3-question and 10-question x 5-option requests. Gate 15.0 GiB.
cd ~/kev; mkdir -p runs/spark/e12; out=runs/spark/e12/v5
mkdir -p $out
( while true; do echo "$(date +%s),$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | tr '\n' ';')" >> $out/gpu-timeline.csv; sleep 0.5; done ) & SAMPLER=$!
docker run -d --rm --name e12-embed --gpus all --network host --ipc=host \
  -v $HOME/.cache/huggingface:/hf -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  --entrypoint vllm sparkrun-eugr-vllm-tf5:latest serve Qwen/Qwen3-Embedding-4B \
  --runner pooling --quantization fp8 --kv-cache-dtype fp8 --max-model-len 8192 --max-num-seqs 2 \
  --kv-cache-memory-bytes 1342177280 --host 127.0.0.1 --port 8051 > /dev/null
until curl -s -m 3 localhost:8051/v1/models >/dev/null; do docker ps --format "{{.Names}}" | grep -q e12-embed || { echo "embed died"; kill $SAMPLER; exit 1; }; sleep 5; done
QUANT_MEM=60g QUANT_DOCKER_FLAGS="-d --network host -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" bash spark/quant_run.sh e12-clef \
  python3 spark/clef_server.py --port 8031 --max_length 8192 --fp8-export runs/spark/c10-clef-flash-fp8 --host-embeddings --text-only > /dev/null
until curl -s -m 3 localhost:8031/v1/models >/dev/null; do docker ps --format "{{.Names}}" | grep -q q-e12-clef || { echo "clef died"; kill $SAMPLER; exit 1; }; sleep 5; done
echo "both up $(date +%s)" > $out/marks.txt
python3 spark/e12_corun.py --embed http://127.0.0.1:8051 --clef http://127.0.0.1:8031 --seconds 180 --embed-streams 2 --clef-big-every 2 --out $out/corun.json > $out/corun.log 2>&1 || echo "corun failed"
echo "load done $(date +%s)" >> $out/marks.txt
docker logs e12-embed > $out/embed-full.log 2>&1; docker logs q-e12-clef > $out/clef-full.log 2>&1
docker stop e12-embed q-e12-clef > /dev/null; sleep 2; kill $SAMPLER
echo E12V5DONE
