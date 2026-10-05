#!/bin/bash
# RTX 4060 Ti (GPU 0) stack: Qwen3-Embedding-4B (vLLM, port 8001) + FP8 Clef-Flash text-only (port 8031).
# Start order matters: embedder first (pins its KV cache), then Clef-Flash. See /home/scott/ai-stack/README.md.
set -e
S=/home/scott/ai-stack
nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | grep -q . && { echo "GPU 0 is busy; run stop-4060-stack.sh first"; nvidia-smi -i 0; exit 1; }
source /home/scott/miniconda3/etc/profile.d/conda.sh && conda activate vllm   # conda activation sets the gcc toolchain + nvcc headers FlashInfer JIT needs (system glibc headers break nvcc)
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
setsid nohup /home/scott/miniconda3/envs/vllm/bin/vllm serve /home/scott/models/Qwen3-Embedding-4B \
  --served-model-name qwen3-embedding-4b --runner pooling --quantization fp8 --kv-cache-dtype auto \
  --max-model-len 8192 --max-num-seqs 1 --kv-cache-memory-bytes 1342177280 \
  --attention-backend flashinfer --host 0.0.0.0 --port 8001 >> $S/embed-4b.log 2>&1 < /dev/null &
echo $! > $S/embed.pid
echo "embedder starting (pid $(cat $S/embed.pid)), log $S/embed-4b.log"
until curl -s -m 3 localhost:8001/v1/models >/dev/null; do kill -0 $(cat $S/embed.pid) 2>/dev/null || { echo "embedder exited; see $S/embed-4b.log"; exit 1; }; sleep 5; done
echo "embedder up"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/home/scott/clef-flash/.triton-cache
cd /home/scott/clef-flash
setsid nohup venv/bin/python clef_server.py --fp8-export /home/scott/clef-flash/export --host-embeddings --text-only \
  --max_length 8192 --host 0.0.0.0 --port 8031 >> $S/clef-flash.log 2>&1 < /dev/null &
echo $! > $S/clef.pid
echo "clef-flash starting (pid $(cat $S/clef.pid)), log $S/clef-flash.log"
until curl -s -m 3 localhost:8031/v1/models >/dev/null; do kill -0 $(cat $S/clef.pid) 2>/dev/null || { echo "clef-flash exited; see $S/clef-flash.log"; exit 1; }; sleep 5; done
echo "clef-flash up"; bash $S/status-4060-stack.sh
