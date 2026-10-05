#!/bin/bash
# The embedder that ran on the RTX 4060 Ti (GPU 0, port 8001) until 2026-10-06, captured from the live process.
# Restores the original Qwen3-Embedding-8B setup. Stop the 4060 stack first: bash /home/scott/ai-stack/stop-4060-stack.sh
cd /home/scott
source /home/scott/miniconda3/etc/profile.d/conda.sh && conda activate vllm   # conda activation sets the gcc toolchain + nvcc headers FlashInfer JIT needs (system glibc headers break nvcc)
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
setsid nohup /home/scott/miniconda3/envs/vllm/bin/vllm serve /home/scott/models/Qwen3-Embedding-8B \
  --served-model-name qwen3-embedding --runner pooling --quantization fp8 --host 0.0.0.0 --port 8001 \
  --max-model-len 32768 --max-num-batched-tokens 16384 --max-num-seqs 16 --gpu-memory-utilization 0.93 \
  --attention-backend flashinfer --trust-remote-code -tp 1 >> /home/scott/embed-4060.log 2>&1 < /dev/null &
echo "started Qwen3-Embedding-8B, pid $!; log /home/scott/embed-4060.log"
