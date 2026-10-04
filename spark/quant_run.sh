#!/bin/bash
# Run a command in kev-spark-quant:latest with the repo at /work (PYTHONPATH=/work), the HF cache at /hf, a 60 GB memory cap.
#   spark/quant_run.sh NAME python3 spark/kev_quant.py --scheme nvfp4 benchmark ...
# NAME becomes the container name (q-NAME); extra docker flags via QUANT_DOCKER_FLAGS (e.g. "-p 8019:8019").
set -euo pipefail
name=$1; shift
exec docker run --rm --name "q-$name" --gpus all --memory=60g --ipc=host \
  -v "$HOME/kev:/work" -v "$HOME/.cache/huggingface:/hf" -e HF_HOME=/hf -e HOME=/tmp \
  -e TRITON_CACHE_DIR=/work/.triton-cache-quant -e TORCHINDUCTOR_CACHE_DIR=/work/.inductor-cache-quant \
  -e PYTHONPATH=/work -e PYTHONUNBUFFERED=1 -w /work --user "$(id -u):$(id -g)" ${QUANT_DOCKER_FLAGS:-} \
  kev-spark-quant:latest "$@"
