#!/bin/bash
# E11a: full transfer-v9 read of FP8 Clef-Flash with --host-embeddings --text-only; must equal E10 FP8 rows bit for bit.
cd ~/kev; port=8031
until grep -q E10BDONE runs/spark/e10b_all.log; do sleep 30; done
QUANT_MEM=60g QUANT_DOCKER_FLAGS="-d --network host -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" bash spark/quant_run.sh e11a python3 spark/clef_server.py --port $port --fp8-export runs/spark/c10-clef-flash-fp8 --host-embeddings --text-only
until curl -s -m 3 localhost:$port/v1/models >/dev/null; do docker ps --format "{{.Names}}" | grep -q q-e11a || { echo died; exit 1; }; sleep 10; done
curl -s localhost:$port/v1/models; echo
QUANT_DOCKER_FLAGS="--network host" QUANT_MEM=8g bash spark/quant_run.sh e11ab python3 -m kev.benchmark --remote http://127.0.0.1:$port --remote-model Cloudflare/clef-flash --suite evals/v9/transfer-v9 --out runs/spark/e10/clef-flash-fp8host-transfer-v9 > runs/spark/e10/clef-flash-fp8host-transfer-v9.log 2>&1
docker stop q-e11a >/dev/null
echo E11ADONE
