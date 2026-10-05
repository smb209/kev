#!/bin/bash
# E10b: (1) default mode after the GEMM-output probe change must reproduce E10 FP8 rows on transfer-v9;
# (2) forced bf16-output fallback (FP8_GEMM_OUT=bfloat16, what an sm_89 without fp32 output would run) on all three suites.
cd ~/kev; port=8031
serve() { local name=$1; shift; local extra=$1; shift
  QUANT_MEM=60g QUANT_DOCKER_FLAGS="-d --network host -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $extra" bash spark/quant_run.sh e10-$name python3 spark/clef_server.py --port $port "$@"
  until curl -s -m 3 localhost:$port/v1/models >/dev/null; do docker ps --format "{{.Names}}" | grep -q "q-e10-$name" || { echo "server $name died"; exit 1; }; sleep 10; done
  curl -s localhost:$port/v1/models; echo
}
bench() { QUANT_DOCKER_FLAGS="--network host" QUANT_MEM=8g bash spark/quant_run.sh e10b-$1-$(basename $2) python3 -m kev.benchmark --remote http://127.0.0.1:$port --remote-model Cloudflare/clef-flash --suite evals/$2 --out runs/spark/e10/$1-$(basename $2) > runs/spark/e10/$1-$(basename $2).log 2>&1 || echo "FAIL $1 $2"; echo "done $1 $2"; }
serve fp8re "" --fp8-export runs/spark/c10-clef-flash-fp8
bench clef-flash-fp8re v9/transfer-v9
docker stop q-e10-fp8re >/dev/null
serve fp8bf16out "-e FP8_GEMM_OUT=bfloat16" --fp8-export runs/spark/c10-clef-flash-fp8
for s in v9/transfer-v9 breadth-v1 v7/decision-v7; do bench clef-flash-fp8bf16out $s; done
docker stop q-e10-fp8bf16out >/dev/null
echo E10BDONE
