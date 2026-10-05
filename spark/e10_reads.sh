#!/bin/bash
# E10 reads on spark-1 (the E8 / E9 machine and image): FP8 Clef-Flash from the export on transfer-v9 / breadth-v1 /
# decision-v7, plus a bf16 control re-read of transfer-v9 that must reproduce runs/spark/e8/clef-flash-transfer-v9.
cd ~/kev; port=8031
serve() {  # name, extra server args...
  local name=$1; shift
  QUANT_MEM=60g QUANT_DOCKER_FLAGS="-d --network host -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" bash spark/quant_run.sh e10-$name python3 spark/clef_server.py --port $port "$@"
  until curl -s -m 3 localhost:$port/v1/models >/dev/null; do docker ps --format "{{.Names}}" | grep -q "q-e10-$name" || { echo "server $name died"; exit 1; }; sleep 10; done
}
bench() {  # name suite
  QUANT_DOCKER_FLAGS="--network host" QUANT_MEM=8g bash spark/quant_run.sh e10b-$1-$(basename $2) python3 -m kev.benchmark --remote http://127.0.0.1:$port --remote-model Cloudflare/clef-flash --suite evals/$2 --out runs/spark/e10/$1-$(basename $2) > runs/spark/e10/$1-$(basename $2).log 2>&1 || echo "FAIL $1 $2"
  python3 -c "import json;r=json.load(open(\"runs/spark/e10/$1-$(basename $2)/report.json\"));print(\"$1 $(basename $2)\",r[\"clean\"][\"n\"],round(r[\"clean\"][\"acc\"],4),r.get(\"latency_ms\"))"
}
mkdir -p runs/spark/e10
serve bf16ctl --repo Cloudflare/clef-flash --revision 17f0b0ad64efb65d273590632833508766b2aae6
bench clef-flash-bf16ctl v9/transfer-v9
docker stop q-e10-bf16ctl >/dev/null
serve fp8 --fp8-export runs/spark/c10-clef-flash-fp8
for s in v9/transfer-v9 breadth-v1 v7/decision-v7; do bench clef-flash-fp8 $s; done
docker stop q-e10-fp8 >/dev/null
echo E10DONE
