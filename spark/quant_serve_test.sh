#!/bin/bash
# Serve kev-4b under each scheme given (default bf16 nvfp4) with kev.serve's defaults (bf16, fused, CUDA graphs) through
# spark/kev_quant.py, measure warm latency on spark/req1.json (spark/quant_serve_client.py), stop the server.
#   [QUANT_FLAGS="--act-quant compile" QUANT_TAG=-compile] bash spark/quant_serve_test.sh [run] [schemes...]
cd ~/kev
run=${1:-jaredpalmer/kev-4b}; shift || true
schemes=${*:-bf16 nvfp4}
tag=$(basename "$run")${QUANT_TAG:-}
for scheme in $schemes; do
  echo "=== serve $tag $scheme $(date -u +%FT%TZ) free=$(free -g | awk '/Mem:/{print $7}')G"
  QUANT_DOCKER_FLAGS="-d -p 8019:8019" spark/quant_run.sh "serve-$scheme" python3 spark/kev_quant.py --scheme $scheme ${QUANT_FLAGS:-} \
    --quant-json runs/spark/quant/serve-$tag-$scheme.quant.json serve --run "$run" --host 0.0.0.0 --port 8019 > /dev/null
  python3 spark/quant_serve_client.py --url http://127.0.0.1:8019 --out runs/spark/quant/serve-$tag-$scheme.json | tail -40
  docker logs "q-serve-$scheme" 2>&1 | grep -E "kev_quant|serving|cuda_graphs|capturing|Error|Traceback" > runs/spark/quant/serve-$tag-$scheme.log
  cat runs/spark/quant/serve-$tag-$scheme.log
  docker stop "q-serve-$scheme" > /dev/null
done
echo ALLDONE
