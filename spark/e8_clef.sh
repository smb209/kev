#!/bin/bash
# E8: serve one Clef model (spark/clef_server.py in kev-spark-quant) and score it with kev.benchmark --remote on the E8
# suites. usage: spark/e8_clef.sh <repo> <revision> <name>
cd ~/kev
repo=$1; rev=$2; name=$3; port=8031
QUANT_MEM=110g QUANT_DOCKER_FLAGS="-d --network host" bash spark/quant_run.sh e8-$name \
  python3 spark/clef_server.py --repo $repo --revision $rev --host 127.0.0.1 --port $port
until curl -s -m 3 localhost:$port/v1/models >/dev/null; do
  docker ps --format "{{.Names}}" | grep -q "q-e8-$name" || { echo "server died"; docker logs q-e8-$name 2>&1 | tail -20; exit 1; }
  sleep 15
done
docker logs q-e8-$name 2>&1 | grep -i loaded
for s in v9/transfer-v9 external/semif-v1 v7/decision-v7 hard-v1 devtools-v1 documents-v1; do
  n=$(basename $s)
  QUANT_DOCKER_FLAGS="--network host" QUANT_MEM=8g bash spark/quant_run.sh e8b-$name-$n python3 -m kev.benchmark --remote http://127.0.0.1:$port --remote-model $repo \
    --suite evals/$s --out runs/spark/e8/$name-$n > runs/spark/e8/$name-$n.log 2>&1 || echo "FAIL $n"
  python3 -c "import json;r=json.load(open('runs/spark/e8/$name-$n/report.json'));print('$name $n',r['clean']['n'],round(r['clean']['acc'],4),r.get('latency_ms'))"
done
docker stop q-e8-$name >/dev/null
echo "E8DONE $name"
