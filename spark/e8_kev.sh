#!/bin/bash
# E8 Kev arms: kev.benchmark in-process (venv) on the E8 suites. usage: spark/e8_kev.sh <run> <name> [suites...]
cd ~/kev; . .venv-spark/bin/activate
export PATH=/usr/local/cuda/bin:$PATH TRITON_CACHE_DIR=$HOME/kev/.triton-cache
run=$1; name=$2; shift 2
suites=${@:-v9/transfer-v9 external/semif-v1 v7/decision-v7 hard-v1 devtools-v1 documents-v1}
mkdir -p runs/spark/e8
for s in $suites; do
  n=$(basename $s)
  python -m kev.benchmark --run "$run" --suite evals/$s --out runs/spark/e8/$name-$n > runs/spark/e8/$name-$n.log 2>&1 || echo "FAIL $n"
  python -c "import json;r=json.load(open('runs/spark/e8/$name-$n/report.json'));print('$name $n',r['clean']['n'],round(r['clean']['acc'],4),r.get('latency_ms'))"
done
echo "E8DONE $name"
