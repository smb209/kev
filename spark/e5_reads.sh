#!/bin/bash
# E5 reads: a Kev-0.8B checkpoint on the four pre-registered dev panels (kev.benchmark, venv, exact fp32 path like the
# H200 reads). usage: spark/e5_reads.sh <run (dir or hub id@rev)> <name>
cd ~/kev; . .venv-spark/bin/activate
export PATH=/usr/local/cuda/bin:$PATH TRITON_CACHE_DIR=$HOME/kev/.triton-cache
run=$1; name=$2
for s in v7/decision-v7 documents-v1 hard-v1 devtools-v1; do
  n=$(basename $s)
  python -m kev.benchmark --run "$run" --suite evals/$s --out runs/spark/e5/$name-$n > runs/spark/e5/$name-$n.log 2>&1 || echo "FAIL $n"
  python -c "import json;r=json.load(open('runs/spark/e5/$name-$n/report.json'));print('$name $n',r['clean']['n'],round(r['clean']['acc'],4))"
done
