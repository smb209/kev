#!/bin/bash
# semif-v1 development reads of kev-4b under NVFP4 with static activation scales (the default), next to the dynamic-scale ones.
cd ~/kev
for scheme in nvfp4 nvfp4-mlp; do
  out=runs/spark/quant/q-kev-4b-$scheme-static-semif
  rm -rf "$out"
  echo "=== 4b $scheme static $(date -u +%FT%TZ)"
  spark/quant_run.sh "bs-4b-$scheme" python3 spark/kev_quant.py --scheme $scheme --act-scale static benchmark \
    --run jaredpalmer/kev-4b --suite evals/external/semif-v1 --out $out 2>&1 | grep -E "\[kev_quant\]|Error|error|Traceback"
done
echo ALLDONE
