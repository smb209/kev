#!/bin/bash
# Isolation test (spark/kev_quant.py isolation) of kev-4b: bf16, nvfp4 static, nvfp4 dynamic; fused off and on.
cd ~/kev
for fused in 0 1; do
  for cfg in "bf16 static" "nvfp4 static" "nvfp4 dynamic"; do
    set -- $cfg
    echo "=== isolation $1 $2 fused=$fused $(date -u +%FT%TZ)"
    spark/quant_run.sh "iso-$1-$2-$fused" python3 spark/kev_quant.py --scheme $1 --act-scale $2 --fused $fused isolation \
      --run jaredpalmer/kev-4b --out runs/spark/quant/isolation-kev-4b-$1-$2-fused$fused.json 2>&1 | grep -E "\[kev_quant\]|Error|Traceback"
  done
done
echo ALLDONE
