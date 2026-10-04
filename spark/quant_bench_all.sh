#!/bin/bash
# The semif-v1 development smoke reads of every scheme (spark/kev_quant.py benchmark), sequentially, on spark-2.
cd ~/kev
for model in 0.8b 4b; do
  for scheme in bf16 fp8 nvfp4 nvfp4-mlp; do
    out=runs/spark/quant/q-kev-$model-$scheme-semif
    rm -rf "$out"
    echo "=== $model $scheme $(date -u +%FT%TZ) free=$(free -g | awk '/Mem:/{print $7}')G"
    spark/quant_run.sh "b-$model-$scheme" python3 spark/kev_quant.py --scheme $scheme benchmark \
      --run jaredpalmer/kev-$model --suite evals/external/semif-v1 --out $out 2>&1 | grep -E "\[kev_quant\]|Error|error|Traceback"
  done
done
for scheme in bf16 nvfp4; do   # the fused layers (kev.fused_qwen35) under quantization, on the benchmark path
  out=runs/spark/quant/q-kev-4b-$scheme-fused-semif; rm -rf "$out"
  echo "=== 4b $scheme fused $(date -u +%FT%TZ)"
  spark/quant_run.sh "b-4b-$scheme-fused" python3 spark/kev_quant.py --scheme $scheme --fused 1 benchmark \
    --run jaredpalmer/kev-4b --suite evals/external/semif-v1 --out $out 2>&1 | grep -E "\[kev_quant\]|Error|error|Traceback"
done
echo ALLDONE
