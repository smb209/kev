#!/bin/bash
# E4: all four arms on spark-1, one container per arm (model loaded once per arm), sequential. Memory sampled alongside.
cd ~/kev
for s in bf16 fp8 nvfp4-mlp nvfp4; do
  QUANT_MEM=100g bash spark/quant_run.sh e4-$s python3 spark/e4_reads.py $s > runs/spark/e4-$s.log 2>&1; grep -E "^E4|Error|Traceback" runs/spark/e4-$s.log
done
echo E4ALLDONE
