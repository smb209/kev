#!/bin/bash
# E4: all four arms on spark-1, one container per arm (model loaded once per arm), sequential. Memory sampled alongside.
cd ~/kev
for s in bf16 fp8 nvfp4-mlp nvfp4; do
  QUANT_MEM=100g bash spark/quant_run.sh e4-$s python3 spark/e4_reads.py $s 2>&1 | grep -v -iE "warn|capab|\(8.0\)|^\s*$|Fetching|Loading" | tail -40
done
echo E4ALLDONE
