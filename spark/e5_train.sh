#!/bin/bash
# E5: round 15's Kev-0.8B documents+skills stage (trial r15-08b/00-trial-0, H200) re-run on a DGX Spark.
# Config = that trial's provenance.json config verbatim, init_from pinned to the tag it meant then (night2-du-release).
# usage: spark/e5_train.sh <out dir> [extra kev.train flags, e.g. --max_steps 60 or --seed 2]
set -e
cd ~/kev; . .venv-spark/bin/activate
export PATH=/usr/local/cuda/bin:$PATH TRITON_CACHE_DIR=$HOME/kev/.triton-cache
out=$1; shift
mkdir -p $out
# memory cap: unified memory; a runaway job must not take the box (guide section 5)
exec systemd-run --user --scope -p MemoryMax=100G -p MemorySwapMax=0 \
  python ${KEV_TRAIN:--m kev.train} --suite evals/v7/decision-v7 --out $out/checkpoint --device cuda --epochs 1 --seed 1 --lr 2e-05 --lora 16 --accum 2 --batch 4 --perm_kl 0.0 --perm_frac 0.3 --ord_w 0.0 --p_none 0.1 --p_none_distract 0.12 --p_distract 0.15 --p_none_pair 0.25 --synthetic_repeat 1 --public_frac 1.0 --head_lr 0.0 --weight_decay 0.01 --anchor_w 0.0 --label_smoothing 0.0 --brier_w 0.0 --focal_gamma 0.0 --dtype bf16 --checkpointing 1 --replay 6000 --max_state 7552 --data evals/round15/joint/train.jsonl --base Qwen/Qwen3.5-0.8B-Base --base_revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 --init_from jaredpalmer/kev-0.8b@night2-du-release "$@" 2>&1 | tee $out/train.log
