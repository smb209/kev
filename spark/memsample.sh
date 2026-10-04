#!/bin/bash
# Samples system memory every 2 s into $1 (CSV: unix_s, MemAvailable_MB, MemFree_MB, Cached_MB). Unified memory: GPU use shows here.
out=${1:-memsample.csv}
echo "t,avail_mb,free_mb,cached_mb" > "$out"
while true; do
  awk -v t=$(date +%s) '/MemAvailable/{a=$2}/MemFree/{f=$2}/^Cached/{c=$2}END{printf "%d,%d,%d,%d\n",t,a/1024,f/1024,c/1024}' /proc/meminfo >> "$out"
  sleep 2
done
