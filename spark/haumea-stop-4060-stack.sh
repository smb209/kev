#!/bin/bash
# Stops the GPU 0 stack started by start-4060-stack.sh (Clef-Flash first, then the embedder).
S=/home/scott/ai-stack
for name in clef embed; do
  f=$S/$name.pid; [ -f $f ] || continue; p=$(cat $f)
  if kill -0 $p 2>/dev/null; then kill -INT $p; for i in $(seq 1 30); do kill -0 $p 2>/dev/null || break; sleep 1; done; kill -0 $p 2>/dev/null && kill -9 $p; echo "stopped $name ($p)"; fi
  rm -f $f
done
sleep 3; nvidia-smi -i 0 --query-compute-apps=pid,used_memory --format=csv,noheader
