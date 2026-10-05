#!/bin/bash
# Health and GPU 0 memory of the 4060 stack.
echo "embedder (8001): $(curl -s -m 5 localhost:8001/v1/models | python3 -c "import json,sys;print([m[\"id\"] for m in json.load(sys.stdin)[\"data\"]])" 2>/dev/null || echo DOWN)"
echo "clef-flash (8031): $(curl -s -m 5 localhost:8031/v1/models | python3 -c "import json,sys;m=json.load(sys.stdin)[\"models\"][0];print(m[\"name\"], \"|\", m[\"dtype\"])" 2>/dev/null || echo DOWN)"
nvidia-smi -i 0 --query-gpu=name,memory.used,memory.total --format=csv,noheader
nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
