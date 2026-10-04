#!/bin/bash
# parity_brief.sh <reference rows.json> <candidate rows.json>...: one compact line per candidate
ref=$1; shift
for c in "$@"; do
  python3 spark/parity.py --reference "$ref" --candidate "$c" | python3 -c "
import json,sys;d=json.load(sys.stdin)
print(f\"{d['candidate']}: n={d['paired']} unpaired={d['unpaired_reference']}/{d['unpaired_candidate']} flips={d['flips']} ({d['flip_rate']:.2%}) dp max={d['dp_max']:.4f} p99={d['dp_p99']:.4f} med={d['dp_median']:.5f} acc {d['acc_reference']:.4f}->{d['acc_candidate']:.4f} delta={d['acc_delta']:+.4f} CI[{d['acc_delta_ci95'][0]:+.4f},{d['acc_delta_ci95'][1]:+.4f}]\")"
done
