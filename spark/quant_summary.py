"""Summarize the spark/kev_quant.py benchmark reads under runs/spark/quant (accuracy, NLL, ECE, wall, memory, latency)."""
import glob, json, statistics as st, sys
root = sys.argv[1] if len(sys.argv) > 1 else "runs/spark/quant"
for d in sorted(glob.glob(f"{root}/q-kev-*-semif")):
    r = json.load(open(f"{d}/report.json")); q = json.load(open(f"{d}/quant.json"))
    lat = [json.loads(l)["prediction"]["latency_ms"] for l in open(f"{d}/predictions.jsonl")]
    c = r["clean"]
    print(f"{d.split('/')[-1]:34s} acc={c['acc']:.4f} nll={c['nll']:.4f} ece={c['ece']:.4f} wall={q['wall_seconds']:6.1f}s load={q['load_seconds']:5.1f}s "
          f"peak={q['gpu_peak_allocated_gib']:.2f} after_quant={q['memory']['allocated_after_gib']:.2f}GiB lat_med={st.median(lat):.1f}ms "
          f"first5={[round(x) for x in lat[:5]]} max={max(lat):.0f}")
