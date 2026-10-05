"""scripts/serving_bench.py with a kev_quant scheme applied at load (fused + CUDA graphs as serving uses them).
    python spark/quant_serving_bench.py <scheme> <serving_bench args...>"""
import runpy, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kev_quant as kq
scheme = sys.argv[1]
kq.Nvfp4Linear.backend = "cutlass"
kq.set_act_quant("torch")
kq.install(scheme, (), None, None, "static", 4.0, 64, None)   # fused / graphs: whatever serving_bench's LoadOptions ask for
sys.argv = ["serving_bench.py", *sys.argv[2:]]
runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "serving_bench.py"), run_name="__main__")
