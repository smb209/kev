import torch, time
from flashinfer import fp4_quantize, mm_fp4
torch.manual_seed(0)
dev = "cuda"
M, K, N = 128, 2048, 6144
x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
ref = x @ w.t()
def rel(a): return ((a.float() - ref.float()).norm() / ref.float().norm()).item()
# fp8 rowwise
ws = w.float().abs().amax(1, keepdim=True) / 448; wq = (w.float() / ws).to(torch.float8_e4m3fn)
def fp8(x):
    s = x.float().abs().amax(1, keepdim=True).clamp(min=1e-12) / 448
    xq = (x.float() / s).to(torch.float8_e4m3fn)
    return torch._scaled_mm(xq, wq.t(), scale_a=s, scale_b=ws.t().contiguous(), out_dtype=torch.bfloat16)
try: print("fp8 rowwise rel", rel(fp8(x)))
except Exception as e: print("fp8 rowwise FAIL", type(e).__name__, str(e)[:300])
# per-tensor
try:
    one = torch.ones((), device=dev)
    o = torch._scaled_mm(wq[:16].t().t().contiguous(), wq.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16); print("fp8 tensorwise ok", o.shape)
except Exception as e: print("fp8 tensorwise FAIL", type(e).__name__, str(e)[:300])
# nvfp4
gw = (448 * 6) / w.float().abs().amax()
wq4, wsf = fp4_quantize(w, gw, 16, False, True)
print("wq4", wq4.shape, wq4.dtype, "wsf", wsf.shape, wsf.dtype)
for backend in ("cutlass", "b12x", "cudnn"):
    def f4(x):
        ga = (448 * 6) / x.float().abs().amax().clamp(min=1e-12)
        aq, asf = fp4_quantize(x, ga, 16, False, True)
        return mm_fp4(aq, wq4.T, asf, wsf.T, 1.0 / (ga * gw), torch.bfloat16, backend=backend)
    try:
        print(backend, "nvfp4 rel", rel(f4(x)))
        for m in (1, 3, 17, 100):
            y = f4(x[:m]); print("  M", m, "ok", y.shape, rel_m := ((y.float()-ref[:m].float()).norm()/ref[:m].float().norm()).item())
        # graph capture
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            f4(x); g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s): yg = f4(x)
        torch.cuda.current_stream().wait_stream(s)
        x.copy_(torch.randn_like(x)); ref = x @ w.t(); g.replay(); torch.cuda.synchronize()
        print("  graph replay rel (new input)", rel(yg))
    except Exception as e: print(backend, "FAIL", type(e).__name__, str(e)[:400])
# fp8 under capture
try:
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fp8(x); g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s): yg = fp8(x)
    torch.cuda.current_stream().wait_stream(s)
    x.copy_(torch.randn_like(x)); ref = x @ w.t(); g.replay(); torch.cuda.synchronize()
    print("fp8 graph replay rel", rel(yg))
except Exception as e: print("fp8 graph FAIL", type(e).__name__, str(e)[:300])
