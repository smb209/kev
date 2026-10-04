# Kev on DGX Spark — lab notebook

Questions: (Q1) can Kev-27B v2 serve on one DGX Spark (GB10, 128 GB unified), how fast, and does it answer the
same as the H200 reference; (Q2) can a quantized Kev-27B (NVFP4 ideally) run there, and what does it cost in
accuracy; (Q3) can the Sparks fine-tune the small Kevs (0.8B / 4B / 9B LoRA) on our own data.

Boxes: spark-1 (192.168.50.10), spark-2 (192.168.50.11). Branch `spark-investigation` on the fork `smb209/kev`
(nothing goes upstream). Env: `~/kev/.venv-spark` on spark-1 (built by `spark/env.sh`).

## §1 Where we are

2026-10-04. Both Sparks idle. Env up on spark-1: torch 2.8.0+cu129 (aarch64) runs on the GB10 (sm_121) with a
"max supported 12.0" warning; triton 3.8.0, fla 0.5.2. Microbenchmarks: 85 bf16 TFLOPS (8k matmul), 224 GB/s
device copy. Kev-27B v2 downloading. No Kev number yet.

## §2 Open items, ranked

1. E1: Kev-27B v2 bf16 loads and serves on one Spark (memory headroom).
2. E2: parity of the Spark bf16 read against the H200 release-verify rows (decision-v7 dev, transfer-v4 dev, semif-v1).
3. E3: serving latency / throughput (scripts/serving_bench.py) vs runs/fused-27b-h100/h200.
4. E4: quantization (FP8 / NVFP4) of Kev-27B v2 built here from the official bf16 weights; accuracy vs E2's bf16 read.
5. E5: LoRA fine-tune of Kev-0.8B / 4B on the Spark (throughput, memory, a smoke-quality delta).
6. Guide verification (§4).

## §3 Closed findings

(none yet)

## §4 DGX-SPARKS-GUIDE.md verification

| claim (§) | verdict | evidence |
|---|---|---|

## Pre-registration

### Pre-registered read conditions (written 2026-10-04, before any Kev read on a Spark)

**E1 feasibility.** Pass = `Checkpoint.load("cuda", bf16)` of `jaredpalmer/kev-27b@28be62e9` completes and a
/v1/systemone request answers, with system `MemAvailable` never below 15 GB during load + one decision-v7 read
(the guide's hang threshold is ~100 % of the shared pool). Fail modes recorded as-is (OOM, hang, kernel errors).

**E2 parity (bf16 Spark vs bf16 H200).** Reference rows: `runs/rel27-public/{0,1,2}-jaredpalmer_kev-27b/rows.json`
(semif-v1 dev 252 q, transfer-v4 dev, decision-v7 dev 1,264 clean q; Modal H200, release verify 2026-09-30).
Known, unremovable confound: different kernel set (sm_121 vs sm_90; this env has no causal-conv1d package, which
the H200 image had; runs/drift-v1 measured that swap alone at max |dp| 0.03-0.06, 0-0.4 % flips on Kev-27B v1).
Primary metric: argmax flip rate per suite over paired questions; secondary: max / p99 |dp| at served T, and the
accuracy delta (paired, record-clustered bootstrap from kev.compare).
- flips <= 1 % on every suite and p99 |dp| <= 0.05 -> "equivalent within known kernel drift".
- flips 1-2 % -> report, investigate the kernel differences before quoting Spark accuracy.
- flips > 2 % or any accuracy delta whose 95 % CI excludes 0 -> the Spark path is wrong; no serving claim.

**E3 latency.** Report, no gate. Physics floor for one bf16 pass = weight read 51 GB / 224 GB/s = 0.23 s; prefill
compute 54 GFLOP/token / 85 TFLOPS = 0.64 ms/token. Expect ~0.3-0.5 s for a short request at concurrency 1.

**E4 quantization (written 2026-10-04 ~22:10Z, before any quantized read).** Arms on Kev-27B v2, all in ONE environment
(image kev-spark-quant, torch 2.13+cu130): bf16 (same-env baseline), fp8, nvfp4-mlp (MLP NVFP4, attention/DeltaNet FP8),
nvfp4 (all eligible Linears). Reads: decision-v7 dev + transfer-v4 dev + semif-v1 dev (all variants, as E2). Comparison
is each quant arm vs the same-env bf16 arm (env drift isolated by also pairing same-env bf16 vs H200).
- MDE: paired delta SE ~ sqrt(flip_rate / n); at 3 % flips and ~2,280 pooled questions SE ~0.36 pp -> MDE ~1.0 pp (80 %
  power, two-sided). Effects below ~1 pp are not resolvable; say so rather than calling them zero.
- Primary (verdict): pooled accuracy delta vs same-env bf16. "Serving-grade" = point estimate >= -1.0 pp AND 95 % CI
  lower bound >= -2.0 pp AND pooled ECE increase <= 0.01. Point estimate < -1.0 pp -> not serving-grade.
- Context, not verdict: per-suite deltas, flips, |dp|, latency (concurrency 1 and 8) and memory.
- Literature anchor (SVPG card, 5090, not our evidence): nvfp4-mlp-style mix -0.07 pp pooled; expect RTN (no Hessian
  calibration) to be worse than that. Plain W4A4 nvfp4 everywhere may fail the gate; that would be a real result.

**E5 fine-tune reproduction (written 2026-10-04 ~22:10Z, before the full run).** Rerun of round 15's Kev-0.8B stage
(r15-08b/00-trial-0: init kev-0.8b@night2-du-release, joint data sha 1bf25e29, replay 6000 decision-v7, lr 2e-5, seed 1)
on spark-2, one seed (n=1, pre-committed; the H200 trial is also n=1, so seed variance is unknown: the tolerance below is
a stand-in for it, and a pass means "no evidence the Spark path differs", not equality).
- Feasibility: completes under the 100 GB memory cap with no box hang; record wall time, s/record, peak memory.
- Reproduction: Spark checkpoint vs the H200 checkpoint's committed rows on decision-v7 dev (runs/r15-08b/00-trial-0),
  documents-v1 dev, hard-v1 dev, devtools-v1 dev (runs/r15-08b-a-*). Pass = every panel |acc delta| <= 2 pp with the 95 %
  CI including 0. Any panel beyond 2 pp with CI excluding 0 -> the Spark training path differs; investigate before use.
- Effect check: init (night2) -> Spark-trained gain on hard-v1 dev and documents-v1 dev >= 80 % of init -> H200 gain
  (init read on the Spark too, so the gain is same-environment).

## §5 Chronological log

- 2026-10-04T20:55Z `tool` env: uv venv py3.13 on spark-1, torch 2.8.0+cu129 aarch64 (PyPI's aarch64 torch 2.8 in uv.lock
  is the 101 MB CPU wheel, so the lockfile cannot be used as-is on the Spark; installed from download.pytorch.org cu129 then
  `uv pip install -e .[serve]`), fla 0.5.2, triton 3.8.0. torch warns GB10 sm_121 > max 12.0 but runs (sm_120 SASS). No
  causal-conv1d package. Script: spark/env.sh.
- 2026-10-04T20:58Z `check` microbench spark-1: bf16 8192^3 matmul 85.4 TFLOPS; device-to-device copy 224 GB/s.
- 2026-10-04T21:00Z `hazard` spark-2 cannot `git clone` from GitHub (fetch-pack: unexpected disconnect, twice); spark-1
  can. Worked around with a git bundle scp'd from the Mac.
- 2026-10-04T21:10Z `check` jaredpalmer/kev-27b downloaded to spark-1 HF cache (snapshot af0e6d55, a README-only commit
  after 28be62e9); `Checkpoint.weights_sha256` = d27af6ab... = the release record's weights. ~70-135 MB/s unauthenticated.
- 2026-10-04T21:15Z `hazard` first `kev.serve` died during load: PermissionError on ~/.triton/cache (root-owned since
  2026-03-02). Fix: TRITON_CACHE_DIR=$HOME/kev/.triton-cache (guide §7 claim confirmed).
- 2026-10-04T21:22Z `finding` **E1 pass.** `kev.serve --run jaredpalmer/kev-27b` (bf16, fused, CUDA graphs on by default)
  on spark-1: ready in 431 s cold (includes Triton compiles). GPU process 72,179 MiB; system used 77 GB, page cache 44 GB
  (reclaimable), MemAvailable min 45 GB during load (sampler runs/spark/e1-mem.csv) -> pre-registered 15 GB floor held.
  README example request (71 input tokens, 3 questions): first 7.8 s (compile), then 0.31-0.32 s (one 0.61 s outlier) over
  5 repeats of the same state (prefix cache likely hit). Answers: billing noul 0.983, team billing 0.999, urgency Urgent 0.954.
- 2026-10-04T21:50Z `tool` spark/parity.py (pairs rows.json on (id, question), recomputes p at the reference T from
  logits*T_row, record-clustered bootstrap of the acc delta). **Correction** to my first version: it divided the saved
  logits by T a second time (rows' logits are already at T: softmax(logits) == p); caught by positive control 2 below.
  Controls (all variants, semif-v1 dev 252 q): ref vs itself 0 flips / dp 0; ref vs runs/r23-27b-k-w85-semif (release
  record says logits exact) 0 flips / dp 0; negative control ref (v2) vs v1-lora 5 flips (1.98 %), dp max 0.62.
- 2026-10-04T21:50Z `run` e2-bf16-semif-v1 (spark-1, kev.benchmark, 468 s incl. ~7 min load): vs H200 rel27-public/0:
  252 paired, **0 flips**, dp max 0.0135 / p99 0.0078 / median 0.0003, acc 0.9802 = 0.9802.
- 2026-10-04T22:00Z `run` e2-bf16-transfer-v4 (634 s): vs rel27-public/1: 764 paired, **2 flips (0.26 %)**, dp max 0.036 /
  p99 0.016 / median 0.0005, acc 0.8469 = 0.8469 (the 2 flips cancel).
- 2026-10-04T21:40Z `check` community quants of Kev-27B on the Hub (none official): SVPG-Labs/kev-27b-NVFP4 (v2, ModelOpt
  NVFP4 MLP + FP8 attention/DeltaNet, a 19.4 GB pickle `backbone.pt` that its README says loads with weights_only=False ->
  arbitrary code; **not loaded**, policy), bowmanslayer/kev-27b-W4A16 (v1 LoRA merged, AutoRound int4, "inference not
  validated"), TheCulliganMan/kev-27b-nf4 (v1 adapter + nf4 base). Decision: build our own FP8 / NVFP4 from the official v2
  bf16 weights (E4); the SVPG card's numbers (5090, pooled -0.07 pp vs bf16) are a literature anchor, not our evidence.
- 2026-10-04T21:45Z `check` FlashInfer 0.6.18 NVFP4 GEMM on GB10 (in sparkrun-eugr-vllm-tf5, torch 2.13+cu130):
  M512 K5120 N17408: cutlass 248-254 TFLOPS vs bf16 78 TFLOPS (3.2x); cudnn 57; trtllm unsupported on sm_121; rel err vs
  bf16 0.134 on gaussian data (W4A4 RTN). Implementation of spark/kev_quant.py delegated (implementer, spark-2 only).
- 2026-10-04T21:55Z `check` E5 positive-control data: round 15 joint train file rebuilt byte for byte
  (spark/build_r15_joint.py; sha256 1bf25e29... = archive manifest; skills part 14db86b9... ok).
- 2026-10-04T22:05Z `run` e5-probe (spark-2, 60 steps of the r15-08b config, spark/e5_train.sh): 2.50 s/rec cumulative at
  step 10, 1.24 at step 60 -> steps 10-60 ~0.99 s/rec (GPU shared with the quant subagent from ~step 15). H200 trial:
  0.103 s/rec (3,136 s / 30,329 records). Projected full epoch ~8.3 h. transformers warns causal_conv1d falls back to
  torch F.conv1d (package not installed; no aarch64 wheel; not in any Spark image checked). Estimated utilization ~5 % of
  85 TFLOPS -> overhead-bound; profile before the full run.
