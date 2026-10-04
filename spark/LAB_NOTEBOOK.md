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

- **Kev-27B v2 serves on one DGX Spark in bf16** (E1): 72 GB GPU process, ready in 431 s cold, ~0.31 s for a
  3-question request on a short repeated state.
- **Spark bf16 reads equal the H200 release-verify reads within known kernel drift** (E2): flips 0 / 0.26 / 0.14 % on
  semif-v1 / transfer-v4 / decision-v7 dev, p99 |dp| <= 0.016; decision-v7 clean acc -0.16 pp (2 discordant, p = 0.5).

## §4 DGX-SPARKS-GUIDE.md verification

| claim (§) | verdict | evidence |
|---|---|---|
| §1 GB10, sm_121, driver 580.142, Ubuntu 24.04.4, kernel 6.17.0-1014-nvidia, 20 cores, both boxes | confirmed | nvidia-smi compute_cap 12.1, uname, lsb_release (2026-10-04) |
| §1 128 GB unified, ~121 GiB in `free` | confirmed | free -g total 121 |
| §1 root NVMe 3.7 TB, ~1.5 / 1.3 TB free | confirmed | df: 1.5T / 1.3T avail |
| §1 passwordless sudo; Docker >= 28 + nvidia runtime | confirmed | sudo -n true; Docker 29.2.1; `docker run --gpus all` works |
| §1 CUDA 13 | confirmed, with a caveat | /usr/local/cuda-13.0 present but **not on PATH** (`nvcc` not found until PATH is set) |
| §1 LAN / cluster IPs, enP7s7 on VLAN 50 | confirmed | ip -br a: 192.168.50.10/.11, 192.168.201.12/.13 on enP2p1s0f0np0, enP7s7.10 on VLAN 10 |
| §1 cluster link "MTU 9000" | **partly wrong** | enP2p1s0f0np0 (the interface holding 192.168.201.x) is **MTU 1500**; eth0 (no IPv4) is 9000; RoCE device roceP2p1s0f0 PORT_ACTIVE, active_mtu 1024. NFS/TCP over the cluster IP therefore runs at 1500 |
| §1 eth0 and enP2p1s0f0np0 "the same physical port" | unverified (plausible) | different MACs (..:9a:37 vs ..:9a:3b): separate PCIe functions; same physical QSFP cannot be checked remotely |
| §2 SSH config entries | confirmed | ~/.ssh/config |
| §2 DNS: spark-1 / spark-2 resolve to 192.168.10.30 | **not reproduced** | from the Mac today `host spark-1` = NXDOMAIN (no resolution at all). Conclusion "use the IPs" still holds |
| §2 no SSH keys between boxes | **wrong (today)** | `ssh -o BatchMode=yes` spark-1 -> 192.168.201.13 and -> 192.168.50.11, and spark-2 -> 192.168.201.12, all succeed |
| §3 sparkrun at ~/.local/bin, recipes dir, sparks / sparks-ip clusters | confirmed | files present; sparks-ip has scheduler occupancy-sparse |
| §3 warm-start dirs 0777 per recipe | mostly confirmed | ~/.cache/sparkrun-warm/* drwxrwxrwx, except spark-1 dsv41-exl3 (0775) |
| §3 "what's running now: GLM-5.3 TP=2, neither box free" | stale (expected) | user stopped it; both boxes ran only vantage-agent / cadvisor / node-exporter |
| §3 image list | mostly confirmed | images present as listed; `unsloth-dgx-spark` only on spark-2; **nvcr.io/nvidia/pytorch:26.03-py3 is not present on either box** (the ~/finetune Dockerfile names it, so a rebuild pulls ~20 GB) |
| §4 LiteLLM at 192.168.50.95:4000 | confirmed | /health/liveliness 200 |
| §5 unified-memory rules | consistent | measured: Kev-27B serve = 77 GB used + 44 GB page cache; page cache counts as "available" but `free` shows 0 GB free |
| §6 NAS automount, spark2-models NFS mount on spark-1, ~/models symlinks -> /mnt/nas-models | confirmed | mount output; symlinks present. spark-2 also has /mnt/spark2-models as a bind of its own NVMe (harmless) |
| §6 dq-runs ~584 GB on spark-2 | confirmed | du 584G |
| §7 ~/finetune Dockerfile FROM pytorch:26.03-py3, docker_memory_gb 100 | confirmed | Dockerfile + config.yaml:322 |
| §7 TRITON_CACHE_DIR: ~/.triton root-owned | **confirmed, bit us** | ~/.triton owned by root since 2026-03-02; first kev.serve died with PermissionError |
| §7 spark-2 ~/unsloth compose (8888, host net, ipc host, memlock), notebooks, snappy-agent.yaml on vllm-node-tf5 | confirmed | files |
| (not in guide) spark-2 cannot git clone from GitHub | new hazard | fetch-pack disconnect, twice; spark-1 clones fine |
| (not in guide) PyPI aarch64 torch is CPU-only | new | uv.lock's aarch64 torch 2.8 wheel is 101 MB CPU build; use download.pytorch.org cu129 |

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
- 2026-10-04T22:35Z `run` e2-bf16-decision-v7 (837 s): vs rel27-public/2: 1,468 paired (all variants), **2 flips
  (0.14 %)**, dp max 0.015 / p99 0.009 / median 0.0004. Clean rows (1,264): acc 0.8647 -> 0.8631 (-0.16 pp), 95 % CI
  [-0.35, 0.00]: two discordant questions, both against the Spark (exact McNemar 2 vs 0, p = 0.5); the bootstrap upper
  bound touches 0 rather than excluding it. ECE 0.0160 (H200) vs 0.0165.
- 2026-10-04T22:35Z `finding` **E2 pass: Spark bf16 Kev-27B v2 = H200 within known kernel drift** (pre-registered rule:
  flips <= 1 % and p99 |dp| <= 0.05 on every suite: 0 / 0.26 / 0.14 % flips, p99 0.008 / 0.016 / 0.009). Kernel set
  differs from the H200 image (report.json environment: causal_conv1d_fn = transformers torch fallback; delta rule = fla).
  kev.benchmark per-request latency median 326-334 ms on the Spark vs 113-118 ms on the H200 (benchmark path, not
  the fused serving path; E3 measures that).
- 2026-10-04T22:40Z `check` spark/train_fla_conv.py routes transformers' Qwen3.5 causal_conv1d_fn to fla's Triton
  causal_conv1d for training. vs the torch fallback (B2 D1536 T777, silu): fp32 rel err out 5e-8, dx 7e-8, dw 2e-7;
  bf16 3e-3 (rounding). Motivation: NVIDIA forum thread cited in spark-1:~/finetune/docker/Dockerfile reports fla +
  causal-conv1d as the largest single win (~6x) for Qwen3.5 LoRA on GB10 (reference; their numbers, not ours).
- 2026-10-05T00:20Z `run` e5-ab (spark-2, GPU otherwise idle, 40 steps each of the r15-08b config, same seed, back to
  back): torch conv fallback 1.287 / 1.079 / 0.928 / 0.954 s/rec cumulative at steps 10/20/30/40 (steps 30-40: 1.03
  s/rec); fla conv (spark/train_fla_conv.py) 2.901 / 2.274 / 1.838 / 1.678 (steps 30-40: 1.20 s/rec, still falling:
  Triton autotune per new shape). Losses identical to 3 decimals at every logged step.
- 2026-10-05T00:20Z `finding` (against my hypothesis) **the causal conv fallback is not what makes Spark training slow**:
  swapping in fla's kernel did not speed up steady-state steps. The forum's ~6x came from a setup without fla's delta
  rule; ours already binds fla's chunk_gated_delta_rule (report.json environment). Next: profile a step.
- 2026-10-05T00:40Z `run` e3-serving-27b-bf16 (spark-1, scripts/serving_bench.py --reference none, bf16 fused + CUDA
  graphs; runs/spark/e3-serving-27b-bf16/report.json) vs runs/fused-27b-h200: load 317 s (H200 19 s), resident 65.5 GB
  (same), 196 graphs captured, 0 failed. Model latency new / cached state, graphs: 2 q short 588 / 304 ms (H200 40 / 22);
  6 q short 844 / 554 (66 / 48); 5 q 370-token 995 / 571 (88 / 51); 5 q 2,200-token 2,510 / 693 (268 / 72). Eager is
  the same within ~5 % (594 / 304 on 2 q). Throughput, decision-v7 dev requests: 1.6 / 2.7 / 3.4 / 3.8 req/s at 1 / 8 /
  32 / 64 clients (H200 21.6 / 31.7 / 36.4 / 39.7); 2,200-token states 0.4 req/s flat. Graphs vs eager bf16 parity: 1 flip
  in 280 q, max dp 0.021.
- 2026-10-05T00:40Z `finding` **bf16 Kev-27B on a Spark is weight-bandwidth bound**, ~10-15x slower than an H200 per
  request: graphs do not help (not launch bound) and a cached-state request (one pass) takes 304 ms against a floor of
  51 GB / 224 GB/s = 228 ms; a new state (two passes) 588 ms against 456. Prediction for E4 (not a gate): NVFP4 weights
  (~14 GB) cut the floor to ~65 ms per pass, so 2-4x faster short requests.
- 2026-10-05T00:30Z `check` py-spy (30 s, e5 training step) + nvidia-smi: GPU 96 % "utilization" at only 29-33 W;
  samples spread over Linear forwards (17 % leaf), grad-norm (9 %), checkpoint recompute (19 % incl.), fla delta-rule
  fwd/bwd (~6 %). No single hotspot. Hypothesis (unproven): small-model LoRA training is memory-bandwidth bound; Spark /
  H200 bandwidth ratio ~1/21 (224 GB/s vs 4.8 TB/s) matches the observed ~1/24 speed better than the compute ratio
  (~1/12). Recipe-preserving speedups are unlikely; recipe-changing ones (--length_sort, shorter max_state) are E6.
- 2026-10-05T00:45Z `run` e5-r15-08b-s1 started on spark-2 (full reproduction, spark/e5_train.sh, recipe unchanged).
