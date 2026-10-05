# Kev on DGX Spark — lab notebook

Questions: (Q1) can Kev-27B v2 serve on one DGX Spark (GB10, 128 GB unified), how fast, and does it answer the
same as the H200 reference; (Q2) can a quantized Kev-27B (NVFP4 ideally) run there, and what does it cost in
accuracy; (Q3) can the Sparks fine-tune the small Kevs (0.8B / 4B / 9B LoRA) on our own data.

Boxes: spark-1 (192.168.50.10), spark-2 (192.168.50.11). Branch `spark-investigation` on the fork `smb209/kev`
(nothing goes upstream). Env: `~/kev/.venv-spark` on spark-1 (built by `spark/env.sh`).

## §1 Where we are

2026-10-05 21:30Z. All planned experiments done and reviewed: E1-E4b (27B serving, parity, quantization), E5
(fine-tune reproduction), E6-E7 (fine-tune throughput), E8-E9 (Clef vs Kev; E9 on breadth-v1 rebuilt locally, rows
never pushed). Nothing running. Open: a write-up; optional next experiments (§2).

## §2 Open items, ranked

1. Write-up of the investigation for sharing.
2. Fine-tuning Clef-Flash on our data (does its release ship training code? same E5-style reproduction impossible
   without a reference run; would need its own pre-registration).
3. A fair latency comparison Clef vs Kev on identical request shapes through one serving harness (E8 latencies are
   not comparable: different paths).
4. NVFP4 calibration at larger n (ECE unresolvable at 2,484 q) and long states (> 2.2k tokens) under quantization.
5. Kev-9B NVFP4 (only 27B and 4B were quantized).

## §3 Closed findings

- **Kev-27B v2 serves on one DGX Spark in bf16** (E1): 72 GB GPU process, ready in 431 s cold, ~0.31 s for a
  3-question request on a short repeated state.
- **Spark bf16 reads equal the H200 release-verify reads within known kernel drift** (E2): flips 0 / 0.26 / 0.14 % on
  semif-v1 / transfer-v4 / decision-v7 dev, p99 |dp| <= 0.016; decision-v7 clean acc -0.16 pp (2 discordant, p = 0.5).
- **bf16 Kev-27B on a Spark is weight-bandwidth bound, ~10-15x slower than an H200** (E3): 588 / 304 ms new / cached
  2-question requests, 3.8 req/s at 64 clients.
- **NVFP4 (our runtime, static scales) cuts serving latency 2.1-3.5x and memory 65.5 -> 30.6 GB** (E4 serving), and on
  the fused path loses no more than ~0.7 pp pooled accuracy (E4b: -0.16 pp [-0.70, +0.36]); FP8 (unfused read) no more
  than ~0.4 pp. ECE not resolvable at n 2,484 (E4 review correction).
- **The causal-conv fallback is not why Spark training is slow** (E5 A/B); training runs ~0.72 s/record for Kev-0.8B,
  ~7x slower than the H200 trial (0.103); cause open (the bandwidth-bound explanation was retracted 05:45Z).
- **Spark LoRA training reproduces the released Kev-0.8B round-15 stage** (E5): all four dev panels within 0.2 pp of
  the H200 checkpoint, 99-100 % of its gains; 6.2 h vs 52 min, same 23.9 GB peak.
- **Clef vs Kev** (E8, E9, both reviewed): on transfer-v9 (Kev-held-out) Clef ~= Kev-27B (-0.3 pp [-2.7, +2.1]),
  Clef-Flash +2.0 pp [-0.4, +4.5] over Kev-9B; on breadth-v1 (Decision-Index datasets) Clef +10.5 [7.5, 13.6] index
  points over Kev-27B, Clef-Flash +22.4 [19.3, 25.9] over Kev-9B (~+5 / +14 without the five most suspect datasets);
  the card's directions reproduce on 8/14 and 6/14 datasets, its magnitudes do not. Kev leads 4-16 pp on its own
  training distributions. Clef's training data is undisclosed.
- **Kev-4B LoRA fine-tuning fits a Spark (peak 20-56 GB)**; the kev-finetune skill's default job shape takes ~38 min
  (~27 min with --length_sort 1) (E6, E7).

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
- 2026-10-05T01:30Z `run` E4 arm bf16 (kev-spark-quant image, torch 2.13+cu130, benchmark path unfused): vs H200
  rel27-public: semif 0 flips / 252 (p99 0.008), transfer-v4 2 / 764 (0.26 %, p99 0.013), decision-v7 1 / 1,468 (0.07 %,
  p99 0.008). The same-env baseline is itself within E2's drift band. Container MemAvailable min 67 GB.
- 2026-10-05T01:35Z `tool` spark/e4_readout.py (pooled verdict per the E4 pre-registration). Positive control bf16 vs
  itself: delta 0, CI [0, 0], ECE delta 0, n 2,484 questions in 1,496 (suite, group) clusters.
- 2026-10-05T03:20Z `run` E4 arms fp8 / nvfp4-mlp / nvfp4 (spark-1, one load per arm, container MemAvailable min 66 GB).
  spark/e4_readout.py (runs/spark/e4/readout.json), pooled 2,484 q vs same-env bf16 (acc 0.8780, ECE 0.0087):
  fp8 -0.12 pp [-0.40, +0.12], ECE 0.0114, discordant +4/-7; nvfp4-mlp -0.40 pp [-0.92, +0.08], ECE 0.0066, +8/-18;
  nvfp4 -0.36 pp [-0.96, +0.24], ECE 0.0128, +19/-28. Per suite (flips / p99 dp / max dp): nvfp4 transfer-v4 22/764
  (2.9 %), 0.20, 0.53; decision-v7 24/1,468, 0.13, 0.20; nvfp4-mlp transfer-v4 15/764, 0.34, 0.64. kev.benchmark median
  request latency (unfused eager, sequential): bf16 287, fp8 192, nvfp4-mlp 147, nvfp4 123 ms.
- 2026-10-05T03:20Z `finding` (pending adversarial review) **all three quant arms meet the pre-registered serving-grade
  gate.** Resolution caveat: CI lower bounds reach -0.92 / -0.96 pp, so the NVFP4 arms are "no loss beyond ~1 pp", not
  "no loss"; single-question probabilities move by up to 0.5-0.6 under NVFP4 (thresholds must be re-frozen per scheme).
- 2026-10-05T03:50Z `review` adversarial review of the E4 claim (subagent; full text in session; recompute script was
  scratch). No arithmetic, pairing or leakage error (deltas, discordants, CIs recomputed from rows; calibration records
  share no id / group / text hash / state with any read). Three problems accepted:
  (a) **correction to my pre-registration**: the ECE criterion had no power. A perfectly calibrated model at bf16's
  confidences scores ECE 0.0122 mean / 0.0184 p95 on n 2,484 (10-bin kev.metrics.ece; I re-simulated, 2,000 draws:
  0.0122 / 0.0184). Every observed ECE is at or below that floor; paired bootstrap of the nvfp4 ECE delta [-0.006,
  +0.013]. ECE is uninformative at this n; the gate's ECE clause is withdrawn as a criterion (kept as context).
  (b) the CI clause (lower >= -2 pp) cannot bind at SE ~0.3 pp, so the gate was in effect "point >= -1 pp"; a true 1 pp
  loss passes ~50 % of the time. Losses up to ~1 pp pooled (~2 pp on transfer-v4, the OOD suite: nvfp4 -0.92 pp, CI to
  -2.2, 2.9 % flips) are not excluded. Clean-only reading does not change the verdict (nvfp4-mlp lower bound -1.08).
  (c) accuracy and latency were read on kev.benchmark's unfused eager path, not the fused + CUDA-graph serving path.
  Minor: semif variant rows carry a different group than their parents (1,496 clusters instead of 1,460); re-clustered
  nvfp4 CI [-0.94, +0.21], immaterial.
- 2026-10-05T03:50Z `finding` (revised) **E4: on kev.benchmark's unfused path, FP8 / NVFP4-MLP / NVFP4 Kev-27B v2 pass
  the pre-registered point gate: pooled -0.12 / -0.40 / -0.36 pp vs same-env bf16, CIs reaching -0.40 / -0.92 / -0.96
  pp. FP8 excludes losses beyond ~0.4 pp; NVFP4 only beyond ~1 pp (OOD suite ~2 pp). ECE not resolvable at n 2,484.**
  Supersedes the 03:20Z wording ("serving-grade").

**E4b fused-path read (pre-registered 2026-10-05T03:55Z, before the data).** Arms bf16 and nvfp4, both with
kev.fused_qwen35 (the kernels kev.serve runs; CUDA graphs only replay the same arithmetic in bf16 up to reassociation),
same image, same three suites, all variants. Primary: pooled acc delta nvfp4-fused vs bf16-fused, with its CI. Claim
rule, stated as resolution not as a gate: report the interval; "no loss beyond X pp" with X = -(CI lower bound). Also
report nvfp4-fused vs nvfp4-unfused flips (does fusion change the quantized answers?). ECE reported, not judged.
- 2026-10-05T04:30Z `run` e4-serving-27b-{nvfp4,nvfp4-mlp} (spark-1, scripts/serving_bench.py through
  spark/quant_serving_bench.py, fused + CUDA graphs, static act scales; compare runs/spark/e3-serving-27b-bf16):
  resident 30.6 / 33.8 GB (bf16 65.5); load 384 / 406 s; 198 graphs, 0 failed. Model latency new / cached, NVFP4: 2 q
  short 168 / 91 ms (bf16 588 / 304: 3.5x / 3.4x); 6 q short 274 / 192 (3.1x / 2.9x); 5 q 370-token 349 / 208 (2.9x /
  2.8x); 5 q 2,200-token 1,172 / 328 (2.1x / 2.1x). NVFP4-MLP: 204 / 110, 323 / 222, 405 / 239, 1,338 / 361.
  Throughput decision-v7 requests at 1 / 8 / 32 / 64 clients: NVFP4 5.3 / 8.0 / 9.4 / 10.3 req/s; NVFP4-MLP 4.4 / 6.8 /
  8.0 / 8.8; bf16 1.6 / 2.7 / 3.4 / 3.8. 2,200-token states: 0.9 / 0.8 / 0.4 req/s.
  Graphs vs eager (same scheme): NVFP4 max dp 0.189, mean 0.016, 1 flip / 280 q; NVFP4-MLP 0.203 / 0.014 / 5 flips;
  bf16 0.021 / 0.0014 / 1 flip. Under FP4 the bucket padding of the graph path moves answers ~10x more than in bf16
  (same mechanism as the isolation check: batch-shape bf16 noise amplified by FP4 rounding). E4b reads the fused path
  without graphs; this graph-vs-eager spread is an additional serving-path noise term it does not cover.
- 2026-10-05T05:40Z `run` E4b (spark-1, kev.fused_qwen35 on, no graphs; runs/spark/e4/*-fused-*, readout-fused.json):
  nvfp4-fused vs bf16-fused pooled 2,484 q: acc 0.8784 -> 0.8768, **-0.16 pp, CI [-0.70, +0.36]**, discordant +18/-22;
  ECE 0.0099 -> 0.0175 (inside the simulated noise band, mean 0.012 / p95 0.018; reported, not judged). Per suite:
  semif -1.19 pp [-3.16, 0.00] (3 flips, n 252), transfer-v4 -0.39 [-1.33, +0.55] (12), decision-v7 +0.14 [-0.55, +0.83]
  (26). Path effect: bf16 fused vs unfused 0 / 0 / 1 flips (p99 dp <= 0.012); **nvfp4 fused vs unfused 2 / 18 / 17 flips
  (1.5 %), p99 dp 0.10-0.19**: the quantized answers depend on the kernel path about as much as on quantization itself.
- 2026-10-05T05:40Z `finding` **E4b: on the fused serving path NVFP4 Kev-27B v2 loses no more than ~0.7 pp pooled
  accuracy vs bf16 (point -0.16 pp)**, per the pre-registered resolution rule. NVFP4's divergence from bf16 (1.6 % flips)
  is the same size as its divergence between two kernel paths (1.5 %), i.e. it behaves like rounding noise, with no
  systematic loss detectable at this n; E4 (-0.36) and E4b (-0.16) agree within it. Not covered: the graph path's extra
  spread (max dp 0.19 vs eager), states > 2.2k tokens, and calibration (ECE not resolvable at n 2,484).
- 2026-10-05T05:45Z `retraction` of the 00:30Z training hypothesis ("small-model LoRA training is memory-bandwidth
  bound; ratio ~1/24 matches bandwidth ~1/21"). The ~1/24 came from the probe (GPU shared with a subagent, cold Triton
  caches). The uncontended full run e5-r15-08b-s1 averages 0.749 s/rec at step 1,160 = **7.3x the H200 trial (0.103)**,
  below both the bandwidth (~21x) and compute (~12x) ratios, so the H200 trial was itself far from either roof (small
  model, overhead-bound) and no roofline explains the gap. Training-speed cause: open (low priority; 7x is usable).
  The 00:30Z entry stays as written; read it through this one.
- 2026-10-05T06:30Z `run` E5 init reads on spark-1 (kev-0.8b@night2-du-release, venv, exact fp32 path,
  spark/e5_reads.sh; runs/spark/e5/init-*): decision-v7 0.8252, documents-v1 0.6326, hard-v1 0.3490, devtools-v1 0.4879.
  H200 parent reads (runs/night2-08b-du2 result.json, runs/dt1-P08, runs/r15-readout/round15.json parent accs): 0.8252 /
  0.6326 / 0.34995 / 0.4869 (round15 readout; dt1-P08 report 0.4879 on 1,074 q). Same to 4 decimals: the fp32 0.8B read
  reproduces across H200 and GB10.
- 2026-10-05T06:30Z `reference` (found before the E5 result exists; does not change the E5 rule) H200 seed variance for
  this exact recipe: round 15 arm 08b-c = trial 2 (seed 2, same config). Seed 1 vs seed 2 dev accs: hard-v1 0.5937 vs
  0.5845 (-0.9 pp), documents-v1 0.8424 vs 0.8348 (-0.8), devtools-v1 0.6017 vs 0.5849 (-1.7). n = 2 seeds, so a range,
  not an sd: the pre-registered +/-2 pp tolerance is about one seed-to-seed difference. Read E5's deltas against it.

**E6 fine-tune throughput options (2026-10-05T06:40Z, report only, no gate).** 40-step probes of the r15 joint recipe
on spark-1 (spark-2 runs E5): (a) Kev-4B delta (init jaredpalmer/kev-4b, base Qwen3.5-4B-Base 1001bb4d), default
batching; (b) Kev-4B with --length_sort 1; (c) Kev-0.8B with --length_sort 1 (vs E5's 0.745 s/rec on identical
hardware). Read: steady s/rec over steps 20-40 and peak memory. --length_sort changes micro-batch composition, so
losses are not comparable across arms; it is a recipe change, to be validated by accuracy before adoption.
- 2026-10-05T08:10Z `hazard` first E6 launch failed at once: evals/round15/joint/train.jsonl existed only on spark-2.
  Rebuilt on spark-1 (sha 1bf25e29 ok) and relaunched.
- 2026-10-05T08:40Z `run` E6 (spark-1, 40 steps = 454 records, 257,756 forward tokens each; training_metrics.json):
  4B default 3.79 s/rec cumulative, wall 1,720 s incl. load, peak device 56.3 GB (MemAvailable min 43 GB);
  4B --length_sort 1 1.72 s/rec, wall 779 s, peak 37.5 GB (2.2x faster, -19 GB); 0.8B --length_sort 1 0.65 s/rec, wall
  296 s, peak 11.0 GB (vs E5's default 0.745 s/rec, ~13 % faster). Losses at logged steps within 0.01 between 4B arms.
  Projected one epoch of the 30k-record r15 corpus: 4B ~32 h default / ~14 h length-sorted; 0.8B ~6.3 h / ~5.5 h.
- 2026-10-05T08:40Z `finding` **Kev-4B LoRA training fits a Spark comfortably (peak 37-56 GB of 128) and
  --length_sort 1 halves its time on long-document data**; the gain is from padding (long, mixed-length states), so it
  shrinks for short-state workloads (0.8B: 13 %). --length_sort is a recipe change: unvalidated for accuracy here.
- 2026-10-05T09:20Z `run` E7 skill-shaped timing (spark-1, timing only, checkpoints discarded): the kev-finetune
  skill's default job shape = 968 short-state records (decision-v7 calibration partition as a stand-in for user data) +
  --replay 2000, 1 epoch, Kev-4B delta, lr 2e-5, batch 4 x accum 2, 60 steps (592 records incl. none-pair siblings,
  115k forward tokens): default 0.627 s/rec, wall 371 s, peak 24.8 GB; --length_sort 1 0.439 s/rec, wall 260 s, peak
  20.2 GB. Full job ~2,968 requests -> ~3,650 records seen -> **~38 min (default) / ~27 min (length-sorted) on one
  Spark** + ~2-5 min load (skill's figure on an H100: ~15 min for 1,000 records).

**E8 Clef vs Kev (pre-registered 2026-10-05T11:10Z, before any Clef read).** Cloudflare/clef (27B, Qwen3.8-27B +
vision, revision 2f3de3dd) and Cloudflare/clef-flash (9B, Qwen3.5-9B + vision, revision 17f0b0ad): verified HF org,
Apache-2.0, safetensors weights, custom `joint_schema_model.py` (sha256 0e304cf7..., identical in both repos) reviewed by
hand: imports json/math/torch/safetensors/transformers only, no pickle, no trust_remote_code, no network beyond
snapshot_download; run pinned to that hash. Their card's "Kev 9B" column is Cloudflare's own run on their own
leaderboard (Decision Index 0.2.1); their training data is undisclosed, so contamination of any public set is unknown.
- Pairs: Clef vs Kev-27B v2; Clef-Flash vs Kev-9B v2 (jaredpalmer/kev-9b@main). All bf16 on a Spark, same harness:
  kev.benchmark; Kev arms in-process (LocalPredictor), Clef arms through kev.benchmark --remote against a local
  /v1/systemone server wrapping their encode/model with full-precision probabilities (spark/clef_server.py).
- **Primary (verdict): transfer-v9 development** (1,264 records; eval-only in Kev by construction: Kev never trained on
  any of its sources; Clef's exposure unknown). Paired accuracy delta, record-clustered bootstrap (spark/parity.py).
  "Clef better OOD" iff the 95 % CI excludes 0 in its favour; "Kev better" symmetric; else "no detectable difference".
  MDE: between different models discordance is ~10-20 %, SE ~ sqrt(0.15 / 1,264) ~ 1.1 pp -> MDE ~3 pp.
- Context, labelled by home field: decision-v7 / hard-v1 / devtools-v1 / documents-v1 dev are Kev training
  distributions (Kev home field; differences there say little about generality). semif-v1 is saturated (report only).
- Also reported, not judged: ECE (noise floor per E4 review), latency on the Spark at concurrency 1 (Clef through its
  own eager path; Kev through kev.serve E3 numbers), memory. Image input is out of scope for Kev, so not compared.
- 2026-10-05T12:30Z `run` e5-r15-08b-s1 finished (spark-2): 2,818 steps, 30,329 records, 15,531,569 forward tokens
  (H200 trial: 15,531,716; 30,329; 2,818), **wall 22,241 s = 6.2 h (H200 3,136 s; 7.1x)**, peak device 23.9 GB (H200
  23.9 GB), MemAvailable min 78 GB, no hang. Reads (spark/e5_reads.sh, runs/spark/e5/spark-*) vs the H200 checkpoint's
  rows (clean): decision-v7 0.8267 -> 0.8275 (+0.08 pp [-0.40, +0.55], 11 flips); documents-v1 0.8424 -> 0.8413 (-0.11
  [-0.76, +0.54], 9); hard-v1 0.5937 -> 0.5919 (-0.18 [-1.28, +0.91], 48 flips, 4.4 %); devtools-v1 0.6021 -> 0.6030
  (+0.09 [-0.73, +0.90], 28; 1,073 paired: the suite's duplicated CodeReviewer id drops one row).
- 2026-10-05T12:30Z `finding` **E5 pass: Spark LoRA training reproduces the released Kev-0.8B stage.** Pre-registered
  rule: every panel |delta| <= 2 pp with CI including 0 -> all four within 0.2 pp, an order of magnitude inside the H200
  seed1-vs-seed2 spread (0.8-1.7 pp). Effect check (same-environment gains from the Spark init read): hard-v1 +24.3 pp
  (H200 +24.4, 100 %), documents-v1 +20.9 pp (H200 +21.0, 99 %); bar was >= 80 %. Forward-token count differs from the
  H200 run by 147 of 15.5M (augmentation RNG on a different torch build), so the runs are not bit-identical, as expected.
- 2026-10-05T14:30Z `tool` spark/parity.py: when either read lacks logits (kev.benchmark --remote saves p only), both
  sides are compared on their served probabilities (each at its own T) and the output says so. Controls re-run
  (ref vs itself 0 / 0; ref vs e2-bf16-semif 0 flips, p99 0.0078: unchanged).
- 2026-10-05T14:30Z `run` E8 arms (spark-1: clef-flash, clef via spark/clef_server.py in kev-spark-quant, kev.benchmark
  --remote; spark-2: kev-9b, kev-27b in venv; kev-27b semif / decision-v7 reused from E2, same venv build). Clef loads in
  372 s, 51.2 GiB GPU; Clef-Flash 140 s, 17.8 GiB. Report clean accs (n): transfer-v9 (1,046) Kev-27B 0.8212, Clef
  0.8184, Kev-9B 0.7801, Clef-Flash 0.8002; semif (144) 0.9653 / 0.9306 / 0.9167 / 0.9028; decision-v7 (1,264) 0.8631 /
  0.8774 / 0.8742 / 0.8853; hard-v1 (1,083) 0.9104 / 0.7572 / 0.8126 / 0.6491; devtools-v1 (1,074) 0.7551 / 0.7188 /
  0.7728 / 0.6611; documents-v1 (920) pending / 0.8793 / 0.9022 / 0.8587.
- 2026-10-05T14:30Z `finding` **E8 primary (transfer-v9 dev, all 1,264 rows, paired, served probabilities): no
  detectable difference in either pair.** Clef vs Kev-27B: 0.7785 vs 0.7816, -0.32 pp [-2.64, +1.81], 269 flips.
  Clef-Flash vs Kev-9B: 0.7611 vs 0.7492, +1.19 pp [-1.10, +3.50], 291 flips (clean-only +1.82 [-0.35, +4.12]). The
  models disagree on 21-23 % of questions, so the near-equal means hide different strengths (per source, n 80-200;
  12 per none/permuted cell): Kev far better when the right option is removed (none_absent: Kev-27B 1.00 vs Clef 0.58
  on each of mmlu / sciq / emotion; Kev trains with none-of-the-above augmentation) and on buried states (0.70 vs
  0.61); Clef better on sciq / paws / emotion / legacy_holdout / tweet_offensive; Clef-Flash beats Kev-9B on MMLU-Pro
  (0.645 vs 0.590) and mmlu (0.85 vs 0.725). Subgroup readings are post hoc and low-n: context, not findings.
- 2026-10-05T15:30Z `run` E8 home-field context (Kev training distributions; paired, all variants, served p; delta =
  Clef side - Kev side): decision-v7 Clef vs Kev-27B -0.27 pp [-1.89, +1.34] / Clef-Flash vs Kev-9B +0.07 [-1.47,
  +1.59]; hard-v1 -15.33 [-18.19, -12.57] / -16.34 [-19.67, -12.92]; devtools-v1 -3.73 [-6.69, -0.73] / -11.18 [-14.12,
  -8.09]; documents-v1 -3.70 [-5.89, -1.51] / -4.35 [-6.67, -2.09]; semif -1.98 [-4.03, 0.00] / -1.19 [-5.12, +2.38].
  Kev-27B documents-v1 clean acc 0.9163 (920). As pre-registered these say little about generality: Kev trained on
  hard-v1 / devtools-v1 / documents-v1 / decision-v7 train partitions (Kev-27B v2's SFT corpus includes them).
- 2026-10-05T16:30Z `review` adversarial review of the E8 claim (subagent). No pairing, key / label alignment,
  truncation (max Clef input 5,404 tokens of 16,384; 0 rejected) or CI error (deltas reproduced exactly). Accepted:
  (a) **correction: my primary row set was wrong.** "All 1,264 rows" includes 110 `unknowable` rows whose labels carry no
  answer (kev/benchmark.py:87: "scored on confidence, never on accuracy"; kev/transfer_v9.py deletes the deciding
  evidence). The suite's accuracy set is the 1,046 knowable clean rows (report.json `clean`). Re-scored (I re-ran it,
  record-clustered, 2,000 resamples, seed 0): **Clef vs Kev-27B -0.29 pp [-2.66, +2.06] (discordant 70 / 67);
  Clef-Flash vs Kev-9B +2.01 pp [-0.39, +4.49] (69 / 90).** The 14:30Z numbers (-0.32 / +1.19 on 1,264) are superseded.
  (b) power: discordance ~18 %, SE ~1.2 pp, MDE ~3.2 pp. 27B pair: CI inside +/-3 pp (equivalence within ~3 pp
  defensible); 9B pair: not equivalent at 3 pp, point estimate favours Clef-Flash.
  (c) transfer-v9 partly favours Kev: ~366 of the 1,046 rows come from Kev's own generator families (legacy_holdout <-
  legacy_policy, composition_holdout <- compositional, unknowable_control <- night2_unknowable_control, buried <- Kev
  decision records), and the none_absent / none_present / permuted variants are Kev's augmentation format. Split:
  public sources (680) Clef - Kev-27B +0.7 pp, Clef-Flash - Kev-9B +2.5 pp; Kev-synthetic (366) -2.2 / +1.1 pp.
  "Eval-only by construction" holds by source name, not by distribution.
  (d) a null on transfer-v9 neither confirms nor refutes Cloudflare's card (a different suite: Decision Index 0.2.1).
  evals/breadth-v1 mirrors the Index, but its partitions are in jaredpalmer/kev-private-evals: **no access from our
  token** (dataset_info -> Repository Not Found). scripts/build_breadth_v1.py claims a byte-for-byte rebuild from public
  sources (some NC / share-alike / unlicensed text: must never be pushed).
- 2026-10-05T16:30Z `finding` (revised; supersedes 14:30Z) **E8: on transfer-v9's 1,046 knowable questions Clef ~= Kev-27B
  (-0.3 pp [-2.7, +2.1], equivalent within ~3 pp) and Clef-Flash leads Kev-9B by +2.0 pp [-0.4, +4.5], not
  significant; the suite leans toward Kev, so Clef is if anything understated. On Kev's own training distributions Kev
  leads by 4-16 pp. Cloudflare's card is untested here.**

**E9 breadth-v1 card comparison (pre-registered 2026-10-05T17:00Z, before the rebuild or any read).** breadth-v1
(14 of the Decision Index 0.2 datasets; dev 1,990 records / 3,075 q) is private upstream; rebuild with
scripts/build_breadth_v1.py on spark-1, **accepted only if development.jsonl sha256 = 9aad8f4a...** (manifest). Partitions
are licence-restricted (SATA-Bench NC, SGD SA, Humicroedit / cfcolor unlicensed): gitignored, kept on the Sparks, never
pushed; only aggregate reports are committed. Development partition only (test is locked).
- Positive controls: Spark reads of Kev-27B v2 and Kev-9B v2 vs committed H200 rows (runs/r23-27b-k-w85-breadth = v2
  weights; runs/fam-9bnew-breadth = r18-9b trial = v2) must pass E2's parity rule (flips <= 1 %, p99 dp <= 0.05; long
  states here may push flips up, report if 1-2 %).
- Arms: Clef, Clef-Flash (Spark, remote harness as E8), Kev-27B v2, Kev-9B v2 (Spark); context only: Kev-9B v1
  (runs/fam-9b-breadth, H200) and Jev (runs/breadth-v1-jev, committed rows).
- **Primary: the card's per-dataset direction.** For each of the 14 datasets and each card pair (Clef vs Kev 9B;
  Clef-Flash vs Kev 9B; their "Kev 9B" version is unknown, so read against v2 and, as context, v1): "reproduces" if our
  paired delta (dataset's metric: accuracy or case-exact, as breadth_report scores it; record-clustered bootstrap) has
  the card's sign with the 95 % CI excluding 0; "contradicted" if the CI excludes 0 with the opposite sign; else
  "unresolved". Card ties (|card delta| < 2 points: RouterBench) are tested for no difference only. Count each class.
  Scores are NOT compared to the card's absolute numbers (breadth-v1 restricts candidate sets; its docstring says scores
  are not comparable to the Index's).
- Secondary: breadth_report's chance-corrected index (5 areas) per system; Clef vs Kev-27B per dataset (same base).
- Caveat set in advance: breadth-v1's mapping onto Kev's question format may itself suit Kev (it was built for Kev's
  API); and per-dataset n is 150 records, so per-dataset MDE is large (~8-10 pp).
- 2026-10-05T19:00Z `run` E9: breadth-v1 rebuilt on spark-1 (scripts/build_breadth_v1.py, raw downloads in
  ~/breadth-raw): **development.jsonl sha256 9aad8f4a... = manifest (byte-identical)**; test 91a64f0a... (not read).
  Installed only in gitignored evals/breadth-v1/ on both Sparks. Reads (dev, 3,075 q): Clef-Flash 0.8221, Clef 0.8049,
  Kev-27B v2 0.7567, Kev-9B v2 0.6995 (question accuracy). Load: Clef 391 s / 51.2 GiB, Clef-Flash 138 s / 17.8 GiB.
- 2026-10-05T19:00Z `check` E9 positive controls: Spark Kev-27B v2 vs runs/r23-27b-k-w85-breadth (H200): 20 flips / 3,075
  (0.65 %), p99 dp 0.015; Spark Kev-9B v2 vs runs/fam-9bnew-breadth: 1 flip (0.03 %), p99 0.0006 -> pass (E2 rule).
- 2026-10-05T19:00Z `run` E9 card-direction test (spark/e9_card.py; runs/spark/e8/e9-card-v2.json, -v1.json): vs Kev-9B v2
  over 14 datasets: Clef 8 reproduces / 4 unresolved / 1 contradicted (SGD: card Kev +20.2, ours Clef +30.7 [+23.3,
  +38.0]) / 1 tie-but-different (RouterBench: card tie, ours Clef +21.3); Clef-Flash 6 / 4 / 2 contradicted (SGD +31.3,
  CLINC150: card Clef-Flash -12.2, ours +16.7) / 2 tie-but-different (RouterBench +38.7, BRIGHT +10.7). Every
  contradiction is in Clef's favour; Kev-9B significantly beats neither Clef model on any dataset (closest: ToolRet
  -2.7 / -4.0, n.s.). vs Kev-9B v1 (context): Clef 9 / 3 / 1 / 1; Clef-Flash 6 / 4 / 2 / 2.
- 2026-10-05T19:00Z `run` E9 breadth_report (runs/spark/e8/e9-breadth-report): Decision-Index-style index (5 areas,
  chance-corrected): Clef-Flash 66.0, Clef 62.3, Jev 53.3 (committed rows), Kev-27B v2 51.6, Kev-9B v2 43.3, Kev-9B v1
  41.7. Clef per dataset vs Kev-27B: higher on 11 of 14; Kev-27B higher on ToolRet (0.687 vs 0.633) and ContractNLI
  (0.806 vs 0.800), equal on BFCL (0.960).
- 2026-10-05T19:00Z `finding` (pending review) **E9: Cloudflare's card directions reproduce on breadth-v1 or understate
  Clef; Clef / Clef-Flash lead Kev-9B (and Kev-27B) on this OOD-for-Kev panel by 10-23 index points.** Caveat stated in
  advance stays: Clef's training data is undisclosed; CLINC150 1.000 and SGD 0.967 are consistent with (not proof of)
  those datasets' train splits in Clef's training, which breadth-v1 deliberately excludes from Kev's.
- 2026-10-05T20:00Z `review` adversarial review of the E9 claim (subagent; every number reproduced: tallies, index).
  Accepted corrections to my 19:00Z framing:
  (a) **"reproduces or understates Clef" is wrong.** The disagreements are artifacts of breadth-v1 asking different
  tasks from the Index: SGD's whole gap is its 50 NONE-answer rows (I re-checked: NONE acc Kev-9B 0.24 / Kev-27B 0.42 /
  Jev 0.52 / Clef 1.00 / Clef-Flash 0.98; intent rows 0.87 / 0.88 / 0.93 / 0.95 / 0.97), where the Index's macro-F1
  weights NONE as one class; CLINC150 here is 10 options, not 151 (both Clef 150/150 vs card Clef-Flash 66.8);
  RouterBench is beaten by a prompt-blind model-name prior (0.47 leave-one-out, reviewer-computed; only Clef-Flash 0.57
  exceeds it). And the card *overstates* Clef 3-13x on API-Bank (+35.6 / +36.8 vs ours +6.0 / +2.7) and ContractNLI
  (+23.6 / +26.5 vs +5.6 / +9.4). Magnitudes were never comparable (pre-registered); "understate" was post hoc.
  (b) "every contradiction favours Clef" is not independent evidence (Clef is higher on 11-12 of 14); opposite-sign
  non-significant deltas favour Kev (ToolRet -2.7 / -4.0, BFCL -0.7). Under Bonferroni (28 tests) 4 'reproduces' per
  model survive (Clef: CLINC150, HellaSwag, MuSR, BRIGHT; Clef-Flash: Humicroedit, HellaSwag, MuSR, SATA-Bench).
  (c) index uncertainty (reviewer's paired record bootstrap, 1,000): Clef - Kev-27B +10.5 [7.5, 13.6]; Clef-Flash -
  Kev-9B +22.4 [19.3, 25.9]; Clef-Flash - Clef +3.6 [0.7, 6.5]; **Jev - Kev-27B +1.6 [-1.1, 4.5]: not separated** (my
  ordering "Jev 53.3 > Kev-27B 51.6" withdrawn). Jev's read is older (2026-09-24, gateway alias, suite manifest hash
  48db8e97 vs 6a391fed now; rows' ids / keys / labels match).
  (d) contamination: most of these datasets have public train splits; the panel is OOD for Kev, possibly in
  distribution for Clef. Dropping RouterBench / SGD / CLINC150 shrinks the leads to ~+8 / +18; also HellaSwag / MuSR ->
  ~+5 / +14 (rough; area weights shift).
  (e) no positive control for the remote scoring path (here or in E8): running now (Kev-9B served by kev.serve fp32,
  read with --remote, vs its in-process rows).
- 2026-10-05T20:00Z `finding` (revised; supersedes 19:00Z) **E9: on breadth-v1 dev the card's direction reproduces
  significantly for Clef on 8/14 datasets and Clef-Flash 6/14 vs Kev-9B v2 (4 each under Bonferroni); the card's
  magnitudes do not (much smaller here on API-Bank / ContractNLI), and its disagreements trace to breadth-v1's task
  construction (SGD NONE option, 10-way CLINC, RouterBench name prior). Index-style: Clef +10.5 [7.5, 13.6] over
  Kev-27B v2, Clef-Flash +22.4 [19.3, 25.9] over Kev-9B v2; ~+5 / +14 without the five most suspect datasets. Jev and
  Kev-27B not separated. OOD for Kev; Clef's exposure unknown.**
- 2026-10-05T21:30Z `check` **remote scoring path positive control passes** (closes E8 / E9 review gap): Kev-9B v2 served
  by kev.serve (KEV_DTYPE=fp32, unfused, no graphs) on spark-2 and read with `kev.benchmark --remote` on breadth-v1 dev
  vs its in-process read (runs/spark/e8/kev-9b-breadth-v1): 3,075 paired, **0 flips, max dp 0.0010, p99 0.0005**, acc
  0.6995 = 0.6995, 0 rejected / truncated. The remote harness (api_request -> /v1/systemone -> RemotePredictor) carries
  the same information and maps probabilities faithfully. Clef's own encode path (clef_server.py) is not covered by this
  control; its 0 rejections and high accuracy rule out gross mapping errors.

**E10 FP8 Clef-Flash for a 16 GB Ada GPU (pre-registered 2026-10-05T23:00Z, before any quantized Clef read).** Target:
the user's 16 GB Ada card (sm_89: FP8 tensor cores, no FP4). Deliverable: a pre-quantized export (safetensors: FP8 e4m3
decoder Linear weights + fp32 per-output-channel scales; embeddings, norms, vision tower, small projections and the
joint head stay bf16) and a loader that builds Clef-Flash without materialising bf16 weights on the GPU. FP8 arithmetic
in a portable form that runs on sm_89 and sm_121 alike (scaled_mm with tensorwise scales of 1 on e4m3 operands, then
per-token x per-channel rescale in fp32), so the Spark read is the arithmetic the Ada card runs (kernel choice may
still differ: not removable here; stated). Activations: dynamic per-token scales (row-independent, no batch coupling).
- Arms: Clef-Flash bf16 (E8 / E9 rows: same image, same clef_server path) vs Clef-Flash FP8 loaded from the export.
- Reads: transfer-v9 dev, breadth-v1 dev, decision-v7 dev (all rows; the breadth / v7 / v9 sets as in E8 / E9; for
  transfer-v9 the accuracy set excludes unknowable rows, the E8 correction).
- **Primary: pooled accuracy delta FP8 - bf16 over the three suites, record-clustered bootstrap**; claim rule =
  resolution: "no loss beyond X pp", X = -(CI lower bound). Per-suite deltas and flips as context. ECE reported, not
  judged (E4 lesson: ~0.012 noise floor at this n).
- **Memory (gate for the deployment claim):** with torch.cuda.set_per_process_memory_fraction set to 15.0 GiB of the
  Spark's pool (headroom for the CUDA context on a 16 GB card), the export loads and answers (a) the longest breadth-v1
  record and (b) a synthetic 16,384-token-state request without OOM. Report weights GiB, peak GiB at each.
- Not covered and stated: kernels on the real Ada card (an Ada validation read is the user's step; a script ships),
  states beyond 16,384 tokens, image / video inputs (not read in E8 / E9 either).
- 2026-10-06T02:00Z `tool` E10 build (subagent, spark-2; spark/clef_fp8.py, clef_server.py --fp8-export, ada_validate.md,
  ada_smoke.py; reviewed): PortableFp8Linear = rowwise reference bit for bit without bias (bias: one bf16 ulp, added in
  fp32); 200 decoder projections FP8 (6.91 B params, 99.9 %), 48 small DeltaNet a/b projections bf16; embed + lm_head
  (untied) 2.03 B, vision tower 0.46 B, joint head 0.12 B stay bf16. Export 11.10 GiB of tensors; load_fp8 weights 11.32
  GiB, peak during load 11.32 GiB (bf16 model 17.8); reload = in-memory model bit for bit (1,086 tensors).
  **Memory gate (cap 15.0 GiB via set_per_process_memory_fraction): pass.** Longest breadth-v1 record (6,587 tokens)
  peak 12.24 GiB alloc / 12.29 reserved; 16,384-token request 13.56 / 13.66 GiB with
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (default allocator: 13.57 / 14.97 reserved, 30 MiB under the cap:
  the setting is required, and the Ada recipe sets it). First version OOM'd on the 16k request (fp32 temporaries);
  fixed in place with identical bits.
- 2026-10-06T02:00Z `run` E10 reads on spark-1 (same machine / image as E8-E9 bf16 rows; export copied over the cluster
  link, SHA256SUMS ok): **bf16 control re-read of transfer-v9 = E8 rows exactly (0 flips / 1,154, dp max 0)**.
  spark/e10_readout.py (runs/spark/e10/readout.json): transfer-v9 (1,154, unknowable excluded) -0.17 pp [-1.04, +0.61],
  26 flips; breadth-v1 (3,075) +0.13 [-0.20, +0.46], 32; decision-v7 (1,468) +0.14 [-0.20, +0.47], 7. **Pooled 5,697:
  0.8315 -> 0.8322, +0.07 pp [-0.18, +0.32]**, discordant +28 / -24, flips 1.14 %, dp p99 0.079, max 0.567. Median
  request latency through the harness (Spark, unfused): FP8 95 / 175 / 105 ms vs bf16 108 / 161 / 115 ms.
- 2026-10-06T02:00Z `finding` (pending review) **E10: FP8 Clef-Flash (11.3 GiB) loses no accuracy beyond 0.18 pp vs
  bf16 on 5,697 questions, and fits a 15 GiB cap at 16k-token states (with expandable_segments).** Not covered: the
  real Ada kernels (user's step: spark/ada_validate.md), images / video, states > 16,384 tokens.
- 2026-10-06T03:00Z `review` adversarial review of the E10 claim (subagent): accuracy half confirmed (independent
  bootstrap [-0.19, +0.33] by group, [-0.18, +0.32] by id / row; clean-only +0.04 pp [-0.22, +0.30]; the pooled set
  includes 312 permuted / none-pair rows, 5.5 %); quantization verifiably applied. Accepted:
  (a) **"fits a 16 GB Ada card" withdrawn as a result**: set_per_process_memory_fraction caps only PyTorch's
  allocator, on unified memory; CUDA context (~0.3-0.8 GB), Triton / fla code memory and other processes on the card
  (the user's embedding models share these GPUs) are outside it. Correct statement: allocator peak 13.56 / 13.66 GiB
  at 16,384 tokens with expandable_segments under a 15.0 GiB cap; plausible on an otherwise empty 16 GB card; not run
  on Ada. Memory JSON now committed (runs/spark/e10/c10/).
  (b) portability risk: `_scaled_mm(..., out_dtype=float32)` on sm_89 is unverified. Fixed robustly: clef_fp8
  probes the output dtype once per device (fp32, else bf16-output fallback; FP8_GEMM_OUT forces), and clef_server
  reports it in /v1/models. E10b (running) re-reads default mode (must reproduce E10 rows) and the forced bf16-out
  fallback on all three suites, so either Ada outcome has a measured accuracy.
  (c) no latency gain: medians mixed, p95 worse on breadth-v1 (604 vs 453 ms) and decision-v7 (743 vs 547).
- 2026-10-06T03:00Z `finding` (revised; supersedes 02:00Z) **E10: FP8 Clef-Flash (11.3 GiB) shows no accuracy loss beyond
  0.18 pp vs bf16 on 5,697 questions (+0.07 pp [-0.18, +0.32]); allocator peak 13.7 GiB at 16k tokens; fit on a real
  16 GB Ada card plausible if the card is otherwise empty, untested; no speed gain.**
- 2026-10-06T03:30Z `reference` user asked about PrismaQuant (github.com/RobTand/prismaquant) and their overbook work.
  overbook (explorer): a vLLM plugin paging MoE routed experts between VRAM and host (expertpager); no quantizer, no
  Linear kernels, nothing for a dense model; its dequant kernels live in gridbook (FP8 / FP4 codebook, sm_120-first).
  PrismaQuant (README): AURA per-Linear mixed-precision allocation (KL-Fisher sensitivity + production-rendered error,
  knapsack, held-out KL gate); exports compressed-tensors (NVFP4 / FP8 / BF16), GGUF, Tessera; served by vLLM or
  llama.cpp. For Clef-Flash on Ada: NVFP4 not native on sm_89, and neither vLLM nor llama.cpp serves Clef's joint head,
  so usable = its allocation idea + an Ada-native 4-bit weight kernel (W4A16), scored on Clef's option probabilities.
- 2026-10-06T03:30Z `check` FP8 Clef-Flash memory composition (from the export report): FP8 decoder 6.91 B params (6.4
  GiB), embed_tokens + lm_head bf16 2.03 B (3.8 GiB), vision tower 0.46 B (0.85 GiB), joint head + small 0.13 B
  (0.25 GiB). Clef's ClefModel.forward passes `get_output_embeddings().weight` to the head, which only gathers rows
  (`output_embedding_weight[token_ids]`, joint_schema_model.py:382); embed_tokens is a gather too. Neither needs a GEMM.

**E11a host-resident embeddings + text-only load (planned 2026-10-06T03:30Z, before implementation).** Keep embed_tokens
and lm_head in pinned CPU memory, gather the needed rows on the CPU and copy them to the GPU; do not load the vision
tower (text-only server; requests with images / videos get a 422). Expected GPU weights ~6.9 GiB. Arithmetic unchanged,
so **pre-registered expectation: probabilities bit-identical to the E10 FP8 export on transfer-v9 dev (0 flips, max
dp 0)**; any difference is a bug, not a result. Also: memtest at the same 15.0 GiB cap (peak at 16k tokens) and the
added latency of CPU gathers (median / p95 vs E10 FP8 on transfer-v9).
- 2026-10-06T06:00Z `run` E10b / E11a (spark-1; spark-1's checkout first failed to pull twice because untracked copies
  of committed scripts blocked the merge: hazard, now cleared). Full transfer-v9 identity reads vs E10 FP8 rows
  (1,264 rows): (i) default mode after the GEMM-output probe change: **0 flips, max dp 0**; (ii) **E11a host-resident
  embeddings + text-only (6.68 GiB GPU weights): 0 flips, max dp 0**, pre-registered expectation met.
  (iii) forced bf16-output fallback (FP8_GEMM_OUT=bfloat16, the path an sm_89 without fp32 GEMM output would take):
  vs bf16 Clef-Flash pooled 5,697 q **+0.02 pp [-0.23, +0.26]** (transfer-v9 -0.09 [-0.78, +0.59], breadth-v1 +0.10
  [-0.26, +0.49], decision-v7 -0.07 [-0.36, +0.21]), 64 flips; vs fp32-out FP8 -0.05 [-0.28, +0.18], 53 flips
  (runs/spark/e10/readout-bf16out-fallback.json).
- 2026-10-06T06:00Z `finding` **E10 / E11a: text-only FP8 Clef-Flash runs in 6.7 GiB of GPU weights (8.9 GiB allocator
  peak at 16k tokens), answers bit-identical to the 11.3 GiB FP8 build, no accuracy loss beyond 0.18 pp vs bf16 (or
  0.23 pp on the bf16-output fallback path).** Untested: the Ada card itself, image input (refused in text-only mode).
- 2026-10-06T07:00Z `reference` user's 16 GB Ada card runs Qwen3-Embedding-8B FP8 at 32k context: 15.9 / 16.4 GB used.
  Research subagent (web, sources in session) on smaller embedder options. Checked by me: KV arithmetic re-derived
  (2 x 36 layers x 8 KV heads x 128 x 2 B = 144 KiB / token; 32k = 4.5 GiB bf16, 2.25 GiB fp8; 4B has the same KV shape);
  param count 7.57 B (6.95 B linear + 0.62 B embedding) -> FP8 weights ~7.6 GiB, so most of the 15.9 GB is vLLM's
  up-front reservation (gpu_memory_utilization), not need. Qwen card (fetched): Eng v2 retrieval 8B 69.44 / 4B 68.46 /
  0.6B 61.83 (matches the agent); **MMTEB retrieval mismatch**: agent 70.88 / 69.60 / 64.64 vs card read 86.40 / 85.05 /
  80.83 (unresolved; agent's cross-model multilingual retrieval figures treated as unverified). Agent's recommendation:
  Qwen3-Embedding-4B FP8 + FP8 KV cache, ~7.4 GiB at 32k (estimate), -0.98 Eng v2 retrieval. Only published 4-bit
  evidence on the 8B (arXiv 2609.24322, round-to-nearest int4): -3.1 % rel nDCG@10, 15.6 % of gold top-1 lost: unverified.
- 2026-10-06T07:00Z `check` co-residency arithmetic (estimates, not measured; ~15.3 GiB usable of 16.4 GB): 4B FP8 at
  32k ~7.4 GiB + Clef-Flash text-only peak 8.9 GiB at 16k + two CUDA contexts ~1 GiB = ~17.3 GiB -> **does not fit**.
  With the embedder at 8k chunks (FP8 KV 0.56 GiB, ~5.7 GiB) and Clef-Flash capped near 8k tokens (~7.8 GiB, between
  the measured 7.6 @ 6.6k and 8.9 @ 16k) -> ~14.5 GiB: fits, tight. Needs a measured co-residency test before any claim.

**E12 co-residency on a simulated 16 GB card (pre-registered 2026-10-06T07:30Z, before any run).** User accepts chunking
and re-embedding. Plan under test: Qwen3-Embedding-4B served by vLLM (FP8 weights via online quantization, FP8 KV cache,
max_model_len 8192, KV pinned with --kv-cache-memory-bytes, max_num_seqs small) + text-only FP8 Clef-Flash
(clef_server --host-embeddings --text-only, --max_length 8192, expandable_segments) as two processes on spark-1.
- Measure: each process's GPU memory (nvidia-smi per-process used_memory, which includes its CUDA context) at idle,
  then at peak under concurrent load: 8,192-token embedding requests and 8,192-token Clef-Flash requests interleaved
  for 3 minutes. Record the peak sum.
- **Pass = peak sum <= 15.0 GiB** (a 16 GB Ada card shows ~15.6 GiB usable after the display and driver; 0.6 GiB margin).
  > 15.0 -> report by how much and which knob (KV bytes, max_num_seqs, Clef max_length) closes it.
- Context, not gate: FP8 vs bf16 Qwen3-Embedding-4B embedding agreement (cosine on 200 chunks of evals/documents-v1
  states, 8k max) and throughput (chunks/s) while Clef-Flash is loaded.
- Not covered and stated: Ada kernels and CUDA context size on the real card (GB10 contexts may differ in size);
  retrieval quality on the user's own webpages (needs their queries; the 8B -> 4B gap is the card's -1.0 Eng v2).
- 2026-10-06T08:30Z `run` E12 v1 (spark-1; runs/spark/e12/corun-v1-cached.json): vLLM log: 4B weights 4.41 GiB, KV
  pinned 1.25 GiB (18,192 tokens, 2.22x concurrency at 8,192). Idle sum 13,191 MiB (embed 6,107 / Clef 7,084); peak
  sum **14,727 MiB = 14.38 GiB** (embed 6,735 / Clef 7,992), 0 errors (embed 2,038 ok, Clef 42 ok; Clef 4.38 s median).
- 2026-10-06T08:30Z `hazard` **E12 v1 is not a valid pass.** The harness sent the identical document every request:
  embed latency 0.089 s for an "8,192-token" request = ~92k tokens/s, i.e. ~740 TFLOPS for a 4B model, impossible on
  GB10 -> almost certainly vLLM prefix caching served repeated prompts without a full prefill, so the embedder's
  activation peak under real (unique-chunk) load was never exercised; the token count actually processed was not
  recorded either. Clef side valid (no prefix cache; 4.4 s per request is real compute). Fix: unique text per request
  (counter + varied content), record usage.prompt_tokens, re-run (E12 v2). The pre-registered gate is unchanged.
- 2026-10-06T09:30Z `run` E12 v2 (unique text per request; runs/spark/e12/corun-v2-6k.json): embed now real compute
  (1.444 s median, 4,257 prompt tokens/s: confirms v1 was served from the prefix cache). Peak sum **14,725 MiB =
  14.38 GiB** (embed 6,733 / Clef 7,992), 0 errors (embed 126, Clef 29 at 6.32 s). **But recorded embed
  prompt_tokens = 6,028-6,036, not 8,192**: my generator (assumed 4.3 chars/token) undershot, so truncation never
  engaged and Clef got the same ~6k text. Valid as a pass **at 6k-token requests** only; the pre-registered 8k load
  is still untested (more activation memory on both sides). Embed peak barely moved from v1 (6,733 vs 6,735 MiB)
  despite real prefills: vLLM's workspace is sized at startup. v3: overshoot the text, both servers truncate to
  exactly 8,192, assert embed prompt_tokens == 8,192 and record Clef usage.input_tokens.
