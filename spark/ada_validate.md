# Clef-Flash FP8 on a 16 GB Ada GPU: validation recipe

E10 (spark/LAB_NOTEBOOK.md). This serves the FP8 export of Cloudflare/clef-flash@17f0b0ad (built on spark-2 by
`spark/clef_fp8.py quantize`) on an sm_89 card through `spark/clef_server.py --fp8-export`. You then compare three answers
with the ones the Spark printed below. The arithmetic matches the Spark's: e4m3 operands, `torch._scaled_mm` with tensorwise
unit scales, an fp32 output, then a per-token x per-channel rescale. cuBLASLt, Triton (fla) and SDPA may still choose
different kernels on Ada, so expect small differences, not bit equality.

## 1. Environment (Linux x86_64, NVIDIA driver with CUDA 13.0 support, i.e. 580 or newer)

These are the versions the Spark ran (image `kev-spark-quant`):

```bash
python3.12 -m venv ~/clef-fp8 && source ~/clef-fp8/bin/activate      # Python 3.12 or 3.13
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
pip install transformers==5.17.0 safetensors==0.8.0 huggingface_hub==1.27.0 tokenizers==0.23.2 \
            fastapi==0.136.3 uvicorn==0.52.3 pillow einops
pip install flash-linear-attention==0.5.2 fla-core==0.5.2             # the Gated DeltaNet kernels (Triton); see note
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
#   expect: 2.13.0+cu130 <your card> (8, 9)
```

Notes:
- If the driver is older than 580, a cu128 build of torch 2.13 should work too (`--index-url .../whl/cu128`). That
  combination was not tested.
- flash-linear-attention: transformers runs Qwen3.5's delta rule through fla when it is installed and otherwise falls
  back to a PyTorch implementation. The Spark had fla 0.5.2, and the memory numbers below assume it. `causal_conv1d` was
  **not** installed on the Spark (transformers prints a fallback warning), so leave it out to match.
- No `kev` package is needed. The server needs only `spark/clef_server.py`, `spark/clef_fp8.py` and the export.

## 2. Copy the export and the scripts

The export is 12 GB (3 safetensors shards plus config, tokenizer, head and the release's `joint_schema_model.py`, which is
imported only if its sha256 is the reviewed one).

```bash
# on the Ada box (adjust host aliases; through your Mac if the boxes cannot see each other)
mkdir -p ~/clef && cd ~/clef
rsync -av --progress spark-2:kev/runs/spark/c10-clef-flash-fp8/ ./c10-clef-flash-fp8/
mkdir -p spark && scp spark-2:kev/spark/{clef_server.py,clef_fp8.py,ada_smoke.py,req1.json} spark/
cd c10-clef-flash-fp8 && sha256sum -c SHA256SUMS && cd ..
```

Expected `SHA256SUMS`:

```
b1161df6d2acb7b7d0e9ab1d78c0b7e12baf072b5d217b1ccbd9e1d0f33bf420  clef-fp8-00001-of-00003.safetensors
baa78663b052f8138abc9646c492a8eefbf44266c458a65e531b73555c9688a2  clef-fp8-00002-of-00003.safetensors
ec539d5a85213045a73785bc3a26da2fa34fae3ee2a76b548fc86bfb37a9bf5f  clef-fp8-00003-of-00003.safetensors
a8cd0af8cf26d7b9242c2f5ed03d08fb5af79841007cabb313624fa1188600c7  clef_fp8.safetensors.index.json
ccfde4bce8bba1f2f7fe66f2d963a61484ee026f5ffc117fb408f5106476191b  clef_fp8_config.json
19cdcec8c81dc9212be320fff47462ab342fbc1278be4368fb3da71241cf5ba0  joint_head.safetensors
0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3  joint_schema_model.py
```

## 3. Start the server

```bash
cd ~/clef
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python spark/clef_server.py --fp8-export c10-clef-flash-fp8 --host 127.0.0.1 --port 8031
#   expect: [clef_fp8] loaded c10-clef-flash-fp8 in ~3-30 s: {'allocated_gib': 11.32, 'peak_gib': 11.32}
#           loaded Cloudflare/clef-flash@17f0b0ad (fp8-e4m3 decoder projections + bfloat16) ...
```

**Recommended on a 16 GB card, text-only requests (E11a):** add `--host-embeddings --text-only`.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python spark/clef_server.py --fp8-export c10-clef-flash-fp8 --host-embeddings --text-only --host 127.0.0.1 --port 8031
#   expect: [clef_fp8] loaded c10-clef-flash-fp8 in ~3 s: {'allocated_gib': 6.68, 'peak_gib': 6.68}
```

`--host-embeddings` keeps the input embedding and the output embedding (bf16, 3.8 GiB) in pinned CPU memory and copies only
the gathered rows to the GPU; `--text-only` does not load the vision tower (0.85 GiB) and answers HTTP 422 to a request with
`images` or `videos`. The arithmetic is unchanged: on the Spark the probabilities were bit-identical to the plain export's on
50 transfer-v9 development records. GPU allocator memory: weights 6.68 GiB (plain 11.32), peak 7.60 GiB at the longest
breadth-v1 record (6,587 tokens) and 8.92 GiB at a 16,384-token request (plain 12.24 and 13.56), so the allocator needs well
under half of a 16 GB card. The host needs about 4 GiB of pinned memory. What the CPU gathers cost in latency is in
`runs/spark/c11a/` (LAB_NOTEBOOK E11a). Use the plain command above only if you need images or videos.

`expandable_segments` matters at long inputs. Under a 15.0 GiB cap on the Spark, a 16,384-token request reserved 13.66 GiB
with it and 14.97 GiB without it (allocator fragmentation; allocated memory was 13.56 GiB either way). Run with no display
attached to the card if you can: a desktop session can take several hundred MiB of the 16 GB.

## 4. Smoke requests

```bash
curl -s localhost:8031/v1/systemone -H 'content-type: application/json' -d @spark/req1.json
python spark/ada_smoke.py --url http://127.0.0.1:8031          # add --json for the full responses
nvidia-smi --query-gpu=memory.used,memory.total --format=csv    # in another shell, while the server is up
```

What the Spark printed (FP8 export, spark-2, GB10, 2026-10-05). The bf16 release's answers are in brackets for scale:

```
req1  (319 tokens)
  billing    noul 0.9179                         [bf16 0.9191]
  team       choice billing (0.9506)             [bf16 billing (0.9525)]
  urgency    score 1.3298 (confidence 0.6112)    [bf16 1.3039 (0.6348)]
req2  (371 tokens)
  damaged    noul 0.9669                         [bf16 0.9681]
  wants      choice replacement (0.9637)         [bf16 replacement (0.9639)]
  sentiment  score 2.0608 (confidence 0.4607)    [bf16 2.0678 (0.4634)]
req3  (262 tokens)
  passes     noul 0.0326                         [bf16 0.0357]
  bug        choice wrong_operator (0.9718)      [bf16 wrong_operator (0.9698)]
```

req1's full FP8 probabilities: team {accounts 0.0325, billing 0.9506, shipping 0.0169}, urgency {0: 0.0295, 1: 0.6112,
2: 0.3593}.

How to read it: the choices should be identical. Kernel differences between GB10 and Ada should move probabilities by much
less than FP8 itself did (FP8 vs bf16 above: up to 0.024 in a probability), so as a rule of thumb anything within ~0.01 of the Spark's FP8
values is a match. A choice that flips, or a gap above ~0.02, is worth sending back. That threshold is a guideline, not a
measurement: no Ada read exists yet.

Latency on the Spark for reference (req1, 10 warm repeats, server `latency_ms` median): FP8 122 ms, bf16 124 ms. At 319
tokens the unfused eager path is launch bound, so FP8 is not faster there. An Ada card will give different numbers.

## 5. Optional: score a suite through the identical harness

From a kev checkout on any machine that can reach the server:

```bash
uv run python -m kev.benchmark --remote http://<ada-host>:8031 --remote-model Cloudflare/clef-flash \
  --suite evals/v9/transfer-v9 --out runs/ada-clef-flash-fp8-transfer-v9
```

Do not copy licence-restricted suites (breadth-v1) to the Ada box. Scoring them remotely sends request bodies (state and
questions) to the server, which is the same exposure as any remote read. Decide that before you run one.


## Which FP8 GEMM path your card took

`GET /v1/models` reports `fp8 GEMM output float32` or `fp8 GEMM output bfloat16`. Both were read on the Spark against bf16
Clef-Flash over 5,697 questions: float32 +0.07 pp [-0.18, +0.32], bfloat16 fallback +0.02 pp [-0.23, +0.26]. The
expected smoke answers above come from the float32 path; on the bfloat16 path expect differences of up to a few
hundredths in probability (the two paths flip 53 of 5,697 answers between them), not identical numbers.
