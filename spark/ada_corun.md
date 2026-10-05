# Qwen3-Embedding-4B + Clef-Flash on one 16 GB Ada card

> **Deployed on haumea's RTX 4060 Ti on 2026-10-05; the run book there (`/home/scott/ai-stack/README.md`, copy in
> `spark/haumea-ai-stack-README.md`) supersedes the commands below.** On that card FlashInfer could not build its FP8-KV
> kernel, so the embedder runs a **bf16 KV cache with `--max-num-seqs 1`** (same 1.25 GiB pin), and both servers must be
> launched under `conda activate`, with `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `setsid`. Measured there: 14,856 of 16,380
> MiB peak, 1.5 GiB headroom.

Measured on a DGX Spark (E12 in `spark/LAB_NOTEBOOK.md`): together the two servers peaked at **14.64 GiB** of GPU memory,
including startup, two concurrent 8k embedding requests and 10-question Clef requests. That leaves about 0.4 GiB below a
15.0 GiB budget. It has **not** been run on an Ada card, whose CUDA contexts may be larger, so check `nvidia-smi` as
described below before relying on it.

## 1. Start the embedder first

```bash
vllm serve Qwen/Qwen3-Embedding-4B \
  --runner pooling --quantization fp8 --kv-cache-dtype fp8 \
  --max-model-len 8192 --max-num-seqs 2 \
  --kv-cache-memory-bytes 1342177280 \
  --port 8051
```

- `--kv-cache-memory-bytes` pins the KV cache at 1.25 GiB (two 8k sequences) instead of vLLM's default, which takes
  about 92 % of the card. That default is why the 8B used 15.9 GB.
- FP8 weights + FP8 KV vs bf16, measured on 300 documents: median cosine 0.994, top-1 retrieval agreement 99 %, top-10
  overlap 92 %.
- Older vLLM versions use `--task embed` instead of `--runner pooling`. These flags were tested on vLLM 0.27.2rc1.

## 2. Start Clef-Flash second

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python spark/clef_server.py --fp8-export <path to clef-flash-fp8 export> \
  --host-embeddings --text-only --max_length 8192 --port 8031
```

Setup and the expected smoke answers are in `spark/ada_validate.md`.

## 3. Client rules

- **Chunk pages client-side to at most ~8,000 tokens**, counted with the Qwen3 tokenizer.
- **Do not send `truncate_prompt_tokens`.** On vLLM 0.27.2rc1, an over-length request that carried it got no response
  (no reply in 90 s, twice), while the same request without it got an immediate HTTP 400.
- Queries use Qwen's instruction prefix (`Instruct: <task>\nQuery: <text>`); documents are sent bare.
- **Re-embed the whole corpus.** 4B vectors are not compatible with the 8B's.

## 4. Check the fit on the card

After both servers are up, then again under real load:

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

Expected from the Spark: about 13.2 GiB at idle, rising to about 14.6 GiB once both have served long requests. If the
card runs out of memory, in this order:

1. Clef `--max_length 6144` (it peaked at 7.8 GiB with ~6k-token inputs, against 8.06 GiB at 8,192).
2. Embedder `--max-num-seqs 1 --kv-cache-memory-bytes 671088640` (one sequence, about 0.6 GiB less).
3. Chunk to 4k tokens.

## What is not known yet

- Behaviour on the Ada card itself: kernels, CUDA context size, and whether Clef's FP8 path uses fp32 or bf16 GEMM
  output (`/v1/models` reports which).
- Retrieval quality of the 4B on your own webpages. Qwen's card puts it about 1 point below the 8B on English
  retrieval (68.46 vs 69.44). A few dozen of your own queries with known-good pages would settle it.
