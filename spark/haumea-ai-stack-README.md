# RTX 4060 Ti (GPU 0) stack: Qwen3-Embedding-4B + Clef-Flash

Set up 2026-10-05. Replaces the Qwen3-Embedding-8B that used to fill this card. The RTX 5090 (GPU 1, Qwen3.8-27B NVFP4
on port 8000) is separate and not managed here.

| service | port | served name | what |
|---|---|---|---|
| embeddings (vLLM, conda env `vllm`) | 8001 | `qwen3-embedding-4b` | Qwen3-Embedding-4B, FP8 weights, bf16 KV cache, 8,192-token max, 1 sequence |
| decisions (`clef_server.py`, venv) | 8031 | `Cloudflare/clef-flash` | Clef-Flash 9B, FP8 export, text-only, `/v1/systemone` (Jev / System One API) |

## Run it

```bash
bash /home/scott/ai-stack/start-4060-stack.sh    # embedder first, then Clef-Flash; refuses if GPU 0 is busy
bash /home/scott/ai-stack/status-4060-stack.sh   # health of both + GPU 0 memory
bash /home/scott/ai-stack/stop-4060-stack.sh     # Clef-Flash, then embedder
```

Logs: `embed-4b.log` and `clef-flash.log` in this directory. PIDs: `embed.pid`, `clef.pid`.

**Nothing starts at boot.** After a reboot, run the start script. The old 8B was also started by hand.

To go back to the old 8B embedder: `bash stop-4060-stack.sh && bash ROLLBACK-qwen3-embedding-8b.sh` (served name
`qwen3-embedding`, port 8001, log `/home/scott/embed-4060.log`). Tested 2026-10-05.

## Client rules

- **The model name changed to `qwen3-embedding-4b` on purpose.** 4B vectors (2,560 dims) are not compatible with the
  8B's (4,096). Re-embed the whole corpus, and never mix the two in one index.
- **Chunk documents to at most ~8,000 tokens client-side** (Qwen3 tokenizer). Longer inputs get an HTTP 400.
- **Do not send `truncate_prompt_tokens`.** On vLLM 0.27.2 (tested on a DGX Spark), an over-length request carrying it
  never returned. It was not tested on this box (0.27.1), because a hung request could wedge the single sequence slot.
- Queries use `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: <text>`;
  documents are sent bare.
- One embedding is processed at a time (`--max-num-seqs 1`); others queue. An 8k chunk takes ~1.45 s alone and ~3.1 s
  while Clef-Flash is also busy.
- Clef-Flash is text-only: requests with images or videos get HTTP 422. States go up to 8,192 tokens including the
  questions. Short requests take ~0.1-0.6 s and 8k requests ~8 s. The first request after a start takes ~15-20 s
  (kernel compile).

## Measured on this card (2026-10-05, 180 s of interleaved 8k load)

- Idle: ~13.7 GB of 16.4. Peak: **14,856 MiB of 16,380 (1.5 GiB headroom)**. Embedder 6.6 GiB, Clef-Flash 8.2 GiB.
- 0 errors over 59 embeddings and 22 Clef-Flash requests.
- Clef-Flash answers match the DGX Spark reference to within a few hundredths. Its FP8 path uses fp32 GEMM output.

## Why the launch scripts look the way they do (each item broke something on 2026-10-05)

1. **`source conda.sh && conda activate vllm`.** The env activation sets conda's gcc toolchain and the nvcc header
   paths. Without it, FlashInfer's runtime kernel builds pick up Ubuntu 26.04's system glibc headers, nvcc fails
   (`rsqrt` exception-specification error), and vLLM dies on its first request.
2. **`CUDA_DEVICE_ORDER=PCI_BUS_ID`.** CUDA's default device order is fastest-first, which makes the 5090 device 0.
   Without it, `CUDA_VISIBLE_DEVICES=0` selects the 5090, not this card.
3. **`setsid nohup ... < /dev/null`.** vLLM shuts down when the ssh session that started it closes, even under nohup.
4. **bf16 KV cache, not FP8.** FlashInfer could not build its FP8-KV prefill kernel for this GPU. A bf16 KV cache pinned
   at 1.25 GiB holds one 8,192-token sequence, and its kernel is already compiled.
5. **`--kv-cache-memory-bytes` instead of `--gpu-memory-utilization`.** vLLM otherwise reserves ~93 % of the card up
   front, which is what used to fill it.

## Files

- Clef-Flash: `/home/scott/clef-flash/`, containing:
  - `clef_server.py` and `clef_fp8.py`; the Clef source code was reviewed and is sha256-gated at load;
  - a venv built with `--system-site-packages` on top of the conda env, so the conda env itself is unchanged;
  - the FP8 export in `export/`, with `SHA256SUMS`.
- Model: `/home/scott/models/Qwen3-Embedding-4B`.
- Source and the full lab record: github.com/smb209/kev, branch `spark-investigation` (`spark/LAB_NOTEBOOK.md`,
  `spark/ada_corun.md`, `spark/ada_validate.md`).
