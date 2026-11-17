# LFM3-VL-2B official Transformers BF16: 40-frame per-frame benchmark

## Result

This run completed on an RTX 5090 with the official Transformers BF16 checkpoint on the frozen 40-frame Replica `room0` set. Every request finished without error, but none produced outputs passing strict JSON/schema validation. The model returned verbose prose rather than structured arrays for both relations and captions. The run measured output format, coverage, runtime, and resource scope; its strict-format result is a clear failure for this harness.

| Measure | Result |
|---|---:|
| Complete frame pairs | 40/40 |
| Completed relation requests | 40/40 |
| Completed caption requests | 40/40 |
| Request errors | 0 |
| Missing image / frame records | 0 |
| Strict relation JSON / schema valid | 0/40 |
| Strict caption JSON / schema valid | 0/40 |
| Usable caption observations | 0/763 |
| Relation output tokens (total) | 98,811 |
| Caption output tokens (total) | 93,477 |
| Relation agreement F1 vs cached Qwen3.8 | null (no parse passed schema) |
| Caption word F1 vs cached Qwen3.8 | null (no parse passed schema) |
| First relation generation time | 28.6426 s |
| First caption generation time | 40.2654 s |
| Mean relation / caption generation latency | 25.8855 / 24.4442 s |
| Relation p50 / p95 generation latency | 25.6211 / 38.9913 s |
| Caption p50 / p95 generation latency | 24.8293 / 40.1013 s |
| Summed request generation time | 2,013.18 s |
| Model load time | 1.20 s |
| Total wall time | 2,018.37 s (33m 38.37s) |
| Effective frames/s | 0.01982 |
| Peak CUDA allocated / reserved | 3,836.3 / 3,890.0 MiB |
| Peak whole-device VRAM (nvidia-smi) | 6,105 MiB (1,982 samples, 1 s interval) |

Requests and captions each received one request per frame. The model produced prose, not structured JSON. One relation request and one caption request reached the 4,096-new-token cap. Because no output passed strict schema validation, Qwen3.8 agreement is null, not zero accuracy. Prompt text and images were held fixed across runners; LFM3 merged the shared system and task text into a single user turn with the image content, while other Transformers routes used distinct system and user turns.

## Configuration and inputs

- **Model:** `LiquidAI/LFM3-VL-2B`, revision `23387beda43c60310348e83e25f78f33c3140d3a`.
- **Precision:** BF16, official Hugging Face checkpoint.
- **Backend:** Hugging Face Transformers 5.22.1, `AutoModelForImageTextToText`, `AutoProcessor`, `trust_remote_code=True`, CUDA 0. PyTorch 2.14.1+cu130, Python 3.13.13.
- **Decoding:** greedy, `do_sample=False`, `max_new_tokens=4096`. Configured timeout of 300s is post-hoc; it did not interrupt generation.
- **Data:** Replica `room0`, 40 sampled frames at stride 50, cached modernized YOLOE detections `s_modernized_yoloe26l_stride50`, 763 detection observations. Same cached Qwen3.8 reference outputs.
- **Prompt serialization:** LFM3 serializes each request as a single user turn containing the image and the combined original shared system instruction plus original task text. The text prompts and images match the legacy profile; role serialization differs from runners using a separate system role and separate text/image user turns. Treat timing and output differences as model-plus-serialization observations, not a pure weights-only comparison.
- **Context size:** not recorded.

## Attempts and failures

The first run encountered a legacy `ast.literal_eval` response containing `...`, producing `Ellipsis`; JSON serialization failed. The staged runner was corrected to reject non-JSON legacy parse values as invalid while retaining raw responses and validation metadata. The retry completed all 40 frame pairs without request errors but produced zero outputs passing strict schema validation. Fresh independent local Qwen adjudication passed for the final runner SHA `82562bc269022e9019c5a2a37256dc30056cfa8f1a7def228fa7f99b88c2034c`.

## Comparison and interpretation

| Candidate and route | Completed | Strict relations | Strict captions | Caption IDs |
|---|---:|---:|---:|---:|
| Qwen3.8 27B, Ollama UD Q4_K_XL MTP4 | 40/40 | 40/40 | 40/40 | 763/763 |
| LFM3-VL-2B NVFP4 (GGUF llama.cpp) | 40/40 | 36/40 | 28/40 | 526/763 |
| LLaVA OneVision Qwen2 7B (structured) | 40/40 | 14/40 | 40/40 | 714/763 usable |
| Qwen2.5-VL 7B Instruct | 40/40 | 38/40 | 40/40 | 762/763 |
| LFM3-VL-2B BF16 (Transformers) | 40/40 | 0/40 | 0/40 | 0/763 |

LFM3-VL-2B BF16 completed all 80 requests without error but achieved 0/40 strict format validity for both relations and captions. This is a strict-format failure for this harness, not zero semantic accuracy. The 825-example general-benchmark run of the same checkpoint scored 63.325% on the full-suite macro and is reported separately; it is not a substitute for the 40-frame ConceptGraphs pipeline comparison. The full-weight BF16 checkpoint has a distinct resource scope from the NVFP4 GGUF variant and must be treated as a separate route.

## Artifacts

- Successful run JSON: [`lfm3_vl_2b_transformers_bf16_40_retry1_20261117.json`](data/lfm3_vl_2b_transformers_bf16_40_retry1_20261117.json), SHA-256 `e0f2ea60b9d081ce4e48fd46f1a4dc769ff5f7214b4205cf9af686c244efac90`, 2,068,501 bytes
- Successful run log: [`lfm3_vl_2b_transformers_bf16_40_retry1_20261117.run.log`](data/lfm3_vl_2b_transformers_bf16_40_retry1_20261117.run.log), SHA-256 `11e9fda445ec81ee7cc4827e8b88e0798324bd8eae24cdd75b67d7174b695e15`
- Partial failure JSON: [`lfm3_vl_2b_transformers_bf16_40_partial_failure_20261117.json`](data/lfm3_vl_2b_transformers_bf16_40_partial_failure_20261117.json), SHA-256 `9cf10cadc3864922ce9bd77243d6e4068e08ece387b3371241e65c54d745d664`
- Partial failure log: [`lfm3_vl_2b_transformers_bf16_40_partial_failure_20261117.run.log`](data/lfm3_vl_2b_transformers_bf16_40_partial_failure_20261117.run.log), SHA-256 `93dfe65a1bd6f39e762ab7099286fec2d2b8c363509c627742ef0d394e58cced`
- Runner SHA (independent review passed): `82562bc269022e9019c5a2a37256dc30056cfa8f1a7def228fa7f99b88c2034c`

## Limitations

- The model serialized prompts as a single user turn with image, differing from runners using a separate system role. Timing and output differences reflect model-plus-serialization effects, not weights-only comparisons.
- Agreement with cached Qwen3.8 is null because no output passed strict schema; this is not semantic accuracy data.
- Context-size limits, sampling defaults, and memory scope (process-allocated vs whole-device sampled) are not fully captured.
- The 4,096-token cap was reached on one relation and one caption request; these may not represent typical response length.
- Persistent results, logs, and the failed partial reside under `models/experiments/vlm/data/` in the project HDD; nothing was written to `/tmp`.
