# Per-frame VLM benchmark: Qwen3.8 27B

## Result

This remains the provisional default. In this run it had complete format and ID coverage and the highest agreement with the cached Qwen3.8 reference. That reference is self-favoring, so the agreement does not independently show that Qwen3.8 annotates better.

| Measure | Result |
|---|---:|
| Frames completed | 40/40 |
| Strict relation lists / strict caption lists | 40/40 / 40/40 |
| Caption object IDs accounted for | 763/763 |
| Valid relation tuples (both IDs resolve) | 339/339 |
| Relation agreement F1 vs cached Qwen3.8 | 0.497647 |
| Caption word F1 vs cached Qwen3.8 | 0.646177 |
| Mean relation / caption request latency | 6.5524 s / 6.7565 s |
| Total paired inference time | 532.4 s |
| Artifact create-to-mtime interval (wall-time proxy) | 537.74 s |
| GPU memory | Model-control sample: 19,194,928,167 bytes model VRAM; 25,635 MiB total used |

## Configuration and inputs

- **Model:** `qwen3.8:27b-mtp4-ud_q4_K_XL-128k`.
- **Backend:** managed Ollama OpenAI-compatible endpoint. The artifact records `http://127.0.0.1:20261/v1`.
- **Quantization / serving variant:** UD Q4_K_XL, MTP4.
- **Context:** 131,072 tokens.
- **Sampling and generation:** the harness sent a 4,096-token output cap. Server temperature/top-p settings were not captured in the run artifact.
- **Data:** Replica `room0`; 40 annotated frames at stride 50 from the modernized YOLOE detection cache `s_modernized_yoloe26l_stride50`. The artifact includes per-frame labels, cached candidate/reference annotations, timings, token usage, and raw responses.
- **Prompts:** shared `system_prompt_only_top` for relations and `system_prompt_captions` for captions; image-only user turn followed by a separate text-only user turn. The exact serialized request body is not retained in this run artifact. User prompt templates are recorded in the matched Transformers artifacts.
- **Failures:** none among 40 frame pairs. Caption consolidation was not run in this artifact (`0` completed).
- **Resource provenance:** the VRAM figures came from a separate model-control status sample during this baseline run; they are not embedded in the JSON. The create-to-mtime interval is a filesystem proxy, not a recorded process wall-time field.

## Reading the score

Agreement is computed against cached Qwen3.8 annotations. Relation F1 therefore measures annotation agreement, not physical-relation accuracy. Caption word F1 is lexical overlap, not semantic quality or factuality. The run supports retaining Qwen3.8 as a provisional baseline because its requests completed in the expected format and covered all listed IDs. Its agreement lead is against its own cached annotations; broader scene and independent-reference evaluation remains necessary.

## Evidence

- JSON and embedded raw responses: [`vlm-qwen38-baseline-40.json`](data/vlm-qwen38-baseline-40.json)
- Captured creation time (`created_utc`): `2026-11-14T00:48:51.983432+00:00`.
- Artifact file modification time: `2026-11-14T00:57:49.721406+00:00`.
