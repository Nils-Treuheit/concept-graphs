# Per-frame VLM benchmark: LiquidAI LFM3-VL-2B NVFP4

## Result

All 40 frame pairs completed. The model was less consistent with the cached Qwen3.8 annotations and produced fewer outputs that passed strict structural checks.

| Measure | Result |
|---|---:|
| Frames completed | 40/40 |
| Strict relation lists / strict caption lists | 36/40 / 28/40 |
| Caption object IDs accounted for | 526/763 |
| Valid relation tuples (both IDs resolve) | 183/186 |
| Relation agreement F1 vs cached Qwen3.8 | 0.398139 |
| Caption word F1 vs cached Qwen3.8 | 0.509366 |
| Mean relation / caption request latency | 7.1324 s / 7.1410 s |
| Total paired inference time | 570.94 s |
| Artifact wall time, create-to-file-mtime | 575.61 s |
| Model VRAM | 2,734 MiB in a separate model-control job/status record |

## Configuration and inputs

- **Model:** LiquidAI LFM3-VL-2B NVFP4. The artifact's model field points to `LFM3-VL-2B-NVFP4.gguf` in snapshot `1add221a5ea6c047fcf1eaaf2d247939d7cf8849`.
- **Weights / projector:** NVFP4 GGUF weights with F16 multimodal projector.
- **Backend:** managed llama.cpp OpenAI-compatible service; the artifact records `http://127.0.0.1:35229/v1`. Exact llama.cpp build/version was not captured.
- **Context:** 16,384 tokens.
- **Sampling and generation:** the matched harness sent a 4,096-token output cap. Service temperature/top-p settings were not captured in the run artifact.
- **Data:** Replica `room0`; the same 40 annotated stride-50 frames and modernized YOLOE detection cache `s_modernized_yoloe26l_stride50` used for the matched comparison. Per-frame labels, annotations, latency, token usage, and raw responses are embedded in the JSON.
- **Prompts:** same shared relation/caption prompt protocol as the other candidates: `system_prompt_only_top` / `system_prompt_captions`, image-only user turn, then text-only user turn. This endpoint artifact does not preserve the serialized request body or prompt template text.
- **Failures:** all 40 frame pairs returned. Strict validation failed for 4 relation responses and 12 caption responses; caption ID accounting covered 526 of 763 expected IDs. No consolidation ran (`0` completed).
- **Resource provenance:** the VRAM figure came from a separate model-control job/status record, not the JSON. The create-to-mtime interval is a filesystem proxy, not an embedded process wall-time field.

## Reading the score

Agreement is measured against cached Qwen3.8 output, not independent truth. Caption word F1 is lexical overlap. The valid relation tuple count (183/186) means both object IDs resolve in the frame labels; it is not relation precision or recall against a scene-graph annotation set. The result makes LFM3 a relevant graph-output candidate, but this run does not support replacing the provisional Qwen3.8 default.

## Evidence

- JSON and embedded raw responses: [`vlm-lfm3-vl-2b-nvfp4-40.json`](data/vlm-lfm3-vl-2b-nvfp4-40.json)
- Captured creation time: `2026-11-14T00:59:51.169218+00:00`.
- Artifact file modification time: `2026-11-14T01:09:26.780235+00:00`.
