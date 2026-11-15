# Per-frame VLM benchmark: Qwen2.5-VL 7B Instruct

## Result

This candidate was fastest for relation requests and achieved 40/40 strict caption outputs after validating the retained raw fenced-JSON responses with the fence-aware parser. The JSON summary predates that correction and reports zero valid caption frames; the corrected count below is post-hoc and leaves the raw artifact unchanged.

| Measure | Result |
|---|---:|
| Requests completed | 40/40 frame pairs |
| Strict relation lists / strict caption lists | 38/40 / 40/40* |
| Caption object IDs accounted for | 762/763; no extra or duplicate IDs |
| Valid relation tuples (both IDs resolve) | 149/149; 298/298 individual endpoint IDs |
| Relation agreement F1 vs cached Qwen3.8 | 0.331109 |
| Caption word F1 vs cached Qwen3.8 | 0.338108 |
| Mean relation / caption request latency | 3.1798 s / 5.6985 s |
| Inference time across 80 requests | 355.13 s |
| Wall time including 177.52 s model load | 537.57 s |
| Peak GPU memory | 16,198.8 MiB allocated; 16,470 MiB reserved; 18,220 MiB system total used |

`*` The artifact's original `caption_parse_valid_frames=0` field reflects its earlier parser. The corrected 40/40 figure comes from rechecking the raw fenced-JSON strings against the same list and caption schema. No raw model response or run output was edited.

## Configuration and inputs

- **Model:** `Qwen/Qwen2.5-VL-7B-Instruct`.
- **Revision:** `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- **Backend:** Transformers, local files only; CUDA device `cuda:0`.
- **Quantization / dtype:** no quantization recorded; `bfloat16`.
- **Software:** Transformers `5.20.2`; PyTorch `2.14.1+cu130`.
- **Context / generation:** context limit not separately recorded; `max_new_tokens=4096`, greedy decoding (`do_sample=false`). The artifact notes a 300-second request timeout was post-hoc only and was not enforced during generation.
- **Data:** Replica `room0`, 40 annotated frames at stride 50, modernized detection suffix `s_modernized_yoloe26l_stride50`; same cached images and labels as Qwen3.8.
- **Prompts and turns:** system prompt source `conceptgraph.utils.vlm` (`system_prompt_only_top` / `system_prompt_captions`), followed by an image-only user turn and a separate text-only user turn. The relation template is “Here is the list of labels for the annotations of the objects in the image: {labels}. Please describe the spatial relationships between the objects in the image.” The caption template is “Here is the list of labels for the annotations of the objects in the image: {labels}. Please accurately caption the objects in the image.”
- **Failures:** 40/40 request pairs completed. Two relation responses did not pass strict parsing; all captions passed the post-hoc fence-aware check. Caption coverage missed one expected object ID.

## Reading the score

Agreement is with cached Qwen3.8 outputs, not independent ground truth. Relation F1 and caption word F1 are agreement measures, not physical-relation precision/recall or semantic caption accuracy. The fence correction changes only validity accounting; it does not change measured agreement or latency.

## Evidence

- JSON and embedded raw responses: [`vlm-qwen2.5-vl-7b-40.json`](../data/vlm-qwen2.5-vl-7b-40.json)
- Model load: 177.5220 s; artifact wall time: 537.5747 s.
- Captured creation / finish: `2026-11-14T01:33:01.897844+00:00` / `2026-11-14T01:41:59.472566+00:00`.
