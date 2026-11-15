# Per-frame VLM benchmark: LLaVA OneVision Qwen2 7B

## Result

All requests completed, but none of the 40 relation responses or 40 caption responses passed the strict list schema. The raw outputs are free prose, so the reported ID and endpoint coverage are zero. This is a harness-format failure, not evidence that the model cannot describe the images when given a different output contract.

| Measure | Result |
|---|---:|
| Requests completed | 40/40 frame pairs |
| Strict relation lists / strict caption lists | 0/40 / 0/40 |
| Caption object IDs accounted for | 0/763 |
| Valid relation tuples (both IDs resolve) | 0/0 |
| Relation agreement F1 vs cached Qwen3.8 | 0.400000* |
| Caption word F1 vs cached Qwen3.8 | Not scored |
| Mean relation / caption request latency | 7.0744 s / 1.3651 s |
| Inference time across 80 requests | 337.58 s |
| Wall time including 85.93 s model load | 435.52 s |
| Peak GPU memory | 17,029.6 MiB allocated; 17,420 MiB reserved; 19,212 MiB system total used |

`*` The artifact records relation agreement F1 of 0.400000 despite no parseable relation lists. Treat this as a mechanically calculated empty-output agreement value, not useful relation-quality evidence.

## Configuration and inputs

- **Model:** `llava-hf/llava-onevision-qwen2-7b-ov-hf`.
- **Revision:** `0d50680527681998e456c7b78950205bedd8a068`.
- **Backend:** Transformers, local files only; CUDA device `cuda:0`.
- **Quantization / dtype:** no quantization recorded; `float16`.
- **Software:** Transformers `5.20.2`; PyTorch `2.14.1+cu130`.
- **Context / generation:** context limit not separately recorded; `max_new_tokens=4096`, greedy decoding (`do_sample=false`). The artifact notes a 300-second request timeout was post-hoc only and was not enforced during generation.
- **Data:** Replica `room0`, 40 annotated frames at stride 50, modernized detection suffix `s_modernized_yoloe26l_stride50`; same cached images and labels as Qwen3.8.
- **Prompts and turns:** system prompt source `conceptgraph.utils.vlm` (`system_prompt_only_top` / `system_prompt_captions`), followed by an image-only user turn and a separate text-only user turn. The relation template is “Here is the list of labels for the annotations of the objects in the image: {labels}. Please describe the spatial relationships between the objects in the image.” The caption template is “Here is the list of labels for the annotations of the objects in the image: {labels}. Please accurately caption the objects in the image.”
- **Failures:** 40/40 request pairs completed. All 80 model responses failed strict list parsing because they were free prose. One initial launch failed during import before CUDA/model execution; the runner import was corrected, and that launch is not counted as a model failure.

## Reading the score

The JSON artifact compares with cached Qwen3.8 output, not independent ground truth. Since neither task produced parseable output, caption word F1 and ID coverage cannot be scored. The speed result is conditional on this run's prompt and output lengths and does not offset the incompatibility with the required structured output contract.

## Evidence

- JSON and embedded raw responses: [`vlm-llava-onevision-qwen2-7b-40.json`](data/vlm-llava-onevision-qwen2-7b-40.json)
- Model load: 85.9339 s; artifact wall time: 435.5243 s.
- Captured creation / finish: `2026-11-14T01:22:33.109834+00:00` / `2026-11-14T01:29:48.634107+00:00`.
