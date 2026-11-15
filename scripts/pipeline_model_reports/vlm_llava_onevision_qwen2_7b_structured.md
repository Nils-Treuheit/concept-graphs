# LLaVA OneVision Qwen2 7B: structured prompt benchmark

## Result

The Hugging Face model-card conversation format plus explicit JSON contracts corrected the earlier free-prose output problem for captions, but relation output remained unreliable. On the frozen 40-frame set, caption JSON passed schema validation on 40/40 frames and yielded 714 usable descriptions from 763 object observations. Relation JSON was valid on 14/40 frames; only 10/40 met endpoint and predicate checks.

| Measure | Result |
|---|---:|
| Complete frame pairs | 40/40 |
| Relation JSON and schema valid | 14/40 (35.0%) |
| Relation endpoints resolve and predicates allowed | 10/40 (25.0%) |
| Valid endpoints / endpoints emitted | 512/526 |
| Allowed predicates / predicates emitted | 263/263 |
| Relation agreement F1 vs cached references | 0.0933 over 14 parseable frames; 3 scored frames had empty references |
| Caption JSON and schema valid | 40/40 (100%) |
| Usable caption observations | 714/763 (93.58%) |
| Exact ID coverage frames | 3/40 |
| ID observations matched / missing / extra | 714 / 49 / 13 |
| Caption word F1 vs cached Qwen3.8 | 0.4088 over 668 shared IDs in 37 frames |
| Mean relation / caption generation latency | 34.317 s / 5.071 s |
| Summed generation latency | 1,575.50 s across 80 requests |
| Wall time / model load | 1,583.10 s / 2.82 s |
| Peak CUDA allocated / reserved | 17,027.3 / 18,090 MiB |
| Peak sampled whole-device use | 19,888 MiB |

The artifact counts 16 empty relation-reference frames in the full 40-frame set. Only 3 of those occur among the 14 parseable relation responses. Agreement is lexical/structural agreement with the cached reference, not semantic accuracy. The cache does not establish whether its Qwen3.8 caption references are raw or consolidated outputs, so exact caption-reference provenance remains unknown.

## Configuration

- **Model:** `llava-hf/llava-onevision-qwen2-7b-ov-hf`, revision `0d50680527681998e456c7b78950205bedd8a068`.
- **Backend:** Hugging Face Transformers `5.19.0`, local files only, `LlavaOnevisionForConditionalGeneration`, CUDA 0.
- **Precision:** float16, no quantization; PyTorch `2.11.0+cu130`.
- **Decoding:** greedy, `do_sample=false`, `max_new_tokens=4096`; 300-second timeout is post-hoc and does not interrupt generation.
- **Data:** Replica `room0`, 40 sampled frames at stride 50, cached modernized YOLOE labels and annotated images from `s_modernized_yoloe26l_stride50`; 763 detection observations. This is a fixed local evaluation sample, not a held-out semantic ground-truth split.
- **Prompt:** one user turn, no system prompt, task text before an image placeholder, rendered with `processor.apply_chat_template`. This follows the pure-Transformers pattern in the [official model card](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf#using-pure-transformers). Relation and caption tasks request one strict JSON array. Captions preserve the cached detection ID and name.
- **Relation contract:** allowed predicates are exactly `on top of` and `under`; endpoint IDs must match the frame’s cached detections.

## Attempts and failures

The first attempt loaded the same checkpoint, then failed before frame 0 with `KeyError: '"id"'`: literal JSON braces in the prompt had not been escaped for Python `.format`. The failed artifact records zero frames and requests, 70.12 seconds wall time, 15,318 MiB peak allocated, 15,320 MiB reserved, and 16,983 MiB peak sampled device use. The harness escaped the literal braces and an independent review cleared the correction. This failure is retained but excluded from the benchmark scores.

The retry completed all 80 generations without request errors. It took 26 minutes 23 seconds wall time. Model loading took 2.82 seconds because the weights were already cached by the preceding attempt.

## Comparison and interpretation

The prior legacy LLaVA run had 0/40 strict relation lists, 0/40 strict caption lists, and 0/763 matched caption IDs. Its relation F1 of 0.4000 is an empty-output artifact and is not comparable. The structured prompt improves caption-format compatibility and usable descriptions to 714/763, while relation JSON still passes only 14/40.

The separate targeted-caption run uses the same model-card turn format and same relation task. Its report compares object-by-object captions against batch JSON over the common 763-observation denominator. The raw relation strings were identical between the two completed runs on all 40 frames.

These results support LLaVA as a viable caption generator under a suitable prompt, but do not establish overall annotation quality. The relation contract and the Qwen3.8 reference provenance remain limitations.

## Artifacts

- Successful raw JSON: [`vlm-llava-onevision-qwen2-7b-structured-40-retry1.json`](data/vlm-llava-onevision-qwen2-7b-structured-40-retry1.json)
- Preserved failed attempt: [`vlm-llava-onevision-qwen2-7b-structured-40.json`](data/vlm-llava-onevision-qwen2-7b-structured-40.json)
- Harness: [`benchmark_vlm_shortlist_transformers.py`](../benchmark_vlm_shortlist_transformers.py)
- Plan: `.agentic_workflow/workspace/pipeline-experiments/llava_prompt_experiment_plan_v5.md`
- Root evaluation: `.agentic_workflow/workspace/pipeline-experiments/llava_harness_root_evaluation_v6.md`
