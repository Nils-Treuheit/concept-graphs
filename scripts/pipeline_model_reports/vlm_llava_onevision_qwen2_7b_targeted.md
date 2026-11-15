# LLaVA OneVision Qwen2 7B: targeted object-caption benchmark

## Result

One-object prompts produced 718 nonempty descriptions from the same 763 cached detections, compared with 714 usable entries from the structured batch-caption arm. The targeted strategy adds four descriptions (0.52 percentage points) and explicitly returns `NOT_VISIBLE` for 45 detections. No target requests errored or returned an empty string.

| Measure | Result |
|---|---:|
| Complete frame pairs | 40/40 |
| Target calls with valid stream record | 763/763 |
| Target request errors / empty responses | 0 / 0 |
| Explicit `NOT_VISIBLE` responses | 45/763 |
| Nonempty target descriptions | 718/763 (94.10%) |
| Caption word F1 vs cached Qwen3.8 | 0.2518 over 666 shared IDs in 37 frames |
| Relation JSON and schema valid | 14/40 (35.0%) |
| Relation endpoints resolve and predicates allowed | 10/40 (25.0%) |
| Relation agreement F1 vs cached references | 0.0933 over 14 parseable frames; 3 scored frames had empty references |
| Mean relation request / total caption generation per frame | 34.332 s / 19.604 s |
| Mean caption generation latency per target | 1.028 s |
| Summed generation latency | 2,157.40 s across 803 requests |
| Wall time / model load | 2,178.98 s / 2.78 s |
| Peak CUDA allocated / reserved | 17,024 / 17,406 MiB |
| Peak sampled whole-device use | 19,703 MiB |

The count `target_requests_with_valid_record` denotes completed responses or caught errors present in the sidecar. This run completed all 763 requests. If interrupted in-flight, exact generation attempts cannot be recovered from the sidecar; `target_requests_without_valid_record` would count missing records, not calls never started.

## Configuration

- **Model and runtime:** same checkpoint, revision, Transformers/PyTorch versions, float16, CUDA 0, greedy decoding, and local-files-only loading as the structured arm. See the [structured arm report](vlm_llava_onevision_qwen2_7b_structured.md).
- **Data:** same 40 Replica `room0` frames at stride 50, cached annotated images and YOLOE labels, and the same 763 observations.
- **Prompt profile:** LLaVA model-card single-user-turn format, task text before the image placeholder, no system prompt, rendered with `apply_chat_template`.
- **Relation generation:** the structured arm’s relation prompt and 4,096-token cap were reused unchanged.
- **Caption generation:** one prompt per cached detection, maximum 256 output tokens per target. It asks for one concise factual sentence about visible appearance and condition, mentions actions only when clear, and permits exact `NOT_VISIBLE` when the target cannot be identified. A soft 3,600-second wall budget is checked between generations; request timeout is post-hoc only.
- **Stitching:** the harness attaches the prompted cached ID and detector name to each returned description. It does not use a second LLM or score these IDs as model-generated. This keeps targeted caption quality separate from identifier generation.

## Comparison with batch JSON

| Caption strategy | Usable descriptions / 763 | Word F1 to cache | Caption generation calls | Total wall |
|---|---:|---:|---:|---:|
| Structured batch JSON | 714 (93.58%) | 0.4088 over 668 IDs | 40 | 1,583.10 s |
| Targeted object prompts | 718 (94.10%) | 0.2518 over 666 IDs | 763 | 2,178.98 s |

The two lexical F1 scores have different scored text and are not semantic accuracy. Targeted prompts add 4 usable descriptions while taking 595.88 seconds longer. They are useful when explicit per-detection uncertainty is desired; batch JSON is faster and has stronger lexical agreement to the cached Qwen3.8 annotations. This small coverage gain alone does not justify changing the production caption path. The cache does not establish whether those Qwen3.8 caption references are raw or consolidated outputs, so the lexical score's precise reference provenance remains unknown.

The relation raw responses match the structured arm exactly on all 40 frames. Thus the second relation pass reproduces the first arm’s outputs exactly in this run. Relation agreement remains weak and only 14 responses were parseable; neither arm resolves the earlier map-level relation propagation issue.

## Artifacts

- Raw JSON report: [`vlm-llava-onevision-qwen2-7b-targeted-40.json`](data/vlm-llava-onevision-qwen2-7b-targeted-40.json)
- Per-target raw response stream: [`vlm-llava-onevision-qwen2-7b-targeted-40.json.targets.jsonl`](data/vlm-llava-onevision-qwen2-7b-targeted-40.json.targets.jsonl)
- Structured comparison: [`vlm_llava_onevision_qwen2_7b_structured.md`](vlm_llava_onevision_qwen2_7b_structured.md)
- Plan: `.agentic_workflow/workspace/pipeline-experiments/llava_prompt_experiment_plan_v5.md`
