# Query-time answerer: Qwen3-VL 30B Thinking

**Status:** local quantized checkpoint and projector are cached; ConceptGraphs query-VQA metrics are **not measured**.

## Exact candidate and readiness

- Model ID: `unsloth/Qwen3-VL-30B-A3B-Thinking-1M-GGUF:UD-Q4_K_XL`
- Checkpoint: `/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub/models--unsloth--Qwen3-VL-30B-A3B-Thinking-1M-GGUF/snapshots/e5dd74ed359da870ef2e010d06dc8e8402f9af55/Qwen3-VL-30B-A3B-Thinking-1M-UD-Q4_K_XL.gguf` (17,740,830,944 bytes).
- Vision projector: adjacent `mmproj-F16.gguf` (1,083,500,128 bytes).
- Intended role from the project plan: answer complex and affordance questions after retrieval supplies object/template evidence. It is not intended for per-frame annotation.
- Local weights exist. The supported serving command, context fit, warm/cold latency, and memory residency for this task were not checked or measured. The `1M` in the model name is not a measured or required working context.
- No result for this model was found in the saved generic VQA panel artifact or in ConceptGraphs query-role artifacts.

## Evaluation status

No held-out ConceptGraphs query/evidence packet was run. Retrieval, answer accuracy, object attribution, relation direction, evidence support, affordance correctness, abstention, schema success, latency, and peak GPU use are **not measured**. Do not infer query quality or production suitability from model size, intended role, or generic VLM rankings.

A valid comparison must feed the same question and byte-identical retrieved evidence to this model and the smaller answerers. Benchmark whether its task-specific gains justify additional latency and residency.

## Fixed comparison protocol

Use the same held-out map gallery, query list, retrieval backend output, crop/view evidence, graph facts, prompt, output schema, context cap, and decoding configuration as the Qwen3.8 and Qwen3-VL 8B Instruct arms. Measure exact/normalized answers, object attribution, predicate direction, evidence support, affordance accuracy on independently labeled questions, abstention/error, schema validity, per-question latency, peak VRAM, model startup, and token/context use. Preserve all raw responses and failures.

| Requested field | Query-time result |
|---|---|
| Dataset/split | Not run; intended ReplicaSSG held-out test-scene intersection |
| Retrieval/VQA metrics | Not measured |
| Runtime/resources | Checkpoint size recorded; query inference, latency, and VRAM not measured |
| Failures | No query attempt |
| Artifact | No query predictions/evidence packets; see [`template_vqa_query_vlm_status.md`](template_vqa_query_vlm_status.md) |

**Decision:** evaluate only as the planned complex-question candidate. The cached checkpoint is not evidence of a winning query model.
