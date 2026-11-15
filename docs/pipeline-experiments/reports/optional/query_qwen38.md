# Query-time answerer: Qwen3.8 27B

**Status:** provisional query-time baseline; ConceptGraphs template-VQA metrics are **not measured**.

## Exact candidate and readiness

- Model ID: `qwen3.8:27b-mtp4-ud_q4_K_XL-128k`
- Quantization/serving: UD Q4_K_XL, MTP4, 128k context profile on the managed local Ollama route. The local model-control inventory also lists the configured Qwen3.8 27B NVFP4 NInfer profile; this report does not substitute that profile for the measured Ollama candidate.
- Prior in-project evidence: the matched per-frame annotation run completed 40/40 frames with 40/40 strict relation and caption outputs. Mean request latency was 6.55 s for relations and 6.76 s for captions. Those are annotation timings and do not estimate query-answering performance.
- Query-time checkpoint readiness: local baseline is configured, but no query-role prompt/evidence packet or query-time result is present in the project artifacts.

## Evaluation status

No template retrieval, map evidence, answer, object-attribution, answer-accuracy, abstention, schema-validity, or query-latency measurement was recorded for this candidate. Generic image-VQA panel scores for other models do not establish Qwen3.8 query quality.

A valid run requires held-out ConceptGraphs maps and a frozen query/evidence manifest. Local ReplicaSSG test annotations intersect local RGB sequences in five scenes (`office_1`, `office_3`, `office_4`, `room_1`, `room_2`), but map outputs are currently absent for those scenes. `room0` is not in the test split.

## Fixed comparison protocol

Use Qwen3.8 as the answerer baseline after retrieval evidence is frozen. Keep the same question, top-k gallery evidence, template images, scene graph facts, prompt, schema, context cap, and decoding settings as the Qwen3-VL candidates. Score exact/normalized answers, correct object attribution, predicate direction, support by supplied evidence, abstention/error on unanswerable queries, schema validity, and latency. Capture model start, warm/cold per-question wall time, prompt and output tokens, context usage, peak GPU VRAM, and failures.

| Requested field | Query-time result |
|---|---|
| Dataset/split | Not run; intended ReplicaSSG test-scene intersection above |
| Retrieval/VQA metrics | Not measured |
| Runtime/resources | Not measured for query role |
| Failures | No query attempt |
| Artifact | No query predictions or evidence packets; see [`template_vqa_query_vlm_status.md`](template_vqa_query_vlm_status.md) |

**Decision:** retain Qwen3.8 as the provisional baseline only. No production query-answerer conclusion follows from the annotation run.
