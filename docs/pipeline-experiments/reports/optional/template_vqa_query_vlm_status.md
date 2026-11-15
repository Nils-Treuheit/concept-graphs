# Template-matching VQA and query-time VLM status

**Status:** no ConceptGraphs template-VQA run has produced end-to-end scores. The available generic image-VQA panel and per-frame annotation benchmark are different tasks. The existing Replica feature benchmark measures category retrieval, visual cross-view identity, and identity effects on fixed graph edges; it does not evaluate question answering.

ReplicaSSG has 11 held-out test scans. Local RGB sequences and poses overlap with five: `office_1`, `office_3`, `office_4`, `room_1`, and `room_2`, containing 91 scene-scoped annotated relation rows. No ConceptGraphs maps or frozen query/evidence manifest exist for these scenes. `room0` is outside the held-out test list. A valid run should generate maps on the five-scene intersection, freeze relation questions and evidence packets, then compare retrieval and answerers separately.

## Query-label audit

The local ReplicaSSG files provide object IDs/categories in `objects.json` and directed predicate tuples in `relationships.json`. They contain no aligned natural-language question/answer annotations or object captions for these test scenes. A deterministic question generated from a relation tuple would measure a synthetic graph-query task, not the planned natural-language template-VQA task, so no answer-accuracy benchmark was fabricated from the triples.

Before a valid query-time run, generate held-out ConceptGraphs maps and freeze a template gallery, questions, retrieved evidence packets, answer schema, predicate-direction rules, and independently reviewed reference answers. The five-scene RGB/pose intersection can then support a matched answerer comparison. The existing GT-mask retrieval run evaluates category retrieval and visual association only.

| Answerer | Readiness | Target VQA result |
|---|---|---|
| [Qwen3.8 27B](query_qwen38.md) | Configured local UD-Q4_K_XL MTP4 baseline. Existing 40-frame result is per-frame annotation only. | Not measured |
| [Qwen3-VL 8B Instruct Q8](query_qwen3vl_8b_instruct.md) | Local GGUF and projector cached; historical generic panel run succeeded. | Not measured |
| [Qwen3-VL 30B Thinking Q4](query_qwen3vl_30b_thinking.md) | Local GGUF and projector cached; intended complex/affordance role. Serving and context fit unverified. | Not measured |
| [Qwen3-VL 8B Thinking Q8](query_qwen3vl_8b_thinking.md) | Historical generic panel result; model files absent in the current cache search. | Not measured |

For every answerer, answer correctness, object attribution, relation direction, evidence support, abstention, schema validity, target latency, and peak GPU use are not measured. Generic panel scores belong only to `.agentic_workflow/workspace/vlm-selection/evidence/qwen-vqa-results.json` and must not be labeled ConceptGraphs query-time results. Full held-out protocol, prerequisites, metrics, and provenance are in this index's per-answerer reports and `scripts/pipeline_model_reports/feature_encoders.md` for existing retriever evidence.
