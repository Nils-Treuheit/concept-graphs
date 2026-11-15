# Query-time answerer: Qwen3-VL 8B Instruct

**Status:** local Q8 checkpoint available; ConceptGraphs template-VQA metrics are **not measured**.

## Exact candidate and readiness

- Model ID: `unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q8_K_XL`
- Checkpoint: `/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub/models--unsloth--Qwen3-VL-8B-Instruct-GGUF/snapshots/b93a7ee713758252c555be4210c00540df954dc2/Qwen3-VL-8B-Instruct-UD-Q8_K_XL.gguf`
- Vision projector: adjacent `mmproj-F16.gguf` (1,159,030,336 bytes). The Q8 model file is 10,824,039,264 bytes.
- Historical generic-panel runtime: `llama.cpp`, 8,192-token context, Q8 KV cache, no MTP. The run completed 25/25 examples in each task. Peak total GPU use was 15.839 GiB. This confirms prior generic-panel execution, not current query VQA readiness or a fresh residency check.
- Generic image-VQA scores: truth 100.0%, count 76.4%, description 62.4%, affordance 4.0%, negation 64.0%. Each task had 25 fixed direct-image examples. They do not use retrieved ConceptGraphs evidence and must not be labeled query-time pipeline scores.

## Evaluation status

No retrieved template packet, held-out map, or ConceptGraphs query has been passed to this model. Answer accuracy, object attribution, evidence support, abstention, schema validity, query latency, and query-run GPU use are **not measured**.

The generic affordance score is 1/25 on its distinct image panel. It cannot establish performance on retrieved object affordance questions. A valid query comparison requires the same held-out query set and byte-identical retrieval evidence supplied to each answerer.

## Fixed comparison protocol

Compare this exact Q8 configuration against the Qwen3.8 baseline and Qwen3-VL 30B Thinking. Freeze question wording, gallery, top-k results, image crops, graph facts, answer schema, context cap, and decoding. Score exact/normalized answer accuracy, object attribution, predicate direction, evidence support, abstention/error, schema validity, latency, and peak GPU memory. Record model startup, cold/warm timing, input/output tokens, and errors.

| Requested field | Query-time result |
|---|---|
| Dataset/split | Not run; intended ReplicaSSG held-out test-scene intersection |
| Retrieval/VQA metrics | Not measured |
| Runtime/resources | Generic panel above only; query role not measured |
| Failures | No query attempt |
| Artifact | Generic panel JSON: `data/generic-vqa-panel-results.json`; query report: [`template_vqa_query_vlm_status.md`](template_vqa_query_vlm_status.md) |

**Decision:** candidate is available for a matched query-time run. Generic panel scores do not justify replacing the provisional baseline.
