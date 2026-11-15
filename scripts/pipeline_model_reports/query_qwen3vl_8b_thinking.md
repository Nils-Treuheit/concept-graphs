# Optional query-time answerer: Qwen3-VL 8B Thinking

**Status:** historical generic-panel result exists; current local checkpoint readiness is unresolved; ConceptGraphs query-VQA metrics are **not measured**.

## Exact candidate and readiness

- Historical model ID: `unsloth/Qwen3-VL-8B-Thinking-GGUF:UD-Q8_K_XL`.
- Historical configuration: `llama.cpp`, UD-Q8_K_XL, 8,192-token context, Q8 KV cache, MTP unsupported. The saved benchmark says the checkpoint contains no MTP layers.
- The prior generic VQA panel completed 25 examples per task. It scored truth 82.4%, count 68.4%, description 54.0%, affordance 4.0%, and negation 60.0%; peak total GPU use was 13.379 GiB. Mean latency was 3.05 s on the generic affordance task and 5.25 s on negation.
- The current local Hugging Face cache search did not find the model snapshot files. Prior panel logs/artifact are historical evidence only; redownload/materialization and a new serving preflight are required before a local query run.

## Evaluation status

No ConceptGraphs retrieval evidence or held-out question was sent to this model. Query answer accuracy, attribution, evidence support, abstention, schema validity, latency, and resource use are **not measured**. Generic direct-image VQA scores do not establish query-time scene-graph reasoning quality.

## Fixed comparison protocol

If the checkpoint is restored, add this as an optional reasoning-mode arm to the same held-out query/evidence manifest used by Qwen3.8, Qwen3-VL 8B Instruct, and 30B Thinking. Keep the prompt, byte-identical evidence, schema, context cap, and decoding fixed. Score exact/normalized answer, object attribution, relation direction, evidence support, abstention/error, schema success, latency, and GPU residency.

| Requested field | Query-time result |
|---|---|
| Dataset/split | Not run; intended ReplicaSSG held-out test-scene intersection |
| Retrieval/VQA metrics | Not measured |
| Runtime/resources | Historical generic panel only; target query role not measured |
| Failures | No query attempt; current cache snapshot absent |
| Artifact | `data/generic-vqa-panel-results.json`; query protocol in [`template_vqa_query_vlm_status.md`](template_vqa_query_vlm_status.md) |

**Decision:** keep optional. Do not compare it to the instruct model as a ConceptGraphs winner until the exact checkpoint is restored and both receive the same retrieved evidence.
