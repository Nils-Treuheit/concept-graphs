# Optional query-time answerer: Qwen3-VL 8B Thinking

**Status:** historical generic-panel result exists; exact checkpoint and projector are now installed locally; ConceptGraphs query-VQA metrics are **not measured**.

## Exact candidate and readiness

- Historical model ID: `unsloth/Qwen3-VL-8B-Thinking-GGUF:UD-Q8_K_XL`.
- Historical configuration: `llama.cpp`, UD-Q8_K_XL, 8,192-token context, Q8 KV cache, MTP unsupported. The saved benchmark says the checkpoint contains no MTP layers.
- The prior generic VQA panel completed 25 examples per task. It scored truth 82.4%, count 68.4%, description 54.0%, affordance 4.0%, and negation 60.0%; peak total GPU use was 13.379 GiB. Mean latency was 3.05 s on the generic affordance task and 5.25 s on negation.
- Checkpoint was downloaded to HF_HOME from `unsloth/Qwen3-VL-8B-Thinking-GGUF` revision `3a82a5f677de74f036032a7b69c6ab756e290ab1`. `Qwen3-VL-8B-Thinking-UD-Q8_K_XL.gguf` is 10,824,039,264 bytes, SHA-256 `4d305afd0180a3ce24cfb105f1f222853c440afb438514b07ddf524f33abf36e`; adjacent `mmproj-F16.gguf` is 1,159,030,336 bytes, SHA-256 `64d5be3f16fb91cfb451155fe4745266e2169ccbe1f29f57bfab27fb7fec389e`. This acquisition did not load the model. A query-role serving preflight and matched evidence run remain pending.

## Evaluation status

No ConceptGraphs retrieval evidence or held-out question was sent to this model. Query answer accuracy, attribution, evidence support, abstention, schema validity, latency, and resource use are **not measured**. Generic direct-image VQA scores do not establish query-time scene-graph reasoning quality.

## Fixed comparison protocol

If the checkpoint is restored, add this as an optional reasoning-mode arm to the same held-out query/evidence manifest used by Qwen3.8, Qwen3-VL 8B Instruct, and 30B Thinking. Keep the prompt, byte-identical evidence, schema, context cap, and decoding fixed. Score exact/normalized answer, object attribution, relation direction, evidence support, abstention/error, schema success, latency, and GPU residency.

| Requested field | Query-time result |
|---|---|
| Dataset/split | Not run; intended ReplicaSSG held-out test-scene intersection |
| Retrieval/VQA metrics | Not measured |
| Runtime/resources | Historical generic panel only; target query role not measured |
| Failures | No query attempt; serving preflight and frozen held-out maps/evidence are still absent |
| Artifact | `data/generic-vqa-panel-results.json`; query protocol in [`template_vqa_query_vlm_status.md`](template_vqa_query_vlm_status.md) |

**Decision:** keep optional. Do not compare it to the instruct model as a ConceptGraphs winner until the exact checkpoint is restored and both receive the same retrieved evidence.
