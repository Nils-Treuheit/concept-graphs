# Integrated 40-frame VLM report update and matched runtime comparison

**Status:** updated LLaVA prompt results are integrated into the 40-frame page. GPU runs for LFM3 BF16 and NInfer routes remain pending the required local-model release workflow. No new model run or GPU-state probe has been made.

## Report-page update

Keep this work in the existing 40-frame per-frame VLM report table so the reader sees the current prompt variants and remaining runtime comparisons together. Keep the separate 825-example general suite as its own section; those scores answer a different question.

Replace the legacy LLaVA row as the current result with these completed outputs, while retaining the legacy format failure as a clearly labeled historical row:

| LLaVA prompt variant | Relations | Captions | Agreement to cached Qwen3.8 | Runtime |
|---|---|---|---|---|
| Structured batch JSON | Relation JSON 14/40; endpoints and predicates valid 10/40 | 40/40 JSON; 714/763 usable | Relation F1 0.0933; caption word F1 0.4088 | 1,583.10 s wall; 40 caption calls |
| Targeted object captions | Same relation outputs as structured run | 718/763 nonempty; 45 `NOT_VISIBLE` | Relation F1 0.0933; caption word F1 0.2518 | 2,178.98 s wall; 763 caption calls |
| Legacy free-prose prompt | 0/40 strict relation and caption parses | 0/763 accepted IDs | Not meaningful | Retain as failed-format history |

Do not collapse these LLaVA variants into one accuracy value: their caption prompts, call counts, and scored text differ. Keep the current report caveat that agreement is against cached Qwen3.8 outputs, not independent semantic ground truth. Link both current LLaVA reports and raw JSON in the 40-frame section.

## Matched rerun candidates

Use the existing 40 Replica `room0` frames, cached YOLOE detections and masks, object IDs, relation/caption prompts, parsers, and cached Qwen3.8 reference outputs. Each model receives the same ordered frame inputs independently. Do not pass one model's response to another.

| Candidate | Exact route |
|---|---|
| Qwen3.8 control | Existing configured Ollama UD Q4_K_XL MTP4 route; rerun with the shared request protocol for a contemporaneous baseline. |
| LFM3-VL-2B | Full-weight official Transformers BF16 checkpoint. Its completed 825-example general benchmark is not a substitute for this run. |
| NInfer Qwen3.8 | User-confirmed OpenCode model ID: `ninfer/qwen3.8-27b-nvfp4`. |
| NInfer Qwen3.6 | User-confirmed OpenCode model ID: `ninfer/qwen3.6-35b-a3b`. |
| NInfer Qwen3.6 MTP3 | User-confirmed OpenCode model ID: `ninfer/qwen3.6-35b-a3b-mtp3`. Compare against the same Qwen3.6 route to isolate the MTP3 profile. |

Run routes sequentially on the shared GPU. Keep model IDs, checkpoint/profile hashes, runner/runtime versions, dtype/quantization, context, MTP settings, and all decode parameters in each run manifest. Hold prompts, images, frame ordering, output caps, and validators constant where supported. If a parameter differs by runner, record the exact difference and do not present the timing as perfectly matched.

## Metrics and report placement

Add a matched-run subsection beside the existing 40-frame table. Report relation/caption schema pass counts, valid endpoint/predicate tuples, caption coverage, model failures, relation and caption agreement to cached Qwen3.8, cold load time, first-request latency, warm request p50/p95, total 40-frame wall time, effective frames/s, and peak process plus whole-device VRAM with the memory scope stated. Report relation/caption request timing separately. Include preprocessing, service/transport, generation, parsing, and serialization when measurable.

Preserve old values as historical where their sampling settings or memory scope differ. Do not overwrite them. Keep the LLaVA prompt variants in the current-results table; keep the new BF16/NInfer route runs in a separate matched-rerun subsection so incomplete runs are visibly preliminary.

## Execution prerequisite and current status

The user confirmed these NInfer model IDs are available in OpenCode. The correct release workflow exists at `.agentic_workflow/src/executor.py` in this checkout; the earlier attempts used the wrong checkout and therefore did not establish that the release entry point was missing. The latest release attempt returned `deep_cleanup=cleanup_failed` and `fast_cleanup=cleanup_failed` at about 2026-11-16 19:36 UTC. Retry only after the one-hour cleanup wait, inspect the controller and session gates separately, and start no GPU probe or model job until release succeeds and idle state is confirmed.

Run each route through the owned OpenCode server/session flow, check GPU release before every probe/job, and verify each owned session is inactive before archiving it. Preserve raw outputs and publish only completed results; label partial runs preliminary. Store persistent outputs and caches under the project HDD, never `/tmp`.

## Existing evidence

- [Current 40-frame comparison](../vlm_roles.md)
- [LLaVA structured prompt result](../../../../scripts/pipeline_model_reports/vlm_llava_onevision_qwen2_7b_structured.md)
- [LLaVA targeted-caption result](../../../../scripts/pipeline_model_reports/vlm_llava_onevision_qwen2_7b_targeted.md)
- [LFM3-VL-2B official BF16 general benchmark](../vlm_lfm3_vl_2b_transformers_bf16.md)
