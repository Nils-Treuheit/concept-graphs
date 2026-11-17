# VLM role benchmark status

**Evidence checked:** 2026-11-17. Five per-frame candidates have a matched 40-frame Replica `room0` comparison, and the NInfer Qwen3.8 27B NVFP4 route completed a sixth measured 40-frame matched-rerun run. LLaVA has an additional prompt-format and targeted-caption comparison. LFM3-VL-2B BF16 completed on RTX 5090 with a strict-format failure for this harness. The contemporaneous Ollama Qwen3.8 control and the two NInfer Qwen3.6 profiles remain pending. Individual run evidence and limitations are in the linked reports below.

## Per-frame annotation shortlist

Use the following focused comparison for object descriptions and relation annotations on the modernized ConceptGraphs path:

| Role | Candidate | Rationale for inclusion |
|---|---|---|
| Provisional baseline | Qwen3.8 27B, `qwen3.8:27b-mtp4-ud_q4_K_XL-128k` | Existing ConceptGraphs annotations and the current configured baseline. It remains provisional because it was faster and more reliable among only the candidates previously tested directly; that evidence does not make it a production winner. |
| Candidate | `llava-hf/llava-onevision-qwen2-7b-ov-hf` | The external VLM benchmark reports this model as the description-task leader. This is the 7B Qwen2 OneVision model, not LLaVA-OneVision-2 8B. |
| Candidate | `LiquidAI/LFM3-VL-2B-GGUF:NVFP4` | The external benchmark reports it as the GRAB-Lite graph-analysis leader. |
| Candidate | `Qwen/Qwen2.5-VL-7B-Instruct` | The external benchmark reports the best six-category VQA panel macro for this model. |

The local VLM benchmark index at `/mnt/HDD1/Project_Code/VLM_benchmark/index.html#benchmark-leaders` reports its general image-task leaders through 2026-11-12. Qwen3.8 scored 63.5% on the complete eight-task macro, Qwen2.5-VL scored 67.33% on the six-category VQA panel macro, LLaVA OneVision led visual description at 80.8%, and LFM3-VL NVFP4 led GRAB-Lite at 33.8%. These external results motivate candidates; they do not establish ConceptGraphs performance.

Do not include Qwen3-VL 32B in this shortlist unless VLM object counting becomes central. It is not a current pipeline objective.

## Completed matched comparison

All six model routes used the same 40 annotated source frames from Replica `room0`, stride 50, and the saved modernized YOLOE detections (`s_modernized_yoloe26l_stride50`). They used the same cached annotated images, labels, relation and caption prompts. The three Transformers runs record the user prompt templates directly; the managed endpoint artifacts (Ollama Qwen3.8, LFM3 NVFP4, and NInfer Qwen3.8) retain raw responses but not a complete serialized request record. LFM3 BF16 merged the shared system and task text into a single user turn with the image content, differing from other Transformers routes that used distinct system and user turns. The NInfer Qwen3.8 route sent the same legacy messages over OpenAI-compatible HTTP, but the NInfer server applied its own chat template and image preprocessing server-side, so its request serialization is not identical to the Transformers routes.

| Model | Strict relation outputs | Strict caption outputs | Caption IDs | Valid relation tuples (both IDs resolve) | Qwen3.8 agreement: relation F1 / caption word F1 | Mean relation / caption latency | Peak memory | Report |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Qwen3.8 27B, UD Q4_K_XL MTP4 | 40/40 | 40/40 | 763/763 | 339/339 | 0.497647 / 0.646177 | 6.5524 / 6.7565 s | separate model-control sample: model 19,194,928,167 bytes; total 25,635 MiB | [Qwen3.8 report](vlm_qwen38_27b.md) |
| LFM3-VL-2B NVFP4 | 36/40 | 28/40 | 526/763 | 183/186 | 0.398139 / 0.509366 | 7.1324 / 7.1410 s | separate model-control record: 2,734 MiB | [LFM3 NVFP4 report](vlm_lfm3_vl_2b_nvfp4.md) |
| LFM3-VL-2B BF16 (Transformers) | 0/40 | 0/40 | 0/763 | 0/0 | null / null | 25.8855 / 24.4442 s | Transformers peak 3,836.3 MiB allocated, 3,890.0 MiB reserved; nvidia-smi 6,105 MiB | [LFM3 BF16 report](vlm_lfm3_vl_2b_transformers_bf16_40.md) |
| LLaVA OneVision Qwen2 7B | 0/40 | 0/40 | 0/763 | 0/0 | 0.400000 / not scored | 7.0744 / 1.3651 s | Transformers peak 17,029.6 MiB allocated, 17,420 MiB reserved; system 19,212 MiB | [LLaVA report](vlm_llava_onevision_qwen2_7b.md) |
| Qwen2.5-VL 7B Instruct | 38/40 | 40/40* | 762/763 | 149/149 (298/298 endpoints) | 0.331109 / 0.338108 | 3.1798 / 5.6985 s | Transformers peak 16,198.8 MiB allocated, 16,470 MiB reserved; system 18,220 MiB | [Qwen2.5 report](vlm_qwen2_5_vl_7b.md) |
| Qwen3.8 27B, NInfer NVFP4 (matched-rerun route) | 0/40 (25/40 legacy parses valid) | 38/40 | 734/763 | 0/0 (strict; no parse passed schema) | null (unscorable, 0 frames) / 0.6744566 | 12.6060 / 5.0626 s; 8.8343 s combined over 80 requests | Whole-device peak 29,720 MiB (nvidia-smi, 697 samples); process scope unavailable (model-control-owned process) | [NInfer Qwen3.8 report](vlm_qwen38_ninfer_room0_40.md) |

`*` Qwen2.5 caption validity is a post-hoc correction from the retained raw fenced-JSON responses. The original artifact summary predates the fence-aware validator and reports zero valid caption frames. The corrected count is 40/40; the raw responses were not changed.

The NInfer Qwen3.8 row is a distinct serving route from the cached Ollama Qwen3.8 provisional baseline: NVFP4 NInfer weights served over loopback HTTP (`127.0.0.1:20301/v1`) with server-side chat templating, versus the Ollama UD Q4_K_XL GGUF route. Its 0/40 strict relation result coexists with 25/40 legacy parses; relation agreement is null (0 scorable frames), which is unscorable, not zero accuracy. Its caption word F1 (0.6744566) is lexical agreement to cached Qwen3.8 outputs, not ground truth.

These are agreement and format results, not ground-truth accuracy. Qwen3.8 remains the **provisional default**. Its agreement results are against its own cached outputs and are self-favoring; they do not independently establish higher annotation quality or an overall production winner. LLaVA's 0/40 strict-format result is an output-format failure for this harness even though all requests completed. The LFM3-VL-2B BF16 Transformers route also achieved 0/40 strict-format validity for both relations and captions, despite completing all 80 requests without error; its agreement metrics are null, not zero accuracy. One initial LLaVA launch failed during import before CUDA/model execution; the runner import was corrected, and that launch is not counted as a model failure. The NInfer Qwen3.8 route completed all 80 requests without error; its 0/40 strict relation result means its relation agreement is null (unscorable), not zero accuracy. Two preflight-only launch failures (a sandboxed loopback denial and a host `Connection refused`, both on the `/v1/models` preflight) preceded the successful run; neither sent a chat completion or wrote benchmark output, so neither is counted as a model failure.

The prompt and image inputs were matched. Generation settings were not proven equivalent: Transformers used greedy decoding, while the managed Qwen3.8 and LFM3 GGUF services used their own uncaptured sampling defaults. The NInfer Qwen3.8 route sent explicit decode parameters (temperature 0, top_p 1, max_tokens 4096, n=1) as a documented greedy approximation, with server-reported token counts. Prompt serialization also differs by route: LFM3 BF16 merged the shared system and task text into a single user turn with the image content, while other Transformers routes used distinct system and user turns. The NInfer Qwen3.8 route sent the same legacy messages over OpenAI-compatible HTTP, but the NInfer server applied its own chat template and image preprocessing server-side. Timing and output differences reflect model-plus-serialization effects, not pure weights-only comparisons. Transformers latency covers model generation after preprocessing, while endpoint latency includes HTTP/server work. Memory scopes differ between model-control samples and Transformers allocator/system peaks. See each report for source and missing fields. Word-F1 measures lexical overlap, not semantic caption quality.

## LLaVA prompt and caption strategy

The official Hugging Face model-card conversation format was evaluated on the same checkpoint and frozen 40-frame cache. A one-user-turn JSON contract yielded valid caption JSON on 40/40 frames and 714 usable descriptions from 763 detections, compared with the previous legacy prompt’s 0/40 strict caption parses. Relation JSON remained weak at 14/40 valid responses and 10/40 endpoint/predicate-valid frames.

The per-object prompt arm returned 718 nonempty descriptions from 763 detections, a gain of four over batch JSON, with 45 explicit `NOT_VISIBLE` responses. It used 763 caption calls instead of 40 and took 595.88 seconds longer. Lexical agreement to the cached Qwen3.8 text was 0.2518 versus 0.4088 for batch mode. These metrics are not semantic accuracy. The current evidence favors batch JSON when efficiency and lexical compatibility matter; targeted prompts may be useful when explicit per-object abstention is valued. The small coverage gain does not justify changing the provisional production caption path.

The targeted harness attaches cached IDs and names deterministically. It does not use a second LLM and does not treat IDs as model output. The relation responses were identical across both runs on all 40 frames. See the separate [structured-prompt report](vlm_llava_onevision_qwen2_7b_structured.md) and [targeted-caption report](vlm_llava_onevision_qwen2_7b_targeted.md). The earlier 0.400 relation F1 is an empty-output artifact and should not be compared with the corrected JSON parser results.

## Query-time question answering

Query-time VLM evaluation is a separate role from per-frame annotation and caption consolidation. The project plan proposes evaluating a fixed answerer on retrieved object/template evidence for complex and affordance questions. No held-out query-time comparison or end-to-end VQA score is present in the available ConceptGraphs results. Revisit that role after the core map, object association, and annotation path are validated.

## Caption consolidation is separate

The measured Qwen3.5/Gemma caption-consolidation pilot is summarized in [`caption_consolidation.md`](caption_consolidation.md). It merges map captions after mapping. Its lexical agreement scores do not select a per-frame annotation or query-time VLM.

## Provenance and limits

- ConceptGraphs candidate roles and planned metrics: [`MODERNIZED_PIPELINE_IMPLEMENTATION_PLAN.md`](../../MODERNIZED_PIPELINE_IMPLEMENTATION_PLAN.md)
- ConceptGraphs benchmark protocol: [`MODERNIZED_COMPONENT_BENCHMARKS.md`](../MODERNIZED_COMPONENT_BENCHMARKS.md)
- Existing ConceptGraphs results: [`MODERNIZED_COMPONENT_RESULTS.md`](../MODERNIZED_COMPONENT_RESULTS.md)
- External leader context: `/mnt/HDD1/Project_Code/VLM_benchmark/index.html#benchmark-leaders`

The per-frame reports preserve the source artifact paths, raw output evidence, exact captured runtime settings, and fields that were not recorded. External VLM leader rankings motivated the candidates but did not determine these ConceptGraphs results.
