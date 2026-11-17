# NInfer Qwen3.8 27B NVFP4: 40-frame per-frame benchmark

## Result

This run completed on an RTX 5090 through the managed NInfer route `qwen3.8-27b-nvfp4` (OpenCode model ID `ninfer/qwen3.8-27b-nvfp4`) on the frozen 40-frame Replica `room0` set. Every one of the 80 requests finished without error, and no response carried a missing or mismatched model field. However, no relation response passed strict JSON-array/schema/valid-ID-predicate validation (0/40), although the legacy list-of-tuples parser was valid on 25/40 relation responses. Caption strict JSON/schema validation with exact cached-ID coverage passed on 38/40 frames. The relation agreement is null because 0 frames were scorable under the strict gate; that is unscorable, not zero semantic accuracy. The caption word F1 is lexical agreement against the cached Qwen3.8 model outputs, not ground-truth accuracy. This is a distinct serving route from the cached Ollama Qwen3.8 provisional baseline, not a re-run of it.

| Measure | Result |
|---|---:|
| Complete frame pairs | 40/40 |
| Completed relation requests | 40/40 |
| Completed caption requests | 40/40 |
| Request errors | 0 |
| Response model mismatches | 0 |
| Missing frame records | 0 |
| Legacy relation parse valid (list of tuples) | 25/40 |
| Strict relation JSON / schema / valid IDs and predicates | 0/40 |
| Legacy caption parse valid | 38/40 |
| Strict caption JSON / schema and exact-ID-coverage | 38/40 |
| Usable caption observations | 734/763 (96.1992%) |
| Relation agreement F1 vs cached Qwen3.8 | null (0 scorable frames; unscorable, not zero accuracy) |
| Caption word F1 vs cached Qwen3.8 | 0.6744566 (35 scored frames, 677 scored IDs; lexical agreement, not ground truth) |
| First relation request | 19.0621 s, finish reason `length` |
| Relation request latency (40 successful requests) | mean 12.6060 s, median 13.7369 s, p95 19.8240 s; min 1.6464 s, max 20.6864 s |
| Caption request latency (40 successful requests) | mean 5.0626 s, median 3.7671 s, p95 18.3851 s; min 2.7452 s, max 19.1972 s |
| Request latency (80 successful requests) | mean 8.8343 s, median 5.4729 s, p95 19.2015 s; range 1.6464–20.6864 s |
| Prompt / completion / total tokens (server-reported) | 104,836 / 172,199 / 277,035 |
| Created / finished (UTC) | 2026-11-17T06:33:40.051762+00:00 / 2026-11-17T06:45:31.088830+00:00 |
| Total wall time | 711.0371 s (11m 51.04s) |
| Effective frames/s | 0.0562559 |
| Peak whole-device VRAM (nvidia-smi) | 29,720 MiB (697 samples, 1 s interval; includes desktop and other processes) |
| Peak process-scope VRAM | unavailable (NInfer process is owned by the model-control lifecycle, not the runner) |
| Model load time | unavailable (controller load job reports no start timestamp or duration) |

One request per frame for relations and one per frame for captions. The first relation request (frame000000) reached the 4,096-token cap (finish reason `length`). Because no relation response passed the strict schema gate, cached-reference relation F1 is null across 0 scorable frames; the 25/40 legacy parses are parser-validity evidence only and do not produce a scored agreement. In 16 frames the cached Qwen3.8 relation reference was empty (recorded as `relation_agreement_empty_reference_frames`); the remaining scorable gate is the strict JSON/schema validation, which passed on 0 frames. Prompt text and images were held fixed with the Transformers reference; the NInfer server applied its own chat template and image preprocessing server-side, so request serialization is not identical to the Transformers routes and token counts are server-reported. The per-task and combined request latencies are endpoint request wall times over the loopback endpoint, including NInfer service and HTTP work; they are not isolated model generation times.

## Configuration and inputs

- **Model/route:** NInfer managed profile `qwen3.8-27b-nvfp4` (OpenCode model ID `ninfer/qwen3.8-27b-nvfp4`); OpenAI-compatible endpoint `http://127.0.0.1:20301/v1`, loopback only. The runner pins both the model ID and the endpoint and performs a `GET /v1/models` preflight; it starts no model and contacts no model-control service.
- **Artifact:** NVFP4 NInfer artifact, 23,719,715,844 bytes, SHA-256 `74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82` (root-captured NInfer profile manifest, read locally by the runner).
- **Profile:** max context 131,072; KV capacity 131,072 with FP8 KV dtype; DFlash2 speculative decoding with 7 draft tokens and LM-head draft; vision enabled. Managed profile readiness was `job_status=done`, `ready=true`; the controller load duration is unavailable from the controller response.
- **Decoding:** temperature 0.0 (greedy approximation), top_p 1.0, max_tokens 4,096 per request, n=1 (server default, recorded), no retries, 300 s client-side request timeout (post-hoc; it did not interrupt any request).
- **Runner:** `benchmark_vlm_shortlist_ninfer.py` v1.0.0; Python 3.10.12; NumPy 1.26.4; stdlib `urllib.request`; OpenAI SDK not used.
- **GPU:** NVIDIA GeForce RTX 5090, 32,607 MiB total.
- **Data:** Replica `room0`, 40 ordered frames at stride 50 (`frame000000` … `frame001950`), cached modernized YOLOE detections `s_modernized_yoloe26l_stride50`, 763 detection observations. Paired cached Qwen3.8 reference outputs.
- **Prompt serialization:** legacy profile: same relation/caption prompts, labels, cached annotated images, and parser semantics as the Transformers reference (`system text; image-only user turn; text-only user turn`). The NInfer server applied its own chat template and image preprocessing server-side; the model-visible token stream and token accounting may differ even though the raw text prompts are identical. Treat timing and output differences as model-plus-route observations, not a pure weights-only comparison.

## Attempts and failures

Two preflight-only launch failures preceded the successful run; neither sent a chat completion or wrote a benchmark output, and neither is counted as a model response failure.

1. A sandboxed loopback `GET /v1/models` check failed with `Errno 1 Operation not permitted`. That diagnostic was observed in the run session and not retained as a separate file.
2. A host-execution run failed at the same `/v1/models` preflight with `Connection refused` (`URLError: [Errno 111] Connection refused`, raised by the runner's preflight before any chat request). The exact traceback is retained in the implementation workspace file `.agentic_workflow/workspace/pipeline-experiments/vlm-40frame-routes-20261116/qwen38-ninfer-implementation-20261117/benchmark_stdout.log`.

After the GPU release and managed profile readiness (controller load `done`, ready `true`), the successful host run completed with 0 failed requests and 0 model mismatches.

## Comparison and interpretation

| Candidate and route | Completed | Strict relations | Strict captions | Caption IDs |
|---|---:|---:|---:|---:|
| Qwen3.8 27B, Ollama UD Q4_K_XL MTP4 | 40/40 | 40/40 | 40/40 | 763/763 |
| LFM3-VL-2B NVFP4 (GGUF llama.cpp) | 40/40 | 36/40 | 28/40 | 526/763 |
| LLaVA OneVision Qwen2 7B (structured) | 40/40 | 14/40 | 40/40 | 714/763 usable |
| Qwen2.5-VL 7B Instruct | 40/40 | 38/40 | 40/40 | 762/763 |
| LFM3-VL-2B BF16 (Transformers) | 40/40 | 0/40 | 0/40 | 0/763 |
| Qwen3.8 27B NVFP4 (NInfer) | 40/40 | 0/40 | 38/40 | 734/763 |

The NInfer route completed all 80 requests without error but achieved 0/40 strict relation validity, even though the legacy list-of-tuples parser was valid on 25/40 frames; no relation output passed the strict JSON-array/schema/valid-ID-predicate checks, so relation agreement is null (unscorable), not zero semantic accuracy. Caption strict validity passed 38/40 frames with 734/763 usable caption observations (96.1992%), and caption word F1 of 0.6744566 against the cached Qwen3.8 outputs is lexical model-output agreement over 35 scored frames and 677 scored IDs; the cached reference is the provisional baseline's own output, so this does not establish semantic caption quality.

This is a matched-rerun candidate route for the same Qwen3.8 27B model family but a distinct serving stack: NVFP4 NInfer weights served over loopback HTTP with server-side chat templating, versus the cached Ollama UD Q4_K_XL GGUF quantization with its uncaptured sampling defaults. The contemporaneous Ollama Qwen3.8 control and the two NInfer Qwen3.6 profiles remain pending, so no contemporaneous cross-route timing conclusion is drawn here. The run used explicit decode parameters (temperature 0, top_p 1, max_tokens 4,096, n=1) as a documented greedy approximation; the first request's `length` finish reason means at least one response may be truncated at the token cap.

## Artifacts

- Successful run JSON: `models/experiments/vlm/data/qwen38_ninfer_room0_40_20261117.json` (Git-ignored project data, not copied to tracked outputs), 936,550 bytes, SHA-256 `22e6ab437d1598c98a41a5435e06fb37bb99f888cd1d2dde80dd5587a8d85d1b`
- Successful run log: [`data/qwen38_ninfer_room0_40_20261117.run.log`](../../data/qwen38_ninfer_room0_40_20261117.run.log), 6,568 bytes, SHA-256 `5b8d885e08a57f27e8130e562896fd1088b8558d36aa1c5f59f238455906366f`
- Runner SHA-256 (independent review passed): `20860b394eb1c281fcbd4b141f036563f943810ff6a079afc5d7d2d0f784548f`
- Review artifact: `review/ADJUDICATION.md` in the implementation workspace (`.agentic_workflow/workspace/pipeline-experiments/vlm-40frame-routes-20261116/qwen38-ninfer-implementation-20261117/review/ADJUDICATION.md`)
- NInfer profile manifest (root-captured before the run): `NINFER-PROFILE-MANIFEST.json` in the implementation workspace

## Limitations

- The NInfer server applied its own chat template and image preprocessing; request serialization differs from the Transformers routes and token counts are server-reported. Timing and output differences are model-plus-route observations, not weights-only comparisons.
- Relation agreement is null because no output passed the strict schema gate; this is unscorability, not zero semantic accuracy, and the 25/40 legacy parses are not an agreement score.
- Caption word F1 is lexical agreement to cached Qwen3.8 model outputs (the provisional baseline), not ground-truth accuracy.
- Process-scope VRAM is unavailable (the NInfer process is owned by the model-control lifecycle); the 29,720 MiB peak is whole-device `nvidia-smi` and includes the desktop and other processes. The controller does not expose a load duration.
- The 4,096-token cap was reached on the first relation request (finish reason `length`); that response may not represent typical response length.
- Persistent raw results remain under `models/experiments/vlm/data/` in the project HDD (Git-ignored); nothing was written to `/tmp` or `/mnt/HDD1/tmp`, and the raw JSON is not included in tracked outputs.
