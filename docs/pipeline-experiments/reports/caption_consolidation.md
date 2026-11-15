# Caption consolidation benchmark

**Evidence date:** 2026-10-30. This report summarizes the existing real-GPU pilot. No benchmark was run for this task.

Caption consolidation follows mapping and is distinct from per-frame VLM annotation. The pilot compared candidate outputs with cached Qwen3.8 outputs. Word F1 and exact match therefore measure agreement with Qwen3.8, not independent semantic correctness.

## Four-group failure-focused pilot

The first sample contained the four room0 object groups whose Qwen response was blank or malformed. The accepted Qwen3.8 baseline was regenerated with reasoning disabled. Candidates used a 256-token cap and an edge-compatible CUDA llama.cpp server on the RTX 5090, one model at a time. Candidate request times are means after model load.

| Candidate | Reasoning mode | Valid outputs | Mean request time | Mean word F1 to Qwen3.8 |
|---|---|---:|---:|---:|
| Qwen3.5 9B UD-Q4_K_XL, first attempt | Default server mode | 0/4 | 2.7905 s | 0.000 |
| Qwen3.5 9B UD-Q4_K_XL | Off | 4/4 | **0.2513 s** | 0.741 |
| Qwen3.5 9B UD-Q8_K_XL | Off | 4/4 | 0.3605 s | 0.769 |
| Gemma 4 12B QAT UD-Q4_K_XL | Off | 4/4 | 0.3178 s | **0.903** |

The failed Qwen3.5 Q4 configuration spent its 256-token response budget on hidden reasoning. Disabling reasoning fixed output validity; this is a configuration failure, not an accuracy result.

## Paired 40-track sample

The fixed sample included all five multi-caption tracks and 35 single-caption tracks from the 40-frame room0 map. All three candidates returned valid outputs for all 40 cases. Timings are mean request times after loading; Qwen3.8 baseline outputs were cached, so its timing was not measured.

| Candidate | Valid | Mean request time | Word F1, all 40 | Exact match, all 40 | Word F1, 5 multi-caption | Exact match, 5 multi-caption |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5 9B UD-Q4_K_XL | 40/40 | **0.2009 s** | 0.946 | 50.0% | 0.730 | 0/5 |
| Qwen3.5 9B UD-Q8_K_XL | 40/40 | 0.3002 s | 0.940 | 42.5% | 0.754 | 0/5 |
| Gemma 4 12B QAT UD-Q4_K_XL | 40/40 | 0.3020 s | **0.966** | **60.0%** | **0.893** | 1/5 |

The all-case scores are dominated by 35 singleton cases. On the five true multi-caption groups, Gemma had the highest lexical agreement with the baseline. The sample contained known track-caption inconsistencies: group 278 combined desk and pillow captions, group 348 paired a cup with a magazine, and group 419 mixed sofa and chair locations. This limits semantic conclusions.

## Recorded deployment choice

Gemma 4 12B QAT Q4 is configured as the modernized pipeline's final caption consolidator. This is a deployment choice, not a finding that Gemma is more accurate. Qwen3.8 remains the frame annotation baseline. The available pilot does not establish a semantic replacement winner because the comparison is small, reference outputs are from Qwen3.8, and some source tracks are inconsistent.

## Provenance

- Four-group candidate JSONs: [`qwen3.5 Q4, reasoning off`](../data/room0-qwen35-9b-q4-caption-consolidation-no-reasoning.json), [`qwen3.5 Q8`](../data/room0-qwen35-9b-q8-caption-consolidation.json), [`Gemma 4 12B`](../data/room0-gemma4-12b-caption-consolidation.json)
- 40-track candidate JSONs: [`Qwen3.5 Q4`](../data/room0-qwen35-9b-q4-caption-consolidation-40.json), [`Qwen3.5 Q8`](../data/room0-qwen35-9b-q8-caption-consolidation-40.json), [`Gemma 4 12B`](../data/room0-gemma4-12b-caption-consolidation-40.json)
- Baseline and sample: [`Qwen3.8 baseline`](../data/room0-qwen3.8-baseline-40.json), [`40-track manifest`](../data/room0-40-case-sample.json)
- Procedure and full context: [`MODERNIZED_COMPONENT_BENCHMARKS.md`](../MODERNIZED_COMPONENT_BENCHMARKS.md), [`MODERNIZED_COMPONENT_RESULTS.md`](../MODERNIZED_COMPONENT_RESULTS.md)

## Limits

The benchmark JSON leaves manual faithfulness and unsupported-claim review fields unset. No independent reference answers were scored. The 40-case Qwen3.8 responses are cached and have no measured candidate latency. The Q4/Q8 model size estimates are about 6.9 GB and 13.9 GB respectively, including projector; Gemma is about 6.9 GB. These are recorded weight-plus-projector estimates, not measured peak VRAM.
