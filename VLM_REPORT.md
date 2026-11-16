# VLM caption consolidation benchmark

**Date:** 2026-10-30  
**Scene:** Replica `room0`  
**Hardware:** RTX 5090, CUDA; one model loaded at a time

## Purpose

This benchmark compares three edge-compatible vision-language models against
the fixed Qwen3.8 foundation-model outputs used by the modernized pipeline.
It evaluates the caption consolidation stage only. It does not measure
end-to-end scene graph quality or compare the detector and CLIP components.

## Method

The paired sample contains 40 object tracks from the 40-frame `room0` pipeline
run. It includes every multi-caption track and 35 single-caption tracks spaced
across map order. Thus, only five cases exercise consolidation of multiple
captions; the other 35 primarily measure single-caption output behavior.

Each candidate ran on CUDA through a llama.cpp server with reasoning disabled.
All 40 requests returned valid captions. Candidate latency is the mean request
time after model loading. Qwen3.8 outputs were cached from the pipeline run,
including four repaired outputs, so Qwen3.8 latency was not measured in this
comparison. Word F1 and exact match measure lexical agreement with Qwen3.8,
not correctness against independent ground truth.

## Results

| Model | Valid | Mean request time | Word F1, all 40 | Exact match, all 40 | Word F1, 5 multi-caption | Exact match, 5 multi-caption |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5 9B Q4 | 40/40 | **0.201 s** | 0.946 | 50.0% | 0.730 | 0/5 |
| Qwen3.5 9B Q8 | 40/40 | 0.300 s | 0.940 | 42.5% | 0.754 | 0/5 |
| Gemma 4 12B Q4 | 40/40 | 0.302 s | **0.966** | **60.0%** | **0.893** | 1/5 |

## Findings

Gemma 4 12B Q4 had the highest lexical agreement with Qwen3.8 on the five
multi-caption tracks. Qwen3.5 9B Q4 was the fastest candidate. These results do
not establish that either model produces more accurate scene descriptions:
Qwen3.8 is the comparison baseline, not independently verified ground truth.

The all-40 scores are dominated by the 35 single-caption tracks. The
multi-caption subset is more relevant to consolidation but contains only five
tracks. Its inputs also expose caption-to-track association noise: track 278 is
stored as a desk but has desk and pillow descriptions; track 348 combines cup
and magazine descriptions; and track 419 has conflicting sofa and chair
locations. These inconsistencies limit the quality comparison.

**Recommendation:** Keep Qwen3.8 as the default consolidator. The benchmark
shows that the smaller models run successfully and provides latency and
agreement measurements, but it does not support a model replacement. Improve
caption-to-track association and build a larger set with independent quality
labels before making a semantic quality claim.

## Raw data

- [40-track sample](.agentic_workflow/workspace/caption-consolidation-benchmark/room0-40-case-sample.json)
- [Qwen3.8 baseline](.agentic_workflow/workspace/caption-consolidation-benchmark/room0-qwen3.8-baseline-40.json)
- [Qwen3.5 9B Q4 outputs](.agentic_workflow/workspace/caption-consolidation-benchmark/room0-qwen35-9b-q4-caption-consolidation-40.json)
- [Qwen3.5 9B Q8 outputs](.agentic_workflow/workspace/caption-consolidation-benchmark/room0-qwen35-9b-q8-caption-consolidation-40.json)
- [Gemma 4 12B outputs](.agentic_workflow/workspace/caption-consolidation-benchmark/room0-gemma4-12b-caption-consolidation-40.json)
- [Detailed component results](scripts/MODERNIZED_COMPONENT_RESULTS.md)
