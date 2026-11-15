# Modernized pipeline component results

Measured on 2026-10-30 with the repository `.venv` and one GPU model at a time.
These are component-level results; the detector and encoder metrics use
different datasets and should not be compared across sections.

## Feature encoders

The full COCO val object bank contained 36,335 non-crowd objects across 80
categories. Each model encoded the same ground-truth padded boxes and
mask-isolated crops. The score is category-macro object-query retrieval mAP.

| Model | Padded-box mAP | Masked-object mAP | Padded objects/s | Peak CUDA allocation |
|---|---:|---:|---:|---:|
| SigLIP2 So400m | 0.6417 | **0.6833** | 182.9 | 2.47 GiB |
| FG-CLIP2 So400m | **0.7399** | 0.6569 | 231.1 | 8.63 GiB |
| SigLIP2 Base | 0.6978 | 0.6044 | **1,531.4** | **0.94 GiB** |
| CLIP ViT-H/14 | 0.6167 | 0.2336 | 367.8 | 2.12 GiB |

FG-CLIP2 So400m also reached 0.8146 mAP with its native full-image RoI feature
path. The modernized pipeline currently stores padded-proposal embeddings, so
the directly matched score is 0.7399. The results support SigLIP2 So400m for
mask-isolated objects and FG-CLIP2 So400m for padded proposals.

Raw results: [modernized_clip_component.json](../.agentic_workflow/workspace/modernized_clip_component.json).

## Detector

YOLOE-26-L-seg and YOLOE-26-M-seg were evaluated on the same seeded 500-image
COCO val2017 sample, containing 3,206 non-crowd instances across 79 represented
categories. Both used the same 80 text prompts, 640-pixel inference size, and
0.001 confidence floor. The mask score below is a provisional custom polygon
AP@0.5 using greedy matching and 101-point interpolation, not standard COCO
mask AP. Ultralytics merges disconnected predicted contours in its polygon
export, and OpenCV rasterization differs from COCO mask decoding. The small
L/M mask-score gap may be affected by those differences.

| Model | Mask AP@0.5 | Box AP@0.5 | Mask recall | Images/s | Median latency | P95 latency | Peak CUDA allocation |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOE-26-L-seg | **0.5881** | **0.5972** | 0.7982 | 79.4 | 10.67 ms | 14.96 ms | 1.61 GiB |
| YOLOE-26-M-seg | 0.5805 | 0.5931 | **0.8013** | **100.6** | **9.20 ms** | **11.63 ms** | 1.57 GiB |

The provisional custom mask score favors L by 0.0075, while measured mask
recall favors M by 0.0031. M is 26.6% faster here. Keep YOLOE-26-L-seg as the
requested implementation default, but do not treat this run as a reliable mask
quality ranking. Re-evaluate with native predicted masks and COCO-decoded ground
truth masks before drawing a segmentation-quality conclusion.

Raw results: [modernized_yoloe_seg_component.json](../.agentic_workflow/workspace/modernized_yoloe_seg_component.json).

## End-to-end run

The full modernized model stack completed on 40 sampled Replica room0 frames
at stride 50 using the repository `.venv` and CUDA. YOLOE-26-L-seg produced
the detections and SigLIP2 So400m plus FG-CLIP2 So400m produced per-object
features. Qwen annotated all 40 frames. Headless RGB-D mapping and CUDA Faiss
then produced a 626-object map, a point-cloud pickle, object JSON, and edge
JSON. The single command was:

```bash
scripts/run_modernized_pipeline_local.sh --scene room0 --stride 50 --resume-from-caches
```

Caption consolidation returned non-empty results for 505 of 509 objects with
source captions. Four Qwen responses returned malformed JSON and remained
blank after a retry attempt stalled. The remaining 117 objects had no usable
VLM caption to consolidate. The retry was interrupted during an atomic map
write; the prior complete map remained readable and its object JSON retained
all 626 objects. Relation extraction emitted 199 2D relations across 24
frames, but mapping retained zero 3D relations in the final edge JSON. The
object-level pipeline ran through its final stage, but relation propagation
remains unresolved. This run covers a stride-50 sample, not every frame in
room0. VLM alternatives and map-level detector effects remain unscored.

## Caption consolidation reliability and model pilot

The four blank results are covered by a real-GPU acceptance test. The
consolidator now accepts JSON captions and plain-text responses, retries empty
or malformed structured output once, and leaves unusable output blank. Map
checkpoint rewrites now occur every 50 visited objects instead of every five.
A GPU diagnostic found that Qwen spent all 256 completion tokens on hidden
reasoning when reasoning was set to `low`. Setting reasoning to `none` returned
a complete JSON caption in 30 tokens. The updated real-GPU acceptance then
produced nonempty captions for all four failed groups and saved the Qwen3.8
baseline at
`/mnt/HDD1/Project_Data/public_datasets/Replica/room0/exps/r_modernized_yoloe26l_stride50/qwen_caption_baseline_failed_groups.json`.

Three smaller models were compared on those same four caption groups. Each ran
alone on the RTX 5090 through a CUDA llama.cpp server built under `/tmp`; all
model layers and multimodal projectors were loaded to GPU. The model server
used reasoning mode `off` to match the Qwen baseline's no-reasoning setting.
Each timing is the mean across four text requests after model load. Word F1
measures overlap with Qwen3.8 and does not measure correctness.

| Candidate | Valid captions | Mean request time | Mean word F1 to Qwen3.8 |
|---|---:|---:|---:|
| Qwen3.5 9B Q4 | 4/4 | **0.251 s** | 0.741 |
| Qwen3.5 9B Q8 | 4/4 | 0.361 s | 0.769 |
| Gemma 4 12B Q4 | 4/4 | 0.318 s | **0.903** |

The first Qwen3.5 Q4 attempt used llama.cpp's default reasoning mode and
returned empty content in all four groups because the 256-token cap was spent
on hidden reasoning. Re-running the same candidate with reasoning disabled
returned four valid captions. This is a configuration failure, not an accuracy
result. Future consolidation runs now disable reasoning in both the configured
Qwen client and the edge-server launcher.

The pilot does not establish a better replacement model. Caption groups 278 and
348 mix descriptions of different objects (`desk` with `pillow`, and `cup` with
`magazine`), despite their stored classes being `desk` and `cup`. All candidates
either preserve or reproduce this association noise. Group 419 also contains
conflicting sofa and chair locations. These input inconsistencies limit the
quality comparison. Gemma 4 12B had the highest agreement with the Qwen baseline,
but the baseline itself makes inconsistent choices on these groups. Retain
Qwen3.8 as the default; use these outputs to motivate a separate track-caption
consistency fix before broader quality claims.

Raw reports: [Qwen3.5 9B Q4](../.agentic_workflow/workspace/caption-consolidation-benchmark/room0-qwen35-9b-q4-caption-consolidation-no-reasoning.json),
[Qwen3.5 9B Q8](../.agentic_workflow/workspace/caption-consolidation-benchmark/room0-qwen35-9b-q8-caption-consolidation.json),
[Gemma 4 12B](../.agentic_workflow/workspace/caption-consolidation-benchmark/room0-gemma4-12b-caption-consolidation.json).

## Expanded 40-track comparison

The broader paired sample is staged at
`.agentic_workflow/workspace/caption-consolidation-benchmark/room0-40-case-sample.json`.
It includes every multi-caption track in the map, plus 35 single-caption tracks
spaced across the map order. The saved Qwen3.8 baseline combines the original
40-frame pipeline outputs with the four repaired results. Since only five of
the 40 tracks contain multiple source captions, the other 35 cases test
single-caption output behavior rather than merging. All three candidates ran
on CUDA with reasoning disabled and produced valid outputs for all 40 cases.
Latency is the mean across candidate requests after model loading. The Qwen3.8
outputs are cached pipeline results, so their response time was not measured in
this comparison. Word F1 and exact match measure agreement with Qwen3.8, not
ground-truth accuracy.

| Candidate | Valid | Mean request time | Word F1, all 40 | Exact match, all 40 | Word F1, 5 multi-caption | Exact match, 5 multi-caption |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5 9B Q4 | 40/40 | **0.201 s** | 0.946 | 50.0% | 0.730 | 0/5 |
| Qwen3.5 9B Q8 | 40/40 | 0.300 s | 0.940 | 42.5% | 0.754 | 0/5 |
| Gemma 4 12B Q4 | 40/40 | 0.302 s | **0.966** | **60.0%** | **0.893** | 1/5 |

The all-case scores are inflated by the 35 singleton groups. On the five actual
multi-caption groups, Gemma 4 had the highest lexical agreement with Qwen3.8.
The sample remains too small and internally inconsistent to establish a
semantic winner. Group 278 is stored as a desk but its two source captions
describe a desk and a pillow. Group 348 pairs a cup with a magazine. Group 419
has conflicting sofa and chair locations. These cases point to caption-to-track
association as a limit on consolidation quality.

The pilot does not justify replacing Qwen3.8. Qwen3.5 Q4 was fastest; Gemma 4
was most similar to the Qwen3.8 outputs. Keep Qwen3.8 as the default until a
larger, cleaner set with independent quality labels is available.

## Modernized pipeline consolidator selection

The modernized pipeline now uses Gemma 4 12B QAT Q4 for final caption
consolidation by default, following the user's model selection. Qwen3.8 remains
the default frame annotation VLM. Gemma starts only after mapping finishes and
is stopped after consolidation, so it does not share GPU memory with YOLOE or
the annotation VLM. This is a deployment choice, not a claim that Gemma has
higher caption accuracy. Select `--consolidation-model qwen3.8` to use the
previous Ollama consolidator. The pipeline run requires a CUDA llama.cpp build;
the Gemma server log is saved beside the map output as `gemma4-server.log`.

Raw reports: [40-case sample](../.agentic_workflow/workspace/caption-consolidation-benchmark/room0-40-case-sample.json),
[Qwen3.8 baseline](../.agentic_workflow/workspace/caption-consolidation-benchmark/room0-qwen3.8-baseline-40.json),
[Qwen3.5 9B Q4](../.agentic_workflow/workspace/caption-consolidation-benchmark/room0-qwen35-9b-q4-caption-consolidation-40.json),
[Qwen3.5 9B Q8](../.agentic_workflow/workspace/caption-consolidation-benchmark/room0-qwen35-9b-q8-caption-consolidation-40.json),
[Gemma 4 12B](../.agentic_workflow/workspace/caption-consolidation-benchmark/room0-gemma4-12b-caption-consolidation-40.json).
