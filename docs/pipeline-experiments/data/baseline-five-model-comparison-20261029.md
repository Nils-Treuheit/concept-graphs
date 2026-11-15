# CLIP Component Benchmark: Six-Model Comparison

**Run date:** 2026-10-29
**Models:** OpenAI CLIP ViT-H/14, SigLIP2 Base, SigLIP2 So400m, FG-CLIP2 Base, FG-CLIP2 So400m, MetaCLIP 2 Worldwide L/14.

## Executive summary

- **The scene-graph recommendation depends on input format.** For padded proposals, FG-CLIP2 So400m is the strongest balanced choice: it ties for top graph edge recall (0.0395) and is close to SigLIP2 So400m on cross-view identity (0.2415 vs 0.2484). For mask-isolated objects, SigLIP2 Base is strongest: cross-view accuracy 0.4224 and edge recall 0.3026, tied with SigLIP2 So400m.
- **Mask isolation improved Replica cross-view identity and graph recovery for every model.** The best shared-condition result was SigLIP2 Base: cross-view top-1 rose from 0.2446 with padded boxes to 0.4224 with isolated masks; oracle-predicate triplet recall rose from 0.0263 to 0.3026.
- **FG-CLIP2 So400m led COCO object retrieval on padded boxes (0.7264 mAP)** and its model-native full-image RoIAlign condition scored 0.8138. It also tied for best padded Replica graph edge recall and was close to the cross-view leader. It used substantially more peak memory (7.25 GiB allocated, 7.70 GiB reserved on COCO).
- **SigLIP2 So400m led COCO caption-to-label MRR** for both shared inputs (0.3216 padded, 0.1563 isolated), but its COCO object encoding was slower than CLIP on padded boxes in this single run.
- Treat latency as preliminary: each measurement used one timed pass, and Replica ran at batch 16 while COCO ran at batch 2. Timing includes image and relevant text encoding, not model loading or post-encoding scoring.

## Benchmark design

| Evaluation | Ground truth and scope | Inputs | Main measures |
|---|---|---|---|
| COCO object-label retrieval | 4,952 validation images, 36,335 human-annotated objects, 80 categories; labels, boxes, and masks are ground truth. | Padded ground-truth box; mask-isolated cutout composited using the model processor RGB mean. | Macro category AP for text category queries ranking the complete object bank. |
| COCO caption-to-label matching | COCO image captions are matched to category names/aliases mentioned in each caption and present in that image. This is weak instance grounding because captions describe whole images. | Same padded-box and mask-isolated object bank. | Macro-category MRR and recall@1/@5. |
| Replica object retrieval and scene-graph impact | Eight ReplicaSSG scenes; 50 sampled frames per scene, 400 frames total, 3,545 oracle object observations, 53 categories. Object masks are raycast from semantic mesh IDs at the RGB camera poses. | Padded ground-truth region; mean-background mask-isolated cutout. | Category retrieval mAP; cross-view instance top-1; identity-driven graph node accuracy and predicate-triplet recall. |

The COCO caption test used 24,774 captions from the used images; 20,230 mention at least one ground-truth category, yielding 28,163 positive caption-category pairs. ReplicaSSG does not provide reference object captions. The scene-graph measure holds annotated predicates fixed and measures errors propagated from object identity association; it does not evaluate predicate generation. Of 3,545 observations, 3,499 contributed to scored graph node identity; 76 annotated edge observations contributed to triplet recall.

All masked inputs use each model processor’s configured image mean, rounded to 8-bit RGB, as the outside-mask background. FG-CLIP2 additionally uses the model-native full-image RoIAlign region path as an exploratory condition. Its `pixel_attention_mask` is a processor validity mask, not a semantic object mask; no GT mask was passed as a semantic attention filter.

## COCO results

### Accuracy

| Model | Object mAP, padded | Object mAP, isolated | Caption MRR, padded | Caption MRR, isolated | Caption R@1, padded / isolated | Caption R@5, padded / isolated |
|---|---:|---:|---:|---:|---:|---:|
| clip | 0.6168 | 0.2336 | 0.2667 | 0.1069 | 0.1672 / 0.0665 | 0.3757 / 0.1396 |
| siglip2-base | 0.6977 | 0.6044 | 0.2815 | 0.1279 | 0.1800 / 0.0814 | 0.3922 / 0.1683 |
| siglip2-so400m | 0.6417 | 0.6833 | 0.3216 | 0.1563 | 0.2081 / 0.1021 | 0.4515 / 0.2049 |
| fg-clip2-base | 0.6843 | 0.5754 | 0.2739 | 0.1165 | 0.1741 / 0.0752 | 0.3828 / 0.1527 |
| fg-clip2-so400m | 0.7264 | 0.6343 | 0.2761 | 0.1258 | 0.1770 / 0.0811 | 0.3839 / 0.1643 |
| metaclip2 | 0.6757 | 0.6086 | 0.3000 | 0.1407 | 0.1936 / 0.0883 | 0.4144 / 0.1856 |

### FG-CLIP2 model-native RoIAlign on COCO

This additional condition is FG-CLIP2-specific and should not be compared directly with the shared crop inputs.

| Model | Object mAP | Caption MRR | Caption R@1 | Caption R@5 | Object inference s | Caption inference s |
|---|---:|---:|---:|---:|---:|---:|
| fg-clip2-base | 0.7368 | 0.2623 | 0.1681 | 0.3632 | 20.2 | 22.3 |
| fg-clip2-so400m | 0.8138 | 0.2413 | 0.1505 | 0.3376 | 78.9 | 86.2 |

### Mean-background RGB used

| Model family | RGB fill |
|---|---:|
| CLIP and MetaCLIP 2 | `[123, 117, 104]` |
| SigLIP2 and FG-CLIP2 | `[128, 128, 128]` |

### Inference time and VRAM

Inference times below include image encoding plus category-text encoding for object-label retrieval, or caption-text encoding for caption matching. `Δ` is the absolute seconds difference from CLIP for the same condition and task. VRAM is per-process PyTorch peak allocated/reserved memory and excludes concurrent processes.

| Model | Object retrieval padded s (Δ) | Object retrieval isolated s (Δ) | Caption matching padded s (Δ) | Caption matching isolated s (Δ) | Peak allocated GiB (Δ vs CLIP) | Peak reserved GiB |
|---|---:|---:|---:|---:|---:|---:|
| clip | 196.4 (+0.0) | 197.4 (+0.0) | 206.0 (+0.0) | 206.9 (+0.0) | 2.12 (+0.00) | 2.46 |
| siglip2-base | 72.1 (-124.3) | 71.7 (-125.7) | 75.4 (-130.6) | 75.0 (-131.9) | 0.94 (-1.18) | 1.06 |
| siglip2-so400m | 358.8 (+162.3) | 260.9 (+63.5) | 368.9 (+162.9) | 271.0 (+64.1) | 2.45 (+0.33) | 2.62 |
| fg-clip2-base | 71.7 (-124.8) | 69.5 (-127.9) | 74.0 (-132.0) | 71.8 (-135.1) | 2.38 (+0.26) | 2.80 |
| fg-clip2-so400m | 136.7 (-59.7) | 164.0 (-33.4) | 143.6 (-62.4) | 170.8 (-36.1) | 7.25 (+5.13) | 7.70 |
| metaclip2 | 110.2 (-86.3) | 117.9 (-79.5) | 113.6 (-92.4) | 121.3 (-85.7) | 2.25 (+0.13) | 2.37 |

Model-load time is separate and is recorded in the raw JSON. It is sensitive to cache and filesystem state, so it is not used for the recommendation.

## Replica results

### Object retrieval and scene-graph impact

`Graph edge recall` is the fraction of annotated subject-predicate-object edge observations recovered when predicate labels are held fixed and subject/object identities come from visual cross-view matching.

| Model | Object mAP padded | Object mAP isolated | Cross-view top-1 padded | Cross-view top-1 isolated | Graph edge recall padded | Graph edge recall isolated |
|---|---:|---:|---:|---:|---:|---:|
| clip | 0.0228 | 0.0244 | 0.2109 | 0.3444 | 0.0395 | 0.1053 |
| siglip2-base | 0.0212 | 0.0338 | 0.2446 | 0.4224 | 0.0263 | 0.3026 |
| siglip2-so400m | 0.0209 | 0.0335 | 0.2484 | 0.4015 | 0.0263 | 0.3026 |
| fg-clip2-base | 0.0239 | 0.0391 | 0.2301 | 0.3750 | 0.0132 | 0.1316 |
| fg-clip2-so400m | 0.0230 | 0.0402 | 0.2415 | 0.3895 | 0.0395 | 0.2500 |
| metaclip2 | 0.0227 | 0.0298 | 0.2366 | 0.3747 | 0.0395 | 0.2105 |

### Replica inference time and VRAM

Replica label retrieval and scene-graph identity evaluation share the same image and category-text embeddings, so the same encoder time applies to both; the table reports that shared inference time. `Δ` is relative to CLIP for the same input condition.

| Model | Padded s (Δ vs CLIP) | Isolated s (Δ vs CLIP) | Peak allocated GiB (Δ vs CLIP) | Peak reserved GiB |
|---|---:|---:|---:|---:|
| clip | 96.8 (+0.0) | 44.4 (+0.0) | 1.99 (+0.00) | 2.25 |
| siglip2-base | 13.4 (-83.4) | 15.3 (-29.1) | 0.80 (-1.19) | 0.83 |
| siglip2-so400m | 34.6 (-62.2) | 37.1 (-7.3) | 2.47 (+0.48) | 2.65 |
| fg-clip2-base | 29.0 (-67.8) | 31.8 (-12.6) | 2.95 (+0.96) | 3.31 |
| fg-clip2-so400m | 71.0 (-25.8) | 81.4 (+37.0) | 6.94 (+4.95) | 7.88 |
| metaclip2 | 121.2 (+24.4) | 86.9 (+42.5) | 2.15 (+0.16) | 2.22 |

### FG-CLIP2 native RoIAlign condition

This model-specific condition uses full-frame image features and RoIAlign with the ground-truth boxes, plus FG-CLIP2 box-text features. It is not directly comparable to the shared crop conditions because other models do not expose the same region-text path.

| Model | Object mAP | Cross-view top-1 | Graph edge recall | Shared encoder seconds |
|---|---:|---:|---:|---:|
| fg-clip2-base | 0.0206 | 0.1589 | 0.0000 | 14.0 |
| fg-clip2-so400m | 0.0217 | 0.1683 | 0.0263 | 12.2 |

## Interpretation

1. **Scene-graph input recommendations:** For padded proposals, FG-CLIP2 So400m is a strong balanced choice: it ties top graph edge recall and trails SigLIP2 So400m by only 0.0069 on cross-view top-1, while beating CLIP and MetaCLIP2 on cross-view top-1. For mask-isolated objects, SigLIP2 Base is strongest, leading cross-view top-1 and tying the top edge-recall score. It is also much lighter than FG-CLIP2 So400m.
2. **Query matching:** SigLIP2 So400m led COCO caption MRR. FG-CLIP2 So400m led shared padded-box category retrieval and native RoIAlign retrieval. Caption scores remain weakly grounded image-level matches, so they do not establish a winner for arbitrary user descriptions.
3. **FG-CLIP2:** FG-CLIP2 So400m is strong on COCO padded-box retrieval and native RoIAlign retrieval, but its Replica masked graph result was below both SigLIP2 variants. It also had the highest memory footprint and slower isolated inference than SigLIP2 Base. The specialized native RoIAlign path is promising but remains a model-specific comparison.
4. **Input choice:** Mean-background mask isolation consistently helped Replica cross-view association and graph recovery across all models. COCO mask-isolated category AP fell for CLIP and several alternatives, but rose substantially for SigLIP2 So400m versus its padded-box result. The dataset and task matter; the results do not support one universal crop mode.
5. **MetaCLIP 2:** It improved COCO caption MRR over CLIP but did not lead object mAP or Replica association/graph recovery.

## Limitations

- One timed pass was collected per model and condition. No variance or confidence intervals are available.
- COCO used batch size 2; Replica used batch size 16. Timing comparisons are within each benchmark only.
- The GPU was shared with other resident workloads. The reported VRAM fields describe this process’s PyTorch allocation and omit memory held by other processes.
- COCO captions are weakly grounded to object instances through category-name and alias matching. Replica has no reference captions.
- Replica scene-graph evaluation covers eight scenes and 76 annotated edge observations. Predicate generation is not tested.
- FG-CLIP2 native RoIAlign results use a distinct architecture pathway and cannot be compared to CLIP without a corresponding region-feature implementation.
- Model loading and disk caching differ between runs; inference tables exclude model-loading time.

## Artifacts

- COCO raw results: `coco-val-five-models-20261029-rerun.json`
- Replica raw results: `replica-five-models-20261029.json`
- COCO benchmark implementation: `conceptgraph/scripts/clip_component.py`
- Replica benchmark implementation: `conceptgraph/scripts/replica_clip_benchmark.py`
