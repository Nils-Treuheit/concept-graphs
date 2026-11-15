# Feature encoder benchmark

**Evidence date:** 2026-10-30. This report summarizes the existing component benchmark. No benchmark was run for this task.

## Setup

The benchmark used all 4,952 images and 36,335 non-crowd instances from COCO 2017 validation across 80 categories. Each encoder received the same ground-truth padded boxes and instance masks. Retrieval queries used `a photo of a {category}`. The encoder ran on CUDA 0 with batch size 16, text batch size 128, 20 pixels of crop padding, and one timing pass. The masked-object crop replaced pixels outside each annotation with the model-specific processor mean.

The primary metric is category-macro object-query retrieval mAP over the complete object bank. These are oracle-proposal component measurements. They do not measure detector proposals, temporal association, or full 3D mapping.

## Results

| Model | Model ID | Padded-box mAP | Masked-object mAP | Padded objects/s | Peak CUDA allocation |
|---|---|---:|---:|---:|---:|
| SigLIP2 So400m | `google/siglip2-so400m-patch14-384` | 0.6417 | **0.6833** | 182.9 | 2.47 GiB |
| FG-CLIP2 So400m | `qihoo360/fg-clip2-so400m` | **0.7399** | 0.6569 | 231.1 | 8.63 GiB |
| SigLIP2 Base | `google/siglip2-base-patch16-224` | 0.6978 | 0.6044 | **1,531.4** | **0.94 GiB** |
| CLIP ViT-H/14 | `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` | 0.6167 | 0.2336 | 367.8 | 2.12 GiB |

FG-CLIP2 So400m also scored 0.8146 mAP using its native full-image RoI feature path. That model-specific condition is not the padded-crop comparison used by the deployed proposal feature path.

## Interpretation

The measured defaults are SigLIP2 So400m for mask-isolated mapping features and FG-CLIP2 So400m for padded proposals. SigLIP2 Base is substantially faster and lower-memory, but its retrieval mAP was lower in both crop conditions. The current evidence supports these feature-slot choices on COCO retrieval; map-level association quality remains an open validation item.

## Provenance and limitations

- Raw results: [`modernized_clip_component.json`](../data/modernized-clip-component.json)
- Benchmark methodology: [`MODERNIZED_COMPONENT_BENCHMARKS.md`](../MODERNIZED_COMPONENT_BENCHMARKS.md)
- Results summary: [`MODERNIZED_COMPONENT_RESULTS.md`](../MODERNIZED_COMPONENT_RESULTS.md)

The benchmark used one timing pass and reports no confidence intervals. COCO categories and imagery differ from Replica and user-recorded scenes. The raw result records per-process PyTorch CUDA allocation, which excludes memory held by concurrent processes.
