# Detector benchmark

**Evidence date:** 2026-10-30. This report summarizes the existing component benchmark. No benchmark was run for this task.

## Setup

YOLOE-26-L-seg and YOLOE-26-M-seg were measured on the same seeded 500-image COCO val2017 sample (`sample_seed=42`). The sample contained 3,206 non-crowd instances across 79 represented categories. Both used the same 80 text prompts, 640-pixel inference size, and 0.001 confidence floor. The reported timing and CUDA allocation came from the existing run record.

The mask score is the benchmark's provisional custom polygon AP@0.5: greedy one-to-one matching with 101-point interpolation. It is not standard COCO mask AP. Ultralytics contour export merges disconnected components, and OpenCV rasterization differs from COCO mask decoding.

## Results

| Model | Mask AP@0.5* | Box AP@0.5 | Mask recall | Box recall | Images/s | Median / P95 latency | Peak CUDA allocation |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOE-26-L-seg | **0.5881** | **0.5972** | 0.7982 | 0.8453 | 79.4 | 10.67 / 14.96 ms | 1.61 GiB |
| YOLOE-26-M-seg | 0.5805 | 0.5931 | **0.8013** | 0.8453 | **100.6** | **9.20 / 11.63 ms** | 1.57 GiB |

\* Provisional custom polygon metric, not native COCO mask AP.

## Interpretation

L led the custom mask AP by 0.0075. M led mask recall by 0.0031 and ran 26.6% faster in this sample. Keep YOLOE-26-L-seg as the configured detector choice, but treat this as the recorded configuration decision rather than a reliable mask-quality ranking. Re-evaluation with native-resolution predicted masks and COCO-decoded ground truth is still optional validation.

## Provenance and limitations

- Raw results: [`modernized_yoloe_seg_component.json`](data/modernized-yoloe-seg-component.json)
- Benchmark methodology: [`MODERNIZED_COMPONENT_BENCHMARKS.md`](../MODERNIZED_COMPONENT_BENCHMARKS.md)
- Results summary: [`MODERNIZED_COMPONENT_RESULTS.md`](../MODERNIZED_COMPONENT_RESULTS.md)

The benchmark is COCO detection quality, not Replica map association or downstream graph recovery. The custom mask metric is provisional and no confidence intervals are recorded.
