# DINOv3 ViT-L/16: ReplicaSSG held-out visual association

**Status:** corrected FP32 run completed on CUDA 0. This is a visual-only association evaluation. It does not evaluate DINOde, text-query retrieval, or VQA.

## Configuration and data

Model: `facebook/dinov3-vitl16-pretrain-lvd1689m`, cached snapshot `ea8dc2863c51be0a264bab82070e3e8836b02d51`. The runner used final-layer `pooler_output` with CLS fallback, L2 normalization, FP32 weights and inputs, 224×224 bicubic resizing, ImageNet mean `[0.485,0.456,0.406]`, and standard deviation `[0.229,0.224,0.225]`. Local processor metadata was absent, so preprocessing details require confirmation against the upstream model card.

The experiment matched the five locally available ReplicaSSG held-out scans: `office_1`, `office_3`, `office_4`, `room_1`, `room_2`. Six of the eleven test scans were excluded for missing local RGB or pose data. It sampled 50 frames per scene, 250 frames total, with 2,204 GT object views across 49 categories. Ground-truth masks were raycast from semantic mesh IDs and poses. The five scenes contain 91 unique annotated relationship tuples. The graph metric scores 49 relation observations because both endpoints must be visible in a query frame and have identity prototypes available. Metrics use leave-one-frame-out visual identity matching and fixed ReplicaSSG predicates.

## Results

| Input condition | Cross-view identity top-1 / MRR | Cross-view R@1 / R@5 / R@10 | Graph node accuracy (2,171 observations) | Oracle relation-triplet recall (49 scored observations) | Image encoding seconds / views/s |
|---|---:|---:|---:|---:|---:|
| GT padded box | 0.2142 / 0.3702 | 0.2142 / 0.5472 / 0.7232 | 0.2142 | 0.0000 | 16.02 / 137.62 |
| GT mask-isolated | **0.3800 / 0.5239** | 0.3800 / 0.6978 / 0.8107 | **0.3800** | 0.3265 | 16.85 / 130.81 |

The checkpoint loaded in 1.23 s. Model-resident CUDA allocation was 1.130 GiB. Peak allocated/reserved memory was 1.319/1.402 GiB, per-process only. Whole-job wall time and host memory were not recorded. DINOv3 has no shared text-image space; category mAP and zero-shot text accuracy are not applicable.

## Failed attempt and interpretation

The initial FP16 run was invalid. A follow-up CUDA diagnostic found non-finite features for all 40 paired real crops. Its identical 0.02994 top-1 across crop modes came from `argmax` collapsing to index zero. These metrics are withdrawn. The corrected runner uses FP32 and raises if embeddings are non-finite. The corrected FP32 metrics above are the only DINOv3 results used here.

On mask-isolated crops, DINOv3 approached SigLIP2 So400m identity accuracy (0.3800 vs 0.3966) and matched FG-CLIP2 graph triplet recall (0.3265). It did not improve the selected SigLIP2 mapping representation. The result is restricted to five held-out scans and GT masks. It does not validate DINOde or an end-to-end map pipeline.

- [Per-model and cross-model comparison](template_retrieval_test5.md)
- [Corrected FP32 JSON](data/dinov3-replica-test5-fp32.json)
- [Invalid FP16 JSON, retained for audit](data/dinov3-replica-test5-invalid-fp16.json)
- [NaN diagnostic, retained for audit](data/dinov3-input-diagnostic.json)
- [DINOv3 benchmark runner](../../conceptgraph/scripts/replica_dinov3_benchmark.py)
