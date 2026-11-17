# DINOv3 + dino.txt ViT-L/16 global shared-space crop category retrieval — measured result

**Status:** measured and reviewed (GT-mask-isolated crop category retrieval). Stage 1 cross-view identity and graph diagnostics are now measured and reported; see the [dedicated graph report](../../../../scripts/pipeline_model_reports/dinotxt_identity_graph_test5.md) for details. Stage 2 (optional pipeline-slot acceptance) remains explicitly pending.

**Runner:** `scripts/pipeline_model_reports/benchmarks/dinotxt_replica_test5.py`  
**Result:** `models/experiments/t_ren/data/dinotxt_test5/dinotxt_test5_global_retrieval_20261116T161741Z_2382729.json`  
**Evaluated:** 16 November 2026; independent recomputation matched the stored aggregate metrics exactly.

---

## Model and checkpoint configuration

- **Backbone:** `dinov3_vitl16_pretrain_lvd1689m` ViT-L/16, LVD-1689M pretrain  
  - File: `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`  
  - SHA-256: `e73f0550f00e5eafe9712177da027e5f34df3de5250b56370bcb9acaa23dab9c` (matches expected `8aa4cbdd` prefix)  
  - Size: 1,213,021,279 bytes
- **Vision head + text encoder:** `dinov3_vitl16_dinotxt_vision_head_and_text_encoder`  
  - File: `dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth`  
  - SHA-256: `a442d8f52a3a7ad715bf6b7d8117fb3a84d54249389b0a13f6956cd0d2eca4f0` (matches expected `a442d8f5` prefix)  
  - Size: 2,253,936,683 bytes
- **BPE vocabulary:** `bpe_simple_vocab_16e6.txt.gz`  
  - SHA-256: `924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a`  
  - Size: 1,356,917 bytes
- **DINOv3 source commit:** `6876159a11b4df116f30f667f8c9888617df0751` (exact match)
- **Embedding dimension:** 2048
- **Logit scale exponent:** 100.0

All assets are project-local under `models/experiments/t_ren/weights/dinov3/` (git-ignored by the existing `weights` rule).

---

## Prompt and transform

- **Prompt template:** `a photo of a {category}` (49 categories)
- **Transform pipeline:**

```
Compose(
      ToImage()
      Resize(size=[256], interpolation=InterpolationMode.BICUBIC, antialias=True)
      CenterCrop(size=(224, 224))
      Compose(
        ToDtype(scale=True)
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
  )
)
```

- **Signature:** `(*, resize_size: int = 256, crop_size: int = 224, interpolation=<InterpolationMode.BICUBIC: 'bicubic'>, mean: Sequence[float] = (0.485, 0.456, 0.406), std: Sequence[float] = (0.229, 0.224, 0.225)) -> torchvision.transforms.v2._container.Compose`
- **Batch size:** 32
- **No tuning** on this held-out set.

---

## Environment

- **Device:** CUDA 0 (NVIDIA GeForce RTX 5090)
- **PyTorch:** 2.11.0+cu130 (CUDA 13.0)
- **Python:** 3.10.12
- **Platform:** Linux-6.8.0-138-generic-x86_64-with-glibc2.35
- **Offline flags:** `HF_DATASETS_OFFLINE=1`, `HF_HUB_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`, `TORCH_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`
- **Mode:** FP32

---

## Data split and scene counts

- **Test subset:** ReplicaSSG test-5 (five locally complete scenes)
- **Scenes and view counts:**

  | Scene | Views |
  |---|---:|
  | office_1 | 338 |
  | office_3 | 553 |
  | office_4 | 402 |
  | room_1 | 505 |
  | room_2 | 406 |
  | **Total** | **2,204** |

- **Crop/mask files:** 4,408
- **Categories:** 49
- **Excluded scenes:** six planned test scenes lack local RGB/poses

---

## Aggregate metrics

| Metric | Value |
|---|---:|
| Macro-category mAP | 0.0261293 |
| Macro-query Recall@1 | 0.0204082 |
| Macro-query Recall@5 | 0.0612245 |
| Macro-query Recall@10 | 0.1428571 |
| Macro-query Recall@50 | 0.4081633 |
| Macro top-1 classification | 0.0139328 |
| Micro top-1 classification | 0.0263158 |
| Micro top-5 classification | 0.1588022 |

**Per-scene breakdown** (mAP, recall@1, recall@5, recall@10, recall@50):

| Scene | mAP | R@1 | R@5 | R@10 | R@50 |
|---|---:|---:|---:|---:|---:|
| office_1 | 0.0571895 | 0.0 | 0.0952381 | 0.1904762 | 0.4761905 |
| office_3 | 0.0746794 | 0.05 | 0.3 | 0.35 | 0.6 |
| office_4 | 0.0665244 | 0.0588235 | 0.1176471 | 0.2352941 | 0.5882353 |
| room_1 | 0.0524760 | 0.0416667 | 0.125 | 0.2083333 | 0.625 |
| room_2 | 0.0831443 | 0.1052632 | 0.1578947 | 0.4210526 | 0.6842105 |

**Per-category highlights** (selected categories sorted by mAP):

| Category | mAP | R@1 | R@50 | Zero-shot top-1 |
|---|---:|---:|---:|---:|
| wall | 0.2794902 | 1.0 | 1.0 | 0.0306859 |
| window | 0.1106587 | 0.0 | 1.0 | 0.0 |
| blinds | 0.0960396 | 0.0 | 1.0 | 0.1337209 |
| lamp | 0.0878645 | 0.0 | 1.0 | 0.0190840 |
| vase | 0.0723449 | 0.0 | 1.0 | 0.0 |
| switch | 0.0695127 | 0.0 | 1.0 | 0.0925926 |
| pipe | 0.0554252 | 0.0 | 1.0 | 0.0192308 |
| vent | 0.0448038 | 0.0 | 0.0 | 0.0098039 |

Full per-category tables (mAP, recall@1/5/10/50, zero-shot top-1, ground-truth object counts) are in the raw result JSON.

---

## Descriptive SigLIP2 and FG-CLIP2 controls

The frozen control benchmark JSON (`data/replica-test-retrieval-controls.json`, SHA-256 `75c0f427163d194844d5fc26d1c6b3edbecaaacec8802969ee6f23a58afaf9c8`) provides unpaired per-model aggregates. No paired per-view scores exist, so **no paired deltas or significance claims are made**.

| Model | Macro mAP | Macro top-1 | Micro top-1 | Micro top-5 | R@5 | R@10 | R@50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SigLIP2 So400m | 0.0313053 | 0.0555370 | 0.0539927 | 0.2186933 | 0.1020408 | 0.1020408 | 0.3265306 |
| FG-CLIP2 So400m | 0.0403068 | 0.0520141 | 0.0798548 | 0.2731397 | 0.0816327 | 0.1224490 | 0.3877551 |
| **DINOtxt (this run)** | **0.0261293** | **0.0139328** | **0.0263158** | **0.1588022** | **0.0612245** | **0.1428571** | **0.4081633** |

DINOtxt records higher recall@10 and recall@50 than both controls, but lower values on the four classification/AP aggregates. The unpaired controls support descriptive comparison only and do not establish production substitution.

---

## Runtime and throughput

| Phase | Time (s) |
|---|---:|
| Model load | 4.659 |
| Text encode | 0.135 |
| Image encode (69 batches) | 7.737 |
| First batch (s) | 0.155 |
| Warm batch mean (s) | 0.112 |
| Retrieval metrics | 0.022 |
| Manifest pre/post | 0.671 / 0.659 |
| **Total wall** | **33.960** |
| **Throughput** | **284.860 views/s** |

- **Cached views:** 0 of 2,204 (cold start; cache rebuilt)
- **Image encode batches:** 69

---

## Memory

| Metric | Value |
|---|---:|
| Peak CUDA allocated | 3.69054 GiB |
| Peak CUDA reserved | 3.962891 GiB |
| Resident CUDA after model load | 3.277241 GiB |
| Peak host RSS | 6.162868 GiB |

---

## Cache and artifact paths

All under the git-ignored project data directory `models/experiments/t_ren/data/dinotxt_test5/`:

- **Result JSON:** `dinotxt_test5_global_retrieval_20261116T161741Z_2382729.json`
- **Embedding cache (NPZ):** `dinotxt_test5_global_embeddings_cache.npz` — 2,204 entries, 18,848,630 bytes, rebuilt (0 hits)
- **Embedding sidecar (JSON):** `dinotxt_test5_global_embeddings_cache.json`
- **Baseline manifest:** `t_ren_test5_crop_manifest_20261116.json`

Verified weights under `models/experiments/t_ren/weights/dinov3/` (git-ignored by the `weights` rule).

---

## Failures and warnings

None recorded. Input manifest unchanged after inference.

---

## Limitations

1. GT-mask-isolated crop category retrieval only; does not measure identity, mapping, graph quality, or VQA.
2. Control comparison is descriptive only — no paired per-view scores.
3. Five scenes locally complete; six test scenes lack local RGB/poses and are excluded.
4. Scores come from the GT-mask-isolated crop mode; they must not be compared against other crop modes as if paired.
5. The five scenes are the independent sampling units; object views within a scene are correlated.
6. Official frozen dino.txt transform and prompt used as-is; nothing tuned on this held-out set.
7. Single-run GPU timings; no multi-run variance estimates.

---

## Scope statement

This result completes the GT-mask-isolated crop category-retrieval subtask. The cross-view identity diagnostics and oracle predicate-triplet recall are now measured and reported; see the [dedicated graph report](../../../../scripts/pipeline_model_reports/dinotxt_identity_graph_test5.md). Optional full pipeline-slot acceptance (Stage 2) remains **pending**. The asset gate for backbone, head, and vocabulary is **passed** (SHA-256 verified).
