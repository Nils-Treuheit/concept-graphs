# T-REN point-conditioned region-text retrieval report

**Status:** Completed on 2026-11-16. CUDA run finished; result artifact created `2026-11-16T14:26:03.578593+00:00`.

**Task label.** T-REN point-conditioned region-text retrieval: zero-shot category retrieval of one text-aligned token per GT-mask point prompt. This is not the global DINOv3.txt (DINOtxt) embedding comparison, not a map-level association benchmark, not an autonomous proposal test, and not VQA. The global DINOtxt-vs-SigLIP2 crop-level comparison is now measured and reviewed (GT-mask-isolated crop-category subtask); see `../../docs/pipeline-experiments/reports/optional/dinotxt_global_retrieval_test5.md`. DINOtxt cross-view identity/graph diagnostics and optional pipeline-slot acceptance remain pending.

## Method and config

**Point protocol.** For each GT mask, the prompt point is the pixel with the maximum Euclidean distance to the mask boundary; ties are broken deterministically in row-major order. Native `(y, x)` coordinates are mapped to the 512-square input coordinate system. The point prompt is fed to the official T-REN RegionEncoder, producing one text-aligned token per view.

**Text queries.** The runner passes the bare category names to the T-REN `TextEncoder` and supplies the prompt prefix itself. The upstream `TextEncoder` in `models/experiments/t_ren/repo/model.py` defines a default prompt of `photo of a `, but the runner overrides it with `TEXT_ENCODER_PROMPT = "a photo of a "`, passed as the `prompt` argument to `TextEncoder.__init__`. `TextEncoder.forward()` prepends that runner-supplied prefix to each bare category name, so the effective per-category query is `a photo of a {category}`.

**Frozen T-REN config.**

| Setting | Value |
|---|---|
| Feature extractor | `dinov3_vitl16` |
| Patch size | 16 |
| Image resolution | 512 |
| Text embed dim | 1024 |
| Hidden dim | 1024 |
| Attention heads | 8 |
| Decoder layers | 2 |
| Multiscale regions | 3 |
| Merge IoU | 0.8 |
| Merge similarity | 0.975 |

**Model path.** Official DINOv3 dino.txt ViT-L/16 factory plus T-REN FeatureExtractor, RegionEncoder, and TextEncoder. CUDA FP32, no autocast. The strict RegionEncoder state load passed. The runner calls the exact official local factory directly and avoids `hubconf.py`; it verifies factory origin under the local pinned checkout.

## Held-out data coverage

Five frozen held-out scenes, 50 frames each: 2,204 labeled object views, 49 categories, 4,408 crop+mask files, 341,988,119 bytes.

| Scene | Labeled views |
|---|---:|
| office_1 | 338 |
| office_3 | 553 |
| office_4 | 402 |
| room_1 | 505 |
| room_2 | 406 |

Inputs: `dinov3-test5-crops-fp32` crop cache, ReplicaSSG `objects.json`, baseline manifest `/mnt/HDD1/Project_Code/concept-graphs/models/experiments/t_ren/data/dinotxt_test5/t_ren_test5_crop_manifest_20261116.json`, controls `scripts/pipeline_model_reports/data/replica-test-retrieval-controls.json`. All mask/crop entries matched the baseline entrywise before and after inference (no data drift).

## Metrics

**Aggregate (2,204 views, 49 categories).**

| Metric | Value |
|---|---:|
| Macro category AP | 0.0246173 |
| Zero-shot top-1, micro | 0.131125 |
| Zero-shot top-1, macro-category | 0.017549 |
| Zero-shot top-5, micro | 0.244555 |
| Macro-query recall R@1 | 0.020408 |
| Macro-query recall R@5 | 0.040816 |
| Macro-query recall R@10 | 0.102041 |
| Macro-query recall R@50 | 0.285714 |

**Per scene (macro AP / top-1 micro / top-5 micro).**

| Scene | Macro AP | Top-1 micro | Top-5 micro |
|---|---:|---:|---:|
| office_1 | 0.064194 | 0.144970 | 0.257396 |
| office_3 | 0.069905 | 0.122966 | 0.216998 |
| office_4 | 0.067225 | 0.149254 | 0.253731 |
| room_1 | 0.048654 | 0.112871 | 0.237624 |
| room_2 | 0.076463 | 0.135468 | 0.270936 |

## Runtime and resources

- Total wall 219.042 s; init + strict checkpoint load 12.435 s; text encoding of the 49 queries 0.183 s; feature stage 44.387 s; region stage 1.445 s; total inference 45.832 s.
- First-batch cold latency 0.389 s; warm inference 45.443 s.
- Throughput 48.089 views/s overall (batch 8); 48.324 views/s warm.
- Peak CUDA allocated 4.980 GiB; peak reserved 5.297 GiB. CUDA stats were reset after component load, so these peaks cover the inference stages only.
- Peak host RSS 6,871,064 KiB (~6.55 GiB).
- Device: RTX 5090, 31.35 GiB usable (32 GiB class).
- Stack: torch 2.14.1+cu130, torchvision 0.29.1+cu130, Python 3.12.13.

## Provenance and local assets

- Source commits: T-REN `0817b0d015a4462756248825eee0c3da5aada37a`; DINOv3 `6876159a11b4df116f30f667f8c9888617df0751`. Both source worktrees were clean at run time.
- Source links: [T-REN](https://github.com/savya08/T-REN), [DINOv3](https://github.com/facebookresearch/dinov3), [Meta DINOv3 access page](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/).
- Local assets (outside Git):

| Asset | Size (bytes) | SHA-256 |
|---|---:|---|
| DINOv3 backbone | 1,213,021,279 | `e73f0550f00e5eafe9712177da027e5f34df3de5250b56370bcb9acaa23dab9c` |
| dino.txt head + text encoder | 2,253,936,683 | `a442d8f52a3a7ad715bf6b7d8117fb3a84d54249389b0a13f6956cd0d2eca4f0` |
| BPE | 1,356,917 | `924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a` |
| T-REN RegionEncoder checkpoint | 1,455,980,684 | `7f85132237001d52176226b3fd6ebde8ecfc08569ae196d56deeb5b052bf9b1f` |

Weights and source repositories stay out of Git.

**License and provenance.** The local T-REN source tree at `models/experiments/t_ren/repo/` is MIT-licensed per its `LICENSE` file. The DINOv3 materials (code, weights, and algorithms) in the pinned checkout of the official `facebookresearch/dinov3` repository at `models/experiments/t_ren/dinov3-repo/` are governed by the DINOv3 License Agreement in its `LICENSE.md`. This report does not redistribute model weights; all assets remain outside Git. This note documents provenance only; it does not itself satisfy or replace any license or agreement acceptance requirement.

## Warnings and failed attempts

- sklearn is absent in the runner venv; an import-spec-valid fail-loud PCA stub covered T-REN's unused visualization import. The region-text path never instantiated PCA.
- Five preserved pre-success attempts in Git-ignored project data:
  - `132224Z_1933153`: false dependency preflight from missing `importlib.util`.
  - `132547Z_1942180`: flattened mask distance index bug.
  - `134106Z_1981097`: stub `sklearn.__spec__ is None`.
  - `135335Z_2013675`: upstream `hubconf.py` requires absent torchmetrics.
  - `140307Z_2037784`: NumPy helper scope NameError after completed inference.
- All issues were repaired and later attempts succeeded. No unresolved model/data blocker remains for this run.

## Descriptive controls (not paired)

GT-mask-isolated crops from a separate run, with no per-view score alignment to the T-REN run:

| Control | Macro AP | Top-1 micro |
|---|---:|---:|
| SigLIP2 So400m | 0.031305 | 0.053993 |
| FG-CLIP2 | 0.040307 | 0.079855 |

These are descriptive context only: no paired deltas, no significance claims. The control protocol is the GT-mask-isolated crop protocol of that separate run; it is not the point-conditioned T-REN protocol and is named as such.

## Interpretation limits

- Absolute scores are low on this zero-shot, point-conditioned protocol.
- Controls are unpaired, so no comparative claim is made against them.
- This result does not answer the separate global DINOtxt-vs-SigLIP2 substitution question, map-level association, autonomous proposals, or VQA.

## Artifacts and reproducibility

- Full score matrix: schema `t-ren-replica-test5-region-retrieval` v1, 2,204 rows by 49 classes. Retained in Git-ignored project data at `/mnt/HDD1/Project_Code/concept-graphs/models/experiments/t_ren/data/dinotxt_test5/t_ren_region_retrieval_test5_20261116/t_ren_test5_region_retrieval_20261116T142603Z_2097135.json`, status `completed`, created `2026-11-16T14:26:03.578593+00:00`, on-disk SHA-256 `5550c9f00a7770e7708eb1286099fbbaffe9e03b3ef941ae9ff238096ee0d126`.
- The `88882675f1d4ad787c67ad2e794313b97d2306c149a23568f59f48c325619463` value recorded inside the JSON is an internal output field, not the on-disk SHA-256 of the file.
- The reviewed runner is copied byte-for-byte to `scripts/pipeline_model_reports/benchmarks/t_ren_replica_test5.py` (SHA-256 `fb07caa7d529d4a71a201ef20d5ce155f8c93e795c4dfdc6404251a774e15d05`) for reproducibility.
