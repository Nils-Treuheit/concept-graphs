# DINOde mask-isolated crop retrieval — test5

**Run timestamp:** 2026-11-17T03:33:42.167032Z
**Benchmark:** DINOde mask-isolated crop retrieval, single template, test5.
**Protocol:** DINOde crop protocol v2: mask-isolated, single-template test5 retrieval.

> This is a mask-isolated crop retrieval benchmark. It measures category-text
> retrieval on frozen ground-truth object crops with oracle masks. It is **not**
> VQA, map association, cross-view identity, detector performance, or a full
> pipeline result. No deployment suitability or upstream rights is inferred.

## Protocol and cohort

- **Cohort:** 2,204 object observations from five ReplicaSSG test scenes:
  `office_1` (338), `office_3` (553), `office_4` (402), `room_1` (505),
  `room_2` (406); 49 ReplicaSSG categories.
- **Input:** frozen ReplicaSSG crop manifest with 4,408 box/mask files,
  341,988,119 bytes; manifest SHA-256
  `f59384e7d3a1f4bdb60a79caf238aa965b2c45b611b9d4aa7467077c4d9c0b11`.
  Crop root: `/mnt/HDD1/Project_Data/public_datasets/Replica_benchmark/results/dinov3-test5-crops-fp32`.
- Each input is a generated crop paired with its ground-truth object mask. The
  mask isolates positive pixels for per-category logit pooling; this is an
  oracle-mask retrieval protocol, not a detector, full-image, or map benchmark.
- **Prompt:** literal single prompt `a photo of a {category}`. Macro AP is
  macro one-vs-rest AP over 49 categories; categories with no positives are
  omitted. Stable descending-score sorting is used over observations sorted by
  scene, frame, and object ID. Top-1/top-5 are observation-level category hits.

## Model and configuration

- **Repository commit:** `0a2c5182c44107fdd8c786b2192fcc3c5e5bebd1`.
- **Evaluation config:** `models/experiments/dinode/repo/configs/dinode_eval.json`,
  SHA `4682f2e9e347ed2fc1fef28499a5fac8fd673bd04e9c600c273d2911730ec9e3`.
- **Checkpoint:** `eccv26_dinode_coco_stuff.pth`, SHA
  `494354fe148a278a99c8186a59d9fa5ef14cf6254e312195eb08cac69cac1377`,
  214,589,829 bytes; epoch 9, 17,873,921 head parameters, effective tau
  0.06999999285.
- **Vision encoder:** frozen DINOv3 ViT-L/16, 448px, local HF revision
  `ea8dc2863c51be0a264bab82070e3e8836b02d51`, SHA
  `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`.
  Preprocessor config SHA `960c41d1f3a7778b936365769a2d90550b318a6c0a53a0296957adacfe5e0dd7`.
- **Text encoder:** OpenCLIP ViT-L/14 LAION 2B S32B B82K, revision
  `ffcbdd7da3d52f9fd7bb15e78ab6815e1a87ef60`, weights SHA
  `5ddb47339f44e4fd9cace3d3960d38af1b51a25857440cfae90afc44706d7e2b`.
- **Configuration:** text flow and CLS flow enabled; `in_channels=1024`,
  `topk=20`; validation slide window crop 448, stride 224, max long side 2048;
  PAMR disabled. DINOv3 remains frozen; learned DINOde head transports text
  features for comparison in the visual feature space.
- **Software:** Python 3.10.12; torch 2.11.0+cu130; transformers 5.19.0;
  open_clip_torch 2.20.0; huggingface_hub 2.2.0; timm 1.0.27; numpy 1.26.4;
  Pillow 12.2.0; scipy 1.15.3; matplotlib 3.10.9; tensorboard 2.20.0;
  protobuf 3.20.3. Runtime used the HDD-local environment at
  `models/experiments/dinode/venv` with CUDA on an NVIDIA RTX 5090.

## Results

- **Macro one-vs-rest AP:** `0.0368157099`.
- **Observation top-1 accuracy:** `0.0526315789`.
- **Observation top-5 accuracy:** `0.2568058076`.

### Per-scene macro AP

| Scene | Macro AP |
|---|---|
| office_1 | 0.0698726620 |
| office_3 | 0.0930023364 |
| office_4 | 0.0916207739 |
| room_1 | 0.0542764758 |
| room_2 | 0.0891226112 |

### Bootstrap interval

Exploratory five-scene cluster bootstrap, 10,000 resamples, seed `20261116`:

- Mean AP: `0.0446117566`
- 95% interval: `[0.0366858446, 0.0645807559]`

This five-cluster interval does not guarantee population coverage.

## Control comparison

Matched aggregate controls in
`scripts/pipeline_model_reports/data/replica-test-retrieval-controls.json`
(SHA `75c0f427163d194844d5fc26d1c6b3edbecaaacec8802969ee6f23a58afaf9c8`):

| Model | Macro AP | Top-1 | Top-5 |
|---|---|---|---|
| DINOde | 0.0368 | 0.0526 | 0.2568 |
| SigLIP2 So400m | 0.0313 | 0.0540 | 0.2187 |
| FG-CLIP2 So400m | 0.0403 | 0.0799 | 0.2731 |

Control validator reports same 2,204 cohort observations, five scene counts,
49-label vocabulary, literal template, and declared AP semantics.

**Caveats on comparison:**

- Controls have no per-view scores.
- Exact crop pixel equivalence to DINOde inputs is unproven.
- Control field wording is not an exact semantic string match.
- No paired bootstrap or significance test was performed.
- The comparison is descriptive only.

## Runtime and resources

- Model load: 3.3567 s.
- Text encoding: 0.2493 s.
- Inference: 185.2271 s.
- Total: 195.4031 s; approximately 11.90 views/s.
- **GPU:** RTX 5090; total 33,661,911,040 bytes; peak allocated
  3,656,526,848 bytes (~3.41 GiB), peak reserved
  4,622,123,008 bytes (~4.30 GiB).
- **Host RSS peak:** 4.3 GiB. Output/cache bytes: 370,485.
- Cache and temporary paths were under
  `models/experiments/dinode/data/replica_test5/.cache`, on the project HDD.

## Prior failures

These are implementation/environment failures, not negative model scores.

- Before creating the dedicated HDD-local Python 3.10 environment, an attempt
  to use the project `.venv` could not complete preflight because `tensorboard`
  was missing. No distinct report artifact for this attempt was retained.

| Artifact | SHA-256 | Failure |
|---|---|---|
| `dinode-replica-test5-attempt01-system-python-missing-open-clip.json` | `4a1fb8fcf525e97933a8641e161da3dd1effb29a486855f54af504ee2bcee3a1` | System Python lacked `open_clip`; moved to dedicated HDD-local Python 3.10 environment. |
| `dinode-replica-test5-attempt02-false-pair-validation.json` | `4e1f8d271a52c9c82b312f6312b805e67bb29d850fb4d70241847dad7cb41f99` | Validator removed nine characters from `-box.png` instead of eight and falsely rejected all box/mask pairs; fixed suffix length. |
| `dinode-replica-test5-attempt03-cuda-mask-pooling-failure.json` | `b81dc8ed6a9fff831b7b8f94434bc1a61ae846515834252bbb4ca4bde85a584e` | CUDA tensor/NumPy pooling and CUDA-to-NumPy conversion failed; mask moved to the logits device, accumulation made float32, output copied to CPU. |
| `dinode-replica-test5-attempt04-label-list-metrics-failure.json` | `2c9833c206bfe1abfdfa2131c083a7ba2917b743df1d00ca6df6a4c9aadbfe4a` | Inference completed but post-run metrics assumed a NumPy `.shape` on a Python list; helper now sizes from `len(labels)`. |

## Durable artifacts

- **Runner:** `scripts/pipeline_model_reports/dinode_replica_test5.py`.
  Current local source SHA-256:
  `dd91d3e6f5e8c0b4857e845b39453cf5f8ebc02e8a2f622b56d50a7fa8858603`.
- **Output JSON:**
  `models/experiments/dinode/data/replica_test5/dinode-replica-test5-mask-isolated-single-template.json`,
  2,993,579 bytes, SHA-256
  `d3715679fea556344bd61c69ed4c9c93fe7aa17d4757d16d604a8fbad7a07adf`.
- **Preflight output** SHA-256:
  `38780d44a506f35b031dda794b3bd01c0199b5ee7877d5dacc48c7db1b28ac21`.

## Limitations and licensing

- External implementation, checkpoint, model assets, Python environment, and
  benchmark data remain local and Git-ignored. No external repository or model
  weights belong in commits.
- DINOde checkout has no declared license. Keep this authorized local result
  and assets local; do not assert redistribution rights.
- The LAION model card describes non-deployment scope; no deployment inference
  is made.

## Source references

- DINOde repository: <https://github.com/yoon307/DINOde> (no LICENSE entry
  observed in the visible root listing).
- LAION OpenCLIP model card: <https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K>
  (lists MIT and describes deployed use as out of scope).

The license observation is not a legal conclusion.
