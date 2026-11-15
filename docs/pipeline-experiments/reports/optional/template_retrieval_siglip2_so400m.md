# SigLIP2 So400m: ReplicaSSG held-out template retrieval and association

**Status:** completed, CUDA 0. **Model:** `google/siglip2-so400m-patch14-384`; verified local cache snapshot `e8e487298228002f3d8a82e0cd5c8ea9c567f57f` (the run JSON omitted this hash). Batch size 16; shared text prompt `a photo of a {category}`. Crop padding was 20 pixels. The masked crop used RGB `[128,128,128]` (processor mean `[0.5,0.5,0.5]`).

## Dataset and protocol

ReplicaSSG `files/test_scans.txt` contains 11 held-out scans. Local RGB and pose sequences exist for five: `office_1`, `office_3`, `office_4`, `room_1`, `room_2`. The other six (`apartment_1`, `apartment_2`, `hotel_0`, `frl_apartment_3`, `frl_apartment_4`, `frl_apartment_5`) were excluded for missing RGB/poses. Split-list SHA-256: `b7f8e16ad992562c76192dbfc96566dc1fdc4ab129a14bc1585e87badeb0dd33`.

Fifty evenly spaced frames per scene yielded 250 frames, 2,204 ground-truth object views, and 49 categories. Masks came from raycasting Replica semantic mesh instance IDs at camera poses. The model encoded GT padded boxes and GT-mask-isolated crops. Retrieval queries were the 49 category names. Cross-view identity used leave-one-frame-out object prototypes. There are 91 unique ReplicaSSG relation tuples across the scenes. The oracle graph metric scores 49 relation observations only when both endpoints are visible in the query frame and identity prototypes are available for both endpoints.

## Results

| Input condition | Category query macro AP | Category top-1 macro | Cross-view identity top-1 / MRR | Cross-view R@1 / R@5 / R@10 | Graph node accuracy (2,171 observations) | Oracle relation-triplet recall (49 scored observations) | Image encoding seconds / views/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| GT padded box | 0.0226 | 0.0399 | 0.2395 / 0.3910 | 0.2395 / 0.5652 / 0.7098 | 0.2395 | 0.0408 | 17.98 / 122.61 |
| GT mask-isolated | 0.0313 | 0.0555 | **0.3966 / 0.5373** | 0.3966 / **0.7057** / **0.8199** | **0.3966** | **0.3878** | 18.77 / 117.41 |

Load time was 19.14 s; category text encoding took 0.44 s. Model-resident CUDA allocation was 2.138 GiB. Peak allocated/reserved memory was 2.471/2.650 GiB, measured per-process and excluding other processes. Whole-job wall time and host memory were not recorded.

## Interpretation and limits

Mask isolation improved identity top-1 by 0.1571 and oracle relation-triplet recall by 0.3470 versus the padded condition. It produced the best identity and graph scores among the three visual association choices in this matched subset. This is an oracle-mask representation result, not detector or end-to-end map accuracy. No threshold tuning was performed.

Earlier eight-scene Replica results reported mask-isolated identity top-1 0.4015 and triplet recall 0.3026. Those are context only because their scene split differs. The matched comparisons and full artifacts are linked below.

- [Per-model and cross-model comparison](template_retrieval_test5.md)
- [Matched CLIP-family JSON](../../data/replica-test-retrieval-controls.json)
- [Historical component report](../feature_encoders.md)
- [Held-out VQA readiness and limits](template_vqa_query_vlm_status.md)
