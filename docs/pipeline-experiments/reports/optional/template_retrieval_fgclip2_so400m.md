# FG-CLIP2 So400m: ReplicaSSG held-out template retrieval and association

**Status:** completed, CUDA 0. **Model:** `qihoo360/fg-clip2-so400m`; verified local cache snapshot `d57d30fe94a107dd6a2610eb4e9a135004d823a4` (the run JSON omitted this hash). Batch size 16; shared text prompt `a photo of a {category}`. Crop padding was 20 pixels. The masked crop used RGB `[128,128,128]` (processor mean `[0.5,0.5,0.5]`).

## Dataset and protocol

ReplicaSSG `files/test_scans.txt` contains 11 held-out scans. Local RGB and pose sequences exist for five: `office_1`, `office_3`, `office_4`, `room_1`, `room_2`. The other six (`apartment_1`, `apartment_2`, `hotel_0`, `frl_apartment_3`, `frl_apartment_4`, `frl_apartment_5`) were excluded for missing RGB/poses. Split-list SHA-256: `b7f8e16ad992562c76192dbfc96566dc1fdc4ab129a14bc1585e87badeb0dd33`.

Fifty evenly spaced frames per scene yielded 250 frames, 2,204 ground-truth object views, and 49 categories. Semantic mesh IDs and camera poses generated the ground-truth masks. The shared crop conditions use padded GT boxes and GT-mask-isolated crops. A third FG-CLIP2-only condition uses full-frame features with its native GT-box RoIAlign path; it is not a direct shared-input comparison.

## Results

| Input condition | Category query macro AP | Category top-1 macro | Cross-view identity top-1 / MRR | Cross-view R@1 / R@5 / R@10 | Graph node accuracy (2,171 observations) | Oracle relation-triplet recall (49 scored observations) | Image encoding seconds / views/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| GT padded box | 0.0247 | 0.0203 | **0.2446** / 0.3834 | 0.2446 / 0.5357 / 0.6799 | 0.2446 | **0.0612** | 26.31 / 83.76 |
| GT mask-isolated | 0.0403 | 0.0520 | 0.3791 / 0.5207 | 0.3791 / 0.6891 / 0.8075 | 0.3791 | 0.3265 | 27.38 / 80.50 |
| Native full-frame RoIAlign | 0.0232 | 0.0130 | 0.1681 / 0.3489 | 0.1681 / 0.5675 / 0.7586 | 0.1681 | 0.0408 | 3.65 / 603.26 |

Model load took 49.39 s; shared-category text encoding took 0.04 s. Model-resident CUDA allocation was 2.204 GiB. Peak allocated/reserved memory was 6.710/7.742 GiB, measured per-process and excluding other processes. Whole-job wall time and host memory were not recorded.

## Interpretation and limits

The mask-isolated crop improved cross-view identity top-1 from 0.2446 to 0.3791 and oracle relation-triplet recall from 0.0612 to 0.3265. SigLIP2 So400m performed better on these identity and graph outcomes, while FG-CLIP2 had higher category retrieval mAP. Native RoIAlign was faster in this run but weaker on identity scores; it is architecture-specific and cannot be interpreted as a matched crop comparison.

This is GT-mask representation evaluation. It does not measure detector, map or generated-predicate quality. No association thresholds were tuned.

- [Per-model and cross-model comparison](template_retrieval_test5.md)
- [Matched CLIP-family JSON](../../data/replica-test-retrieval-controls.json)
- [Historical component report](../feature_encoders.md)
- [Held-out VQA readiness and limits](template_vqa_query_vlm_status.md)
