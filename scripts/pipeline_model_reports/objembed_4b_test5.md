# ObjEmbed-4B object-region embedding benchmark

**Result:** the model and official inference path ran successfully on the RTX 5090. This oracle-box experiment measures retrieval and identity association; it does not replace the separate WeDetect proposal benchmark.

## Configuration and inputs

- **Model:** `fushh7/ObjEmbed-4B`, Hugging Face revision `18b9b26c6d93f94045d2acc4914d1e4e1ded2a0a`, loaded locally from `HF_HOME`.
- **Repository:** official ObjEmbed source at commit `9df13ff912bbb21ced3dcf46ee32e236ce1ae105`.
- **Runtime:** Python 3.10.12, PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, Transformers 4.57.1; BF16, SDPA, CUDA 0.
- **Split:** locally complete subset of ReplicaSSG `test_scans.txt`: `office_1`, `office_3`, `office_4`, `room_1`, and `room_2`. Six requested test scenes were unavailable locally and excluded before evaluation.
- **Input:** frozen manifest SHA-256 `b04b2e90e15a96d5a1e04154c7fbe1935135336b1aee5e8903c1e5aab77ab277`; 250 frame views, 2,204 visible GT object views, 49 categories. Inputs use full RGB frames and semantic-instance GT boxes through the official ObjEmbed object-token path. Proposals are oracle boxes.
- **Text queries:** `a photo of a {category}`; all 49 held-out category labels were queried against a 2,204-object gallery.

## Results

| Metric | Score |
|---|---:|
| Category-text macro AP | 0.02537 |
| Category-text Recall@1 / @5 / @10 | 0.0204 / 0.0612 / 0.0816 |
| Zero-shot category accuracy | 0.1384 |
| Cross-view identity top-1 | 0.3782 |
| Cross-view identity MRR | 0.5348 |
| Cross-view identity Recall@5 / @10 | 0.7342 / 0.8554 |
| Graph node identity accuracy | 0.3782 |
| Oracle predicate-triplet recall | 0.1837 across 49 edge observations |

The graph metric holds ReplicaSSG predicate labels fixed and propagates visual identity matching errors to node and triplet recovery. It does not measure predicate generation. The result JSON reports 2,171 node observations scored.

On the same protocol, ObjEmbed-4B versus ObjEmbed-2B has lower identity top-1 (`0.3782` vs `0.3805`) and slightly lower category-text macro AP (`0.02537` vs `0.02554`). It has higher cross-view Recall@5 (`0.7342` vs `0.7195`), category accuracy (`0.1384` vs `0.1261`), and oracle triplet recall (`0.1837` vs `0.1429`). These mixed outcomes do not support a single overall winner.

The matched SigLIP2 mask-isolated reference reported identity top-1 `0.3966` and oracle triplet recall `0.3878`; its category-text macro AP was `0.0313`. Those numbers are contextual only because SigLIP2 used isolated-mask crops while ObjEmbed uses full frames plus GT boxes and its object-token path. This is not a controlled encoder substitution. Keep the deployed feature choice unchanged pending a same-input comparison.

## Runtime and GPU resources

| Measure | Result |
|---|---:|
| Model load | 4.020 s |
| Text encoding | 0.418 s |
| Object/frame encoding | 29.197 s |
| Total model run | 122.549 s |
| Peak allocated / reserved CUDA memory | 9.417 / 9.582 GiB |
| Device | NVIDIA GeForce RTX 5090, 31.35 GiB reported total |

The runner completed all 250 images without a recorded item or job failure. It emitted a deprecation warning for `torch_dtype` and upstream warnings for tensor copying and `meshgrid` indexing; neither affected completion. Object-view throughput was approximately 75.5 views/s during image encoding, excluding load and text encoding.

## Interpretation and artifacts

This experiment is valid for the frozen oracle-box protocol. It does not measure WeDetect proposal recall/AP or full pipeline map association. It supplies an actual ObjEmbed score, unlike the prior feasibility-only status. No threshold tuning was performed.

- Result JSON: `models/experiments/objembed/results/replicassg-five-scene-20261115-objembed-4b-run1/4b-metrics.json`.
- Input manifest and crops: `models/experiments/objembed/data/replicassg-five-scene-v1.json` and `models/experiments/objembed/data/gt-crops-v1/`.
- Official source: `models/experiments/objembed/repo/`.
- HF snapshot and shard hashes: the `model.snapshot_path` and `model.weight_shards` fields in result JSON.
- Runner: `models/experiments/objembed/adapter/benchmark_gt_object_embeddings.py`.

The split is limited to five locally available ReplicaSSG scenes. Do not treat the contextual SigLIP2 comparison as a causal winner determination.
