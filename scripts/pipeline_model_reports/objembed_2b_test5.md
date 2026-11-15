# ObjEmbed-2B object-region embedding benchmark

**Result:** the model and official inference path ran successfully on the RTX 5090. This oracle-box experiment measures retrieval and identity association; it does not replace the separate WeDetect proposal benchmark.

## Configuration and inputs

- **Model:** `fushh7/ObjEmbed-2B`, Hugging Face revision `0e8aafeb53a95fa624de76de3a3c0f59615acbc2`, loaded locally from `HF_HOME`.
- **Repository:** official ObjEmbed repository at commit `9df13ff912bbb21ced3dcf46ee32e236ce1ae105`.
- **Runtime:** Python 3.10.12, PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, Transformers 4.57.1; BF16, SDPA, CUDA 0.
- **Split:** locally complete subset of ReplicaSSG `test_scans.txt`: `office_1`, `office_3`, `office_4`, `room_1`, and `room_2`. Six requested test scenes were unavailable locally and excluded before evaluation.
- **Input:** frozen manifest SHA-256 `b04b2e90e15a96d5a1e04154c7fbe1935135336b1aee5e8903c1e5aab77ab277`; 250 frame views, 2,204 visible GT object views, 49 categories. Inputs use full RGB frames and semantic-instance GT boxes through the official ObjEmbed object-token path. Proposals are oracle boxes.
- **Text queries:** `a photo of a {category}`; all 49 held-out category labels were queried against a 2,204-object gallery.

## Results

| Metric | Score |
|---|---:|
| Category-text macro AP | 0.02554 |
| Category-text Recall@1 / @5 / @10 | 0.0000 / 0.0816 / 0.1020 |
| Zero-shot category accuracy | 0.1261 |
| Cross-view identity top-1 | 0.3805 |
| Cross-view identity MRR | 0.5301 |
| Cross-view identity Recall@5 / @10 | 0.7195 / 0.8466 |
| Graph node identity accuracy | 0.3805 |
| Oracle predicate-triplet recall | 0.1429 across 49 edge observations |

The graph metric holds ReplicaSSG predicate labels fixed and propagates visual identity matching errors to node and triplet recovery. It does not measure predicate generation. The result JSON reports 2,171 node observations scored.

The matched SigLIP2 mask-isolated reference reported identity top-1 `0.3966` and oracle triplet recall `0.3878`; its category-text macro AP was `0.0313`. Those numbers are contextual only because SigLIP2 used isolated-mask crops while ObjEmbed uses full frames plus GT boxes and its object-token path. This is not a controlled encoder substitution. Keep the deployed feature choice unchanged pending a same-input comparison.

## Runtime and GPU resources

| Measure | Result |
|---|---:|
| Model load | 34.681 s |
| Text encoding | 1.845 s |
| Object/frame encoding | 24.394 s |
| Total model run | 106.706 s |
| Peak allocated / reserved CUDA memory | 4.716 / 4.855 GiB |
| Device | NVIDIA GeForce RTX 5090, 31.35 GiB reported total |

The runner completed all 250 images with no item or job failures. It emitted a deprecation warning for `torch_dtype` and upstream warnings for tensor copying and `meshgrid` indexing; neither affected completion. Throughput is 2204 object views over 24.394 seconds of image encoding, approximately 90.4 object views/s, but this excludes model load and category text encoding.

## Interpretation and artifacts

This experiment is valid for the frozen oracle-box protocol. It does not measure WeDetect proposal recall/AP or full pipeline map association. It supplies an actual ObjEmbed score, unlike the prior feasibility-only status. No threshold tuning was performed.

- Result JSON: `models/experiments/objembed/results/replicassg-five-scene-20261115-objembed-2b-run1/2b-metrics.json`.
- Input manifest and crops: `models/experiments/objembed/data/replicassg-five-scene-v1.json` and `models/experiments/objembed/data/gt-crops-v1/`.
- Official source: `models/experiments/objembed/repo/`.
- HF snapshot and shard hashes: the `model.snapshot_path` and `model.weight_shards` fields in result JSON.
- Runner: `models/experiments/objembed/adapter/benchmark_gt_object_embeddings.py`.

The split is limited to five locally available ReplicaSSG scenes. Do not treat the contextual SigLIP2 comparison as a causal winner determination.
