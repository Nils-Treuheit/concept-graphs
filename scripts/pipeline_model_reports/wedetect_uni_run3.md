# WeDetect-Uni proposal recall and GT-matched diagnostic — run3

## Identity and frozen protocol

- Official WeDetect repository `models/experiments/objembed/wedetect-repo`, commit `42c34c6742a22f26280bb0e59f95a701baee1a32`.
- Model `fushh7/WeDetect@3b9d0670cbf300d0b48f900ae87a6c4785d88939`; `wedetect_base_uni.pth`, 433,889,662 bytes, SHA-256 `e595578aeeb5f2a36cca383c23279cf7ea72e0bfb7c546dc39420359bf2d2c82`.
- Adapter `models/experiments/objembed/adapter/benchmark_wedetect_uni_proposals.py`, SHA-256 `a96934f04ff8f145d13ff90c0e59ec6130ee6b94f2e8d7ba9b57d48a4b7de24b`.
- Frozen manifest `models/experiments/objembed/data/replicassg-five-scene-v1.json`, SHA-256 `b04b2e90e15a96d5a1e04154c7fbe1935135336b1aee5e8903c1e5aab77ab277`.
- Manifest contents: five locally available test scenes (office_1, office_3, office_4, room_1, room_2), 250 frames, 2,204 visible object views, 49 categories. Six requested test scenes were unavailable locally.
- Official WeDetect-Uni detector, FP32, CUDA 0, 300 proposals/frame, recall at IoU 0.50/0.75 and K=100/300. Official letterbox 640x640 RGB/255 and NMS top 300.
- Python 3.10.12; Torch 2.7.1+cu128; torchvision 0.22.1+cu128; CUDA 12.8; RTX 5090 32 GiB.

## Completion and artifact validation

Run3 completed successfully with no failures. Root artifact checks confirmed 250 per-frame records, 2,204 summed object views, per-frame proposal lists of length 300, 2,204 rows in `per_object.csv`, and finite float32 `assigned_proposal_embeddings` of shape (2,204, 768).

- `results.json`, 10,262 bytes, SHA-256 `008ca97861d5b50c9ff3958d314a84cde8b47910f069ab8679de6de4eb59dac7`.
- `per_object.csv`, 290,751 bytes, SHA-256 `d780498a72facbc6e63354d6d279b5aabfb308b4a7b67500eb4b6d07ee6839a0`.
- `per_frame_proposals.json`, 6,372,754 bytes, SHA-256 `3fc673d21573e8e673d1b0e5b308ac40fe33ce6996f624097397814e7144d81a`.
- `assigned_proposal_embeddings.npy`, 6,770,816 bytes, SHA-256 `7e9e2a7b841c72b67cdac361a164c9b2d64ab4c80757a2a7825f1c3d29a6e458`.
- `run-input.json`, 1,004 bytes, SHA-256 `d8d2a0e0ff5c6bac5a2a43c8672bba4a29898cfb2c3e1ba36f3bc84c73042dbd`.
- External log `models/experiments/objembed/results/wedetect-uni-run3.log`, 968 bytes, SHA-256 `d03aab8286be52948b9e05460a66ff7d1f7be61e05124f834cfae2eb28fa9f0c`.
- GNU time `models/experiments/objembed/results/wedetect-uni-run3.time.txt`, 1,083 bytes, SHA-256 `7700f1111fe948a0becf359a27bf152d0c9f7615cc1f88bf5d862979faed49f0`.

## Proposal recall

### Aggregate

| Metric | Result |
|---|---:|
| Recall IoU 0.50, K=100 | 0.373412 |
| Recall IoU 0.75, K=100 | 0.114791 |
| Recall IoU 0.50, K=300 | 0.465064 |
| Recall IoU 0.75, K=300 | 0.158802 |
| Mean best IoU, K=100 / K=300 | 0.410205 / 0.465790 |

### Per scene

| Scene | Views | Recall .50 K100 | Recall .50 K300 | Recall .75 K100 | Recall .75 K300 |
|---|---:|---:|---:|---:|---:|
| office_1 | 338 | 0.488166 | 0.568047 | 0.168639 | 0.218935 |
| office_3 | 553 | 0.273056 | 0.329114 | 0.063291 | 0.090416 |
| office_4 | 402 | 0.390547 | 0.497512 | 0.136816 | 0.189055 |
| room_1 | 505 | 0.368317 | 0.481188 | 0.104950 | 0.158416 |
| room_2 | 406 | 0.403941 | 0.512315 | 0.130542 | 0.172414 |

## Timing

- Model load: 1.01237 s.
- Inference loop: 19.7457 s.
- Total adapter wall time: 32.4239 s.
- Process elapsed (GNU time): 45.97 s.

### Per-frame inference latency (250 frames)

| Statistic | Seconds |
|---|---:|
| Mean | 0.072866 |
| Median | 0.059080 |
| p95 | 0.065354 |
| Max | 3.29594 |
| Standard deviation | 0.204275 |

The maximum is a single-frame outlier; the p95 and median remain under 0.066 s.

## Memory

### GNU time (maximum resident set size)

- 2,155,964 KiB (~2.06 GiB).

### PyTorch CUDA (CUDA 0)

- Peak allocation: 0.602313 GiB.
- Peak reservation: 0.740234 GiB.
- At load: 0.402452 / 0.410156 GiB.

## Warnings and state-dict notes

- Stdout contains one non-fatal `torch.meshgrid` indexing deprecation warning.
- State-dict loaded with `strict=false`: zero missing keys and four unexpected keys under `backbone.norm` and `backbone.head`. These unused checkpoint keys are disclosed here.

## Prior failures

- Run1 failed before proposal scoring with `KeyError: 'hit_iou_0.75_k100'`.
- Run2 processed all 250 frames and 2,204 object views but failed during final aggregation with `ModuleNotFoundError: No module named 'conceptgraph'`. No run2 aggregate score is accepted; the partial rows from run2 are not salvaged. See the [separate run2 failure report](wedetect_uni_run2_failure.md).
- Run3 set the repository root in `PYTHONPATH` and completed successfully. It supplies the first accepted WeDetect-Uni proposal recall and GT-matched diagnostic on this frozen local subset.

## GT-matched proposal-embedding diagnostic (not autonomous map association)

The adapter assigns each GT object view the embedding of its highest-IoU proposal among the frame's score-ranked 300 proposals, including matches below IoU 0.50. This is **GT-matched proposal embedding evaluation, not autonomous map association or predicate prediction.**

- Cross-view retrieval top-1 / Recall@1: 0.181483.
- Mean reciprocal rank: 0.321201.
- Recall@5: 0.461078.
- Recall@10: 0.628282.
- Scene-graph node identity accuracy: 0.181483.
- Oracle predicate triplet recall: 0.020408.

2,171 node observations and 49 edge observations were scored. Predicate labels are fixed and reused; this does not measure predicate generation.

Do not compare these to mask-isolated SigLIP2 or FG-CLIP2 results. Proposal coverage and GT-conditioned identity are distinct tasks. These diagnostics do not justify replacing the selected feature encoder or tuning map thresholds.

## Interpretation limits

- The available detector and map protocols are not equivalent to this frozen test subset. No direct comparison to the selected detector is reported.
- The run does not alone justify replacing the selected detector.
- This is the first accepted WeDetect-Uni proposal recall on the locally complete five-scene subset. A broader scene coverage and an independent detector benchmark would be needed for a production comparison.

## Run status

Run3 completed. This is the first accepted WeDetect-Uni proposal recall and GT-matched diagnostic on the frozen five-scene ReplicaSSG local subset.
