# WeDetect-Uni proposal benchmark: run 2 failed after full inference

## Outcome

**No accepted benchmark score.** The official WeDetect-Uni proposal path processed all 250 frozen frames and all 2,204 visible object views, then failed while importing the ConceptGraphs association helper for final aggregation. The saved per-object IoU values are partial artifacts and are not reported as recall scores. No retry or score salvage was performed.

Run 1 failed at zero frames with `KeyError: 'hit_iou_0.75_k100'`. A separately accepted repair changed threshold-key formatting, allowing run 2 to complete proposal inference. Run 2 then failed with `ModuleNotFoundError: No module named 'conceptgraph'` at the post-inference association import. This improves execution coverage only; it supplies no model comparison result and does not justify selecting WeDetect over the fixed detector.

## Configuration and inputs

- **Model:** `fushh7/WeDetect`, Hugging Face revision `3b9d0670cbf300d0b48f900ae87a6c4785d88939`.
- **Official source:** `models/experiments/objembed/wedetect-repo`, commit `42c34c6742a22f26280bb0e59f95a701baee1a32`.
- **Checkpoint:** `wedetect_base_uni.pth`, SHA-256 `e595578aeeb5f2a36cca383c23279cf7ea72e0bfb7c546dc39420359bf2d2c82`, 433,889,662 bytes, installed in the local Hugging Face cache.
- **Runtime:** local `models/experiments/envs/objembed`, Python 3.10.12, PyTorch/CUDA environment from the installed benchmark stack. `CUDA_VISIBLE_DEVICES=0`; model uses FP32 on `cuda:0`. `HF_HOME=/mnt/HDD1/unsloth_and_hugging_face_models/huggingface`; `TRANSFORMERS_OFFLINE=1`.
- **Data:** frozen five-scene ReplicaSSG test subset: `office_1`, `office_3`, `office_4`, `room_1`, and `room_2`; 250 frames, 2,204 visible object views, 49 categories. Manifest SHA-256 `b04b2e90e15a96d5a1e04154c7fbe1935135336b1aee5e8903c1e5aab77ab277`.
- **Proposal protocol:** 300 proposals per frame, recall at IoU 0.50 and 0.75 for K=100 and K=300; official base-Uni detector, FP32, official preprocessing and NMS.

## Measurements and limitations

Run 2 reached 250/250 frames and 2,204/2,204 object views. The elapsed interval from run-input creation to failure was 20.630445 seconds; it is not a per-frame latency statistic. The exception occurred after per-frame proposals and assigned proposal embeddings were saved, but before aggregate recall, association diagnostics and `results.json` were written.

The adapter queried PyTorch peak allocated/reserved memory counters, but their values were only written to the success result file that was never produced. No external GPU-memory sampler or maximum-RSS wrapper was used. Numeric peak GPU memory and maximum RSS are unavailable. After release, the RTX 5090 returned to 1,125 MiB / 32,607 MiB and 0% utilization, with no compute applications. This is idle state, not run peak.

## Comparison and decision

| Run | Completed frames | Object views | Outcome |
|---|---:|---:|---|
| Run 1 | 0/250 | 0/2,204 | Serialization `KeyError` before proposal scoring |
| Run 2 | 250/250 | 2,204/2,204 | Proposal inference complete; final aggregation failed on missing `conceptgraph` import |

The threshold-key repair fixed run 1's serialization defect. Run 2 still has no aggregate benchmark metrics, so no accuracy or model-choice comparison is possible. It remains unscored and is excluded from pipeline model selection.

## Artifacts

- Failure record: [`failure.json`](../../models/experiments/objembed/results/replicassg-five-scene-20261115-wedetect-uni-run2/failure.json), SHA-256 `86bd2cea630f884588ca8105b75a52eee48ae58780cc76f9bee3b1d7588654a9`.
- Raw per-frame proposals: [`per_frame_proposals.json`](../../models/experiments/objembed/results/replicassg-five-scene-20261115-wedetect-uni-run2/per_frame_proposals.json), SHA-256 `76c03a92ba74de99648559948bae0c54d802c9f4fc81a2573d903a61c067de84`.
- Raw per-object rows: [`per_object.csv`](../../models/experiments/objembed/results/replicassg-five-scene-20261115-wedetect-uni-run2/per_object.csv), SHA-256 `d780498a72facbc6e63354d6d279b5aabfb308b4a7b67500eb4b6d07ee6839a0`.
- Proposal-conditioned embeddings: [`assigned_proposal_embeddings.npy`](../../models/experiments/objembed/results/replicassg-five-scene-20261115-wedetect-uni-run2/assigned_proposal_embeddings.npy), SHA-256 `7e9e2a7b841c72b67cdac361a164c9b2d64ab4c80757a2a7825f1c3d29a6e458`.
- Full stdout/stderr: [`wedetect-uni-run2.log`](../../models/experiments/objembed/results/wedetect-uni-run2.log).
