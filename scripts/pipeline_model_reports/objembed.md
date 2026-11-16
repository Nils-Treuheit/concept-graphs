# ObjEmbed and WeDetect experiment status

## Completed ObjEmbed encoder runs

The official ObjEmbed object-token path has now run on the frozen, five-scene ReplicaSSG subset using oracle GT boxes. Two model-specific reports contain exact configuration, metrics, runtime, memory, and artifacts:

- [ObjEmbed-2B benchmark](objembed_2b_test5.md): cross-view identity top-1 `0.3805`, Recall@5 `0.7195`, oracle relation-triplet recall `0.1429`; peak allocated memory `4.716 GiB`.
- [ObjEmbed-4B benchmark](objembed_4b_test5.md): cross-view identity top-1 `0.3782`, Recall@5 `0.7342`, oracle relation-triplet recall `0.1837`; peak allocated memory `9.417 GiB`.

Both runs share manifest SHA-256 `b04b2e90e15a96d5a1e04154c7fbe1935135336b1aee5e8903c1e5aab77ab277`, 250 frame views, 2,204 visible object views, and 49 categories. Each loads BF16 weights locally from `HF_HOME` and uses SDPA on CUDA 0. The reports are separate because the checkpoints represent separate model choices.

The two checkpoints yield mixed outcomes. 2B has slightly better identity top-1 and category-text macro AP. 4B has higher category accuracy, cross-view Recall@5, and oracle graph triplet recall, at roughly twice the peak allocated memory and longer total run time. These metrics do not identify one overall winner.

The earlier SigLIP2/FG-CLIP2 reference uses mask-isolated or padded crops, while ObjEmbed consumes full frames plus GT boxes through its official object-token path. Those scores are contextual and are not a controlled encoder substitution. The evidence does not justify replacing the current feature encoder or tuning map thresholds.

## WeDetect proposal path

The WeDetect repository and local HF checkpoint are installed. Run 2 completed proposal inference on all 250 frames and 2,204 object views, but final aggregation failed because the isolated Python environment could not import `conceptgraph`. No aggregate metrics were produced; the partial IoU rows are not accepted as scores. Run 1 failed before inference with the threshold-key `KeyError`. See the [separate WeDetect run 2 report](wedetect_uni_run2_failure.md). This proposal path remains distinct from the oracle-box ObjEmbed measurements above. Do not read the ObjEmbed scores as WeDetect results.

## Protocol and artifacts

The frozen input is the local-complete subset of ReplicaSSG `test_scans.txt`: `office_1`, `office_3`, `office_4`, `room_1`, and `room_2`; six requested scenes were unavailable locally. Source repository commit: `9df13ff912bbb21ced3dcf46ee32e236ce1ae105`. Runtime: Python 3.10.12, Torch 2.7.1+cu128, torchvision 0.22.1+cu128, Transformers 4.57.1. The GPU-only runner is `models/experiments/objembed/adapter/benchmark_gt_object_embeddings.py`.

- ObjEmbed-2B raw JSON: `models/experiments/objembed/results/replicassg-five-scene-20261115-objembed-2b-run1/2b-metrics.json`.
- ObjEmbed-4B raw JSON: `models/experiments/objembed/results/replicassg-five-scene-20261115-objembed-4b-run1/4b-metrics.json`.
- Manifest and crops: `models/experiments/objembed/data/replicassg-five-scene-v1.json` and `models/experiments/objembed/data/gt-crops-v1/`.
- Model snapshots and shard hashes are recorded in each result JSON.

Protocol source: `MODERNIZED_PIPELINE_IMPLEMENTATION_PLAN.md`, “Further embedding research” and “Evaluation plan.”
