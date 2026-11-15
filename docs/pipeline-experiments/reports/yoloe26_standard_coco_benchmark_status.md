# YOLOE-26 standard COCO box and mask AP benchmark status

## Status

The four-arm benchmark harness is implemented and committed. The first real CUDA run exposed a serialization defect after inference. All 500 images completed for the YOLOE-26-L segmentation arm, but final JSON output failed. The remaining three arms have not run. No standard COCO AP scores are available yet.

The implementation repair loop exhausted its local Qwen attempts. The configured OpenCode default and one policy-approved fallback both returned explicit free-usage limits. The required exhaustion diagnosis directed the workflow to stop retries and report the unresolved state.

## Planned comparison

All arms use the same fixed 500-image COCO val2017 subset, selected from sorted image IDs whose files exist using seed 42. Inference uses 640 px input, confidence 0.001, IoU 0.7, max detections 300, batch one, and CUDA 0. Segmentation arms also report mask AP; derived detection-only arms report box AP.

| Arm | Construction | Run status | Box AP | Mask AP |
|---|---|---|---:|---:|
| YOLOE-26-L-seg | Official `models/yoloe-26l-seg.pt` | 500/500 inferred; JSON serialization failed | Not scored | Not scored |
| YOLOE-26-M-seg | Official `models/yoloe-26m-seg.pt` | Not run | Not measured | Not measured |
| YOLOE-26-L-detect-only | `yoloe-26l.yaml` plus matching L segmentation checkpoint and saved prompts | Not run | Not applicable | Not applicable |
| YOLOE-26-M-detect-only | `yoloe-26m.yaml` plus matching M segmentation checkpoint and saved prompts | Not run | Not applicable | Not applicable |

The detection-only arms are derived heads from the matching official segmentation checkpoints. They are not official released detect-only checkpoint variants. See the [Ultralytics YOLOE documentation](https://docs.ultralytics.com/models/yoloe) for the documented construction path.

## Failed L-seg run details

- Software: repository `.venv` Python 3.12.13, PyTorch 2.14.1+cu130, Ultralytics 8.4.183.
- Source revision: `5df48ae6f19039e40f88d1bd05d91b99d31a5758`.
- Checkpoint SHA-256: `a612d2d505f24e14d87ec82d688b823b6cb600646664f16125ce6c84ce360da9`.
- COCO annotation SHA-256: `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`.
- Prompt encoder SHA-256: `35d7f213e4d75f38514e4656ad3cb91158bd33e3805d8ac349f23b186f66982f`.
- Resource observations: startup 1.58 GiB total GPU memory used and 30.26 GiB free; during inference 4.91–4.92 GiB total used, with 3.31 GiB attributed to this process.
- Failure: `TypeError: Object of type _CUuuid is not JSON serializable`, raised by `json.dump()` after all images completed. Exact source: `preflight()` stores the CUDA device UUID object without converting it to a JSON primitive.
- Wall time: not recorded reliably because serialization failed before the timing record could be saved.
- No benchmark metric was computed.

## Preserved artifacts

- [Run 1 failure report](yoloe26_l_seg_run1_failure.md)
- [Durable L-seg rows](data/yoloe26/yoloe-26l-seg_20261115_predictions.json.rows.jsonl)
- [L-seg progress status](data/yoloe26/yoloe-26l-seg_20261115_predictions.json.status.json)
- [Failed final-write temporary](data/yoloe26/yoloe-26l-seg_20261115_predictions.json.tmp)

These sidecars remain intact and must not be treated as complete scorer input. A future retry requires a repaired JSON-serializable provenance field and a new unique output path.
