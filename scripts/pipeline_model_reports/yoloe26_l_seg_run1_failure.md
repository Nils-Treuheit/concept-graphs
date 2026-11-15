# YOLOE-26-L segmentation, run 1

## Configuration

- Checkpoint: `models/yoloe-26l-seg.pt` (official L segmentation checkpoint).
- Dataset: COCO val2017, fixed 500-image sample selected from sorted annotation IDs with seed 42. The same sample protocol is planned for all four arms.
- Inference: `imgsz=640`, confidence 0.001, NMS IoU 0.7, max detections 300, batch one, `cuda:0`, `retina_masks=True`; one unscored warmup.
- Software: repository `.venv` Python 3.12.13, PyTorch 2.14.1+cu130, Ultralytics 8.4.183.
- Source revision: `5df48ae6f19039e40f88d1bd05d91b99d31a5758`.
- Checkpoint SHA-256: `a612d2d505f24e14d87ec82d688b823b6cb600646664f16125ce6c84ce360da9`.
- COCO annotation SHA-256: `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`.
- Prompt encoder SHA-256: `35d7f213e4d75f38514e4656ad3cb91158bd33e3805d8ac349f23b186f66982f`.

## Outcome

CUDA inference processed all 500 images, then failed before writing the final self-contained prediction JSON. `json.dump()` raised `TypeError: Object of type _CUuuid is not JSON serializable` because provenance stored `torch.cuda.get_device_properties(0).uuid` without converting it to a JSON primitive.

No COCO scores are available from this run. The preserved sidecars are not accepted scorer input, and the run is not counted as a completed benchmark result.

## Runtime and resources

- Per-image inference reached 500 / 500. Total wall time was not captured reliably because the result serializer failed before the timing record was saved.
- GPU memory at startup: 1.58 GiB used, 30.26 GiB free.
- During inference: approximately 4.91–4.92 GiB total GPU memory used; approximately 3.31 GiB attributed to this process.
- Failure stage: final JSON serialization. Process exit code: 1.

## Preserved artifacts

- Durable predictions: [`yoloe-26l-seg_20261115_predictions.json.rows.jsonl`](data/yoloe26/yoloe-26l-seg_20261115_predictions.json.rows.jsonl), 66,704,059 bytes, 99,568 rows.
- Progress record: [`yoloe-26l-seg_20261115_predictions.json.status.json`](data/yoloe26/yoloe-26l-seg_20261115_predictions.json.status.json), 500 / 500 marked complete by the per-image loop; state remains `running` because final serialization failed.
- Failed atomic-write temporary: [`yoloe-26l-seg_20261115_predictions.json.tmp`](data/yoloe26/yoloe-26l-seg_20261115_predictions.json.tmp), 15,764 bytes.

These artifacts are retained for diagnosis. The retry must use a new output path; the harness will refuse the existing run path.
