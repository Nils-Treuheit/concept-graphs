# EOVSAM setup, failure history, and fusion scoring

**History scope:** this report preserves setup and pre-run2 failures. The completed run2 inference, fusion scoring, and independent AP recomputation are documented below.

## Completed run2 inference artifacts

Current authoritative run: `models/experiments/eovsam/output/coco500-mask-fusion-run2/`.

- `runner-metadata.txt`: inference exit status 0; `time.txt`: 71.11 elapsed seconds, max host RSS 9,493,828 KiB, exit 0.
- Detectron2 log: 500 inference batches; 500 panoptic PNGs were persisted. Panoptic metrics from the log: PQ 51.9504, SQ 80.2088, RQ 63.0339; not the mask-fusion comparison metric.
- `inference/retained-output-manifest.json` declares 500 PNGs and 500 hashes. Root recomputation matched all 500 PNG hashes, all sample image-ID filenames, the 500 prediction image IDs, the 500 annotations, and prediction JSON SHA-256. Config and runner metadata hashes matched.
- The manifest console-log SHA-256 matched `models/experiments/eovsam/output/coco500-mask-fusion-run2.log`. The separate convenience `run2/log.txt` copy has a different hash and should not be used for that manifest claim.
- `gpu-samples.csv`: 72 device-level one-second samples; memory used min 1,262 MiB, max 14,411 MiB, mean 9,804.125 MiB; utilization min 0%, max 100%, mean 53.181%. Report this as whole-device sampled memory, not process-specific allocation.

## Fusion scoring

Scoring run `coco500-mask-fusion-score-run4` on the fixed COCO val2017 500-image subset (seed 42, manifest SHA-256 `512abb4eabfcadf8ee4818a1dfb2beaef4763679bbf127119ddbdc7b4348af62`; 3,245 GT instances, 80 categories):

- 49,784 detections; 3,298 same-class thing matches at mask IoU >=0.50; 4,124 EOVSAM thing segments; 1,569 stuff segments; mean matched predicted-mask IoU 0.797084; zero invalid YOLOE masks and zero EOVSAM image errors.
- Mask AP / AP50 / AP75: baseline `0.367779519 / 0.586910637 / 0.389058637`; replace `0.361840461 / 0.585248578 / 0.374229757`; union `0.369562404 / 0.587784843 / 0.392528405`. Box AP is `0.444025009` for all arms. Baseline reproduced target within `1e-6`.
- An independent CPU COCOeval recomputation verified row/image-ID counts, identical non-mask fields, and main mask AP/AP50/AP75 for all arms. Runtime 8.40s; peak RSS 560,496 KiB.
- The stored score-run4 metrics for all three arms are independently reproduced when each persisted result row is projected exactly to `image_id`, `category_id`, `segmentation`, and `score` before `pycocotools.COCO.loadRes()`. This selects the segmentation branch, which derives detection area from mask RLE. All six stored mask metrics (AP, AP50, AP75, AP-small, AP-medium, AP-large) match with maximum absolute delta 0.0. The earlier mismatching recomputation passed rows containing both `bbox` and `segmentation`, which selected the bbox-result branch and used box area instead. This verifies scoring from persisted predictions; raw-pixel fusion reconstruction remains unavailable because run2's raw inference directory was removed. Do not claim map association, VQA, or production-detector acceptance.
- Fusion build 39.204s, scoring 15.212s, input verification 1.097s; total pre-promotion 57.296s, promotion 0.6s per log, overall 59.495s. CPU scoring; original scorer peak RSS not measured.

## Fusion scorer failure history

- Score attempt v5 failed on channel order.
- Score attempt v6 reproduced the primary baseline and arms but failed while serializing NumPy int64 parameter IDs.
- Score attempt v7 corrected the serialization issue and atomically promoted the scored arrays and metrics.

These were scorer implementation failures, not negative model results.

## Claim limits

Fixed-subset detector mask-fusion comparison only. It does not validate ConceptGraphs mapping, map association, VQA, or production-detector acceptance.

## Outcome at the earlier run2 attempt: pre-failure baseline

EOVSAM source and weights are installed, Detectron2 built successfully, and the frozen COCO500 preflights passed. The initial bounded CUDA attempt loaded the checkpoint and initialized a 500-image loader, then failed on its first batch with `KeyError: 'lvis_v1_val'`. No prediction or COCO score was produced from that attempt. This is an adapter/runtime failure, not a negative model result. A subsequent retained run2 completed all 500 batches, producing the artifacts documented above.

## Installed implementation and weights

- Official EOVSAM repository: `models/experiments/eovsam/repo`, commit `3450761453da50c0f6a689516b03e6aed826c926`.
- EOVSAM HF checkpoint: revision `51a9003a32aa87254675e492f94087eaa55c4f6b`, 7,694,435,929 bytes, SHA-256 `717ee1cf425396945fb38dbc6977262e115daf80230db4f5cf657004732ebda4`.
- SAM3 dependency checkpoint: local HF snapshot, 3,450,062,241 bytes, SHA-256 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.
- Environment: Python 3.10.12; PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, torchaudio 2.7.1+cu128, Transformers 5.0.0, Detectron2 0.6.
- Detectron2 source: official commit `247691a1c292654867deb35cabb27f18d75a1363`; built with CUDA 12.8 and `TORCH_CUDA_ARCH_LIST=12.0` for the RTX 5090. Build completed successfully.

## Dataset and preflight

The frozen input is 500 COCO val2017 panoptic samples, selected by seed 42 from the shared detector evaluation set. Sample manifest SHA-256: `512abb4eabfcadf8ee4818a1dfb2beaef4763679bbf127119ddbdc7b4348af62`. All extracted panoptic masks and category mappings validated. The Detectron2 panoptic evaluator's thing/stuff ID round trips passed. Preflight metadata is in `models/experiments/eovsam/adapter/run-manifest.json`.

The adapter now gates on its own authentic val inputs and does not require or create COCO train files. A first preflight found a manifest-variable shadowing bug in the local registration adapter; the variable was renamed and subsequent assets/evaluator preflights passed.

After installing `decord==0.6.0` and `psutil==7.2.2` in the isolated Python 3.10.12 environment, the SAM3 dataset import preflight passed. An early authorized CUDA run loaded the EOVSAM checkpoint and began inference on 500 batches. The first input's COCO path did not match EOVSAM's filename heuristics; `get_dataname()` fell back to `lvis_v1_val`, while the registered evaluator metadata is `eovsam_coco_val2017_500_seed42`. The model then raised `KeyError: 'lvis_v1_val'` before producing a prediction. Process exit code was 1. The visible Detectron2 log spans 03:00:44 to 03:04:00 local time through the first-batch failure; no dedicated end-to-end timer was recorded. Peak GPU memory was not sampled. After cleanup the controller listed no loaded model or session and `nvidia-smi` showed no compute process. A corrected run2 completed 500 batches successfully.

## Attempt history and failure

Earlier attempts with Torch 2.5.1+cu124 could not build Detectron2 against the host CUDA compiler or SM 12.0. The newly accepted bounded attempt installed the PyTorch cu128 stack and built the official Detectron2 source successfully.

The first evaluation launch used the project root as CWD and failed because upstream code opens `./maft/data/datasets/coco_panoptic_with_prompt_eng.txt`. Running from the EOVSAM repository fixed that working-directory issue. A later import failed on missing `decord`; the bounded dependency diagnosis installed `decord==0.6.0` and `psutil==7.2.2`, after which the prescribed SAM3 import preflight passed. The early authorized CUDA attempt then reached checkpoint loading and failed on the dataset-name fallback described above. It processed 0/500 images to completion. No predictions, AP metrics, or peak-memory measurement exist from that attempt. A subsequent retained run2 completed all 500 batches successfully.

## Earlier fusion protocol plan and disposition

Fusion was to remain disabled by default and use the shared 500-image COCO val split. Predictions would be matched one-to-one by category and mask overlap. The initial condition would discard unmatched EOVSAM masks, then compare the better individual mask against union masks while tracking instance separation. Planned measures include box/mask AP, IoU, boundary quality, precision/recall, runtime, memory, and error counts.

The early attempt failed before producing a fusion result. The retained run2 completed successfully, and fusion scoring has been performed (see above). Run5 later scored the panoptic model on the fixed 500-image subset, but did not test EOVSAM mask fusion or ConceptGraphs mapping. Do not treat that score as a production replacement decision; core map-level acceptance remains separate.

## Artifacts

- Environment: `models/experiments/envs/eovsam-cu128-attempt1/`.
- Adapter and frozen data: `models/experiments/eovsam/adapter/`.
- Official EOVSAM and Detectron2 source: `models/experiments/eovsam/repo/` and `models/experiments/eovsam/detectron2-repo/`.
- EOVSAM checkpoint is cached at the HF_HOME snapshot recorded above and was loaded during run2; no copy was made into the repository.
- Run2 inference output: `models/experiments/eovsam/output/coco500-mask-fusion-run2/`.
- Run status and prior attempt details: [`eovsam_run_status_20261115.md`](eovsam_run_status_20261115.md).
