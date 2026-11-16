# EOVSAM setup and failure history

**History scope:** this report preserves setup and pre-run5 failures. The final COCO500 score is in [the run5 report](eovsam_coco500_run5.md).

**Outcome at the earlier run2 attempt:** EOVSAM source and weights are installed, Detectron2 built successfully, and the frozen COCO500 preflights passed. The bounded CUDA attempt loaded the checkpoint and initialized a 500-image loader, then failed on its first batch with `KeyError: 'lvis_v1_val'`. No prediction or COCO score was produced. This is an adapter/runtime failure, not a negative model result.

## Installed implementation and weights

- Official EOVSAM repository: `models/experiments/eovsam/repo`, commit `3450761453da50c0f6a689516b03e6aed826c926`.
- EOVSAM HF checkpoint: revision `51a9003a32aa87254675e492f94087eaa55c4f6b`, 7,694,435,929 bytes, SHA-256 `717ee1cf425396945fb38dbc6977262e115daf80230db4f5cf657004732ebda4`.
- SAM3 dependency checkpoint: local HF snapshot, 3,450,062,241 bytes, SHA-256 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.
- Environment: Python 3.10.12; PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, torchaudio 2.7.1+cu128, Transformers 5.0.0, Detectron2 0.6.
- Detectron2 source: official commit `247691a1c292654867deb35cabb27f18d75a1363`; built with CUDA 12.8 and `TORCH_CUDA_ARCH_LIST=12.0` for the RTX 5090. Build completed successfully.

## Dataset and preflight

The frozen input is 500 COCO val2017 panoptic samples, selected by seed 42 from the shared detector evaluation set. Sample manifest SHA-256: `512abb4eabfcadf8ee4818a1dfb2beaef4763679bbf127119ddbdc7b4348af62`. All extracted panoptic masks and category mappings validated. The Detectron2 panoptic evaluator's thing/stuff ID round trips passed. Preflight metadata is in `models/experiments/eovsam/adapter/run-manifest.json`.

The adapter now gates on its own authentic val inputs and does not require or create COCO train files. A first preflight found a manifest-variable shadowing bug in the local registration adapter; the variable was renamed and subsequent assets/evaluator preflights passed.

After installing `decord==0.6.0` and `psutil==7.2.2` in the isolated Python 3.10.12 environment, the SAM3 dataset import preflight passed. A 2026-11-16 real CUDA run loaded the EOVSAM checkpoint and began inference on 500 batches. The first input's COCO path did not match EOVSAM's filename heuristics; `get_dataname()` fell back to `lvis_v1_val`, while the registered evaluator metadata is `eovsam_coco_val2017_500_seed42`. The model then raised `KeyError: 'lvis_v1_val'` before producing a prediction. Process exit code was 1. The visible Detectron2 log spans 03:00:44 to 03:04:00 local time through the first-batch failure; no dedicated end-to-end timer was recorded. Peak GPU memory was not sampled. After cleanup the controller listed no loaded model or session and `nvidia-smi` showed no compute process.

## Attempt history and failure

Earlier attempts with Torch 2.5.1+cu124 could not build Detectron2 against the host CUDA compiler or SM 12.0. The newly accepted bounded attempt installed the PyTorch cu128 stack and built the official Detectron2 source successfully.

The first evaluation launch used the project root as CWD and failed because upstream code opens `./maft/data/datasets/coco_panoptic_with_prompt_eng.txt`. Running from the EOVSAM repository fixed that working-directory issue. A later import failed on missing `decord`; the bounded dependency diagnosis installed `decord==0.6.0` and `psutil==7.2.2`, after which the prescribed SAM3 import preflight passed. The single authorized CUDA attempt then reached checkpoint loading and failed on the dataset-name fallback described above. It processed 0/500 images to completion. No predictions, AP metrics, or peak-memory measurement exist.

## Earlier fusion protocol plan and disposition

Fusion was to remain disabled by default and use the shared 500-image COCO val split. Predictions would be matched one-to-one by category and mask overlap. The initial condition would discard unmatched EOVSAM masks, then compare the better individual mask against union masks while tracking instance separation. Planned measures include box/mask AP, IoU, boundary quality, precision/recall, runtime, memory, and error counts.

The earlier run2 inference did not complete an image, so it yielded no fusion result. Run5 later scored the panoptic model on the fixed 500-image subset, but did not test EOVSAM mask fusion or ConceptGraphs mapping. Do not treat that score as a production replacement decision; core map-level acceptance remains separate.

## Artifacts

- Environment: `models/experiments/envs/eovsam-cu128-attempt1/`.
- Adapter and frozen data: `models/experiments/eovsam/adapter/`.
- Official EOVSAM and Detectron2 source: `models/experiments/eovsam/repo/` and `models/experiments/eovsam/detectron2-repo/`.
- EOVSAM checkpoint is cached at the HF_HOME snapshot recorded above and was loaded during run2; no copy was made into the repository.
- Run status and prior attempt details: [`eovsam_run_status_20261115.md`](eovsam_run_status_20261115.md).
