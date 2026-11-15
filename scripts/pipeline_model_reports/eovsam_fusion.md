# EOVSAM fusion experiment

**Outcome:** EOVSAM source and weights are installed, Detectron2 built successfully, and the frozen COCO500 asset/evaluator preflights passed. The bounded inference attempt stopped during upstream imports because `decord` is absent. The checkpoint was not loaded; no score is available. This is a setup blocker, not a negative model result.

## Installed implementation and weights

- Official EOVSAM repository: `models/experiments/eovsam/repo`, commit `3450761453da50c0f6a689516b03e6aed826c926`.
- EOVSAM HF checkpoint: revision `51a9003a32aa87254675e492f94087eaa55c4f6b`, 7,694,435,929 bytes, SHA-256 `717ee1cf425396945fb38dbc6977262e115daf80230db4f5cf657004732ebda4`.
- SAM3 dependency checkpoint: local HF snapshot, 3,450,062,241 bytes, SHA-256 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.
- Environment: Python 3.10.12; PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, torchaudio 2.7.1+cu128, Transformers 5.0.0, Detectron2 0.6.
- Detectron2 source: official commit `247691a1c292654867deb35cabb27f18d75a1363`; built with CUDA 12.8 and `TORCH_CUDA_ARCH_LIST=12.0` for the RTX 5090. Build completed successfully.

## Dataset and preflight

The frozen input is 500 COCO val2017 panoptic samples, selected by seed 42 from the shared detector evaluation set. Sample manifest SHA-256: `512abb4eabfcadf8ee4818a1dfb2beaef4763679bbf127119ddbdc7b4348af62`. All extracted panoptic masks and category mappings validated. The Detectron2 panoptic evaluator's thing/stuff ID round trips passed. Preflight metadata is in `models/experiments/eovsam/adapter/run-manifest.json`.

The adapter now gates on its own authentic val inputs and does not require or create COCO train files. A first preflight found a manifest-variable shadowing bug in the local registration adapter; the variable was renamed and subsequent assets/evaluator preflights passed.

## Attempt history and failure

Earlier attempts with Torch 2.5.1+cu124 could not build Detectron2 against the host CUDA compiler or SM 12.0. The newly accepted bounded attempt installed the PyTorch cu128 stack and built the official Detectron2 source successfully.

The first evaluation launch used the project root as CWD and failed because upstream code opens `./maft/data/datasets/coco_panoptic_with_prompt_eng.txt`. That file exists in the EOVSAM repository, so the retry ran from that repo directory. The retry then failed while importing `sam3.train.data.sam3_image_dataset` with `ModuleNotFoundError: No module named 'decord'`. The authorized fresh attempt is exhausted at this import failure. No model weights loaded; there are no EOVSAM predictions, AP metrics, latency, or inference-memory measurements.

## Intended fusion protocol and disposition

Fusion was to remain disabled by default and use the shared 500-image COCO val split. Predictions would be matched one-to-one by category and mask overlap. The initial condition would discard unmatched EOVSAM masks, then compare the better individual mask against union masks while tracking instance separation. Planned measures include box/mask AP, IoU, boundary quality, precision/recall, runtime, memory, and error counts.

Because inference did not start, no fusion conclusion or production recommendation is supported. Keep EOVSAM disabled. The next attempt requires a separately reviewed diagnosis; this run will not continue installing dependencies beyond the bounded attempt.

## Artifacts

- Environment: `models/experiments/envs/eovsam-cu128-attempt1/`.
- Adapter and frozen data: `models/experiments/eovsam/adapter/`.
- Official EOVSAM and Detectron2 source: `models/experiments/eovsam/repo/` and `models/experiments/eovsam/detectron2-repo/`.
- EOVSAM checkpoint is cached at the HF_HOME snapshot recorded above. The failed launch stopped before loading it; no copy was made into the repository.
- Run status and prior attempt details: [`eovsam_run_status_20261115.md`](eovsam_run_status_20261115.md).
