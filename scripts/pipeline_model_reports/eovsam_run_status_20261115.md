# EOVSAM COCO val2017 run status: prior attempts through run2

> **Latest result (2026-11-16):** Run5 completed the corrected COCO500 panoptic evaluation with PQ 51.9266 / SQ 80.2159 / RQ 62.9975. The earlier `lvis_v1_val` batch failure and run4 GT-scoring failure remain historical attempts. See [the run5 report](eovsam_coco500_run5.md).


**Prior run2 attempt: CUDA failed before the first prediction.** The repository and checkpoint are installed, the frozen 500-image COCO panoptic manifest and evaluator/category preflight pass, and the isolated environment now passes the SAM3 import preflight with `decord==0.6.0` and `psutil==7.2.2`. The 2026-11-16 run loaded the EOVSAM checkpoint, started the 500-image loader, then failed on the first batch because EOVSAM inferred `lvis_v1_val` from the image path instead of using the registered validation dataset name. No predictions or COCO metrics were produced.

## Installed assets

- Repository: `models/experiments/eovsam/repo`, commit `3450761453da50c0f6a689516b03e6aed826c926`.
- Checkpoint: `HF_HOME` snapshot of `HaominPeng/EOVSAM/eovsam.pth`, 7,694,435,929 bytes, SHA-256 `717ee1cf425396945fb38dbc6977262e115daf80230db4f5cf657004732ebda4`.
- SAM3 checkpoint dependency: local `facebook/sam3` snapshot, 3,450,062,241 bytes, SHA-256 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.
- Dataset: COCO val2017 panoptic, 500 selected images, manifest SHA-256 `512abb4eabfcadf8ee4818a1dfb2beaef4763679bbf127119ddbdc7b4348af62`. The accepted data-only run extracted all 500 source PNGs and verified ZIP-member hashes and image dimensions. Extracted-manifest SHA-256: `0eb28b44a543e2748a0262030823aabdacba085250d5a159c7334b5382ed1620`. The corrected Detectron2 evaluator category round-trip passed; its manifest is `models/experiments/eovsam/adapter/run-manifest.json`.

## Environment work and failures

- Created `models/experiments/envs/eovsam` with Python 3.10.12.
- Installed PyTorch 2.5.1+cu124, torchvision 0.20.1+cu124, torchaudio 2.5.1+cu124, and the repository's Python requirements.
- Detectron2 v0.6 installation failed twice. The first build selected host CUDA 13.2, which mismatched Torch CUDA 12.4. The one repair attempt used CUDA 12.8 but rejected detected architecture `12.0` under the pinned Torch/toolchain. These attempts remain exhausted; their logs are preserved.
- The accepted fresh attempt installed Torch `2.7.1+cu128`, torchvision `0.22.1+cu128`, torchaudio `2.7.1+cu128`, and the pinned EOVSAM Python requirements into `models/experiments/envs/eovsam-cu128-attempt1`. CUDA 12.8 was selected explicitly with `CUDA_HOME=/usr/local/cuda-12.8`; the RTX 5090 reported SM 12.0. Official Detectron2 v0.6 built successfully from source commit `247691a1c292654867deb35cabb27f18d75a1363` with `TORCH_CUDA_ARCH_LIST=12.0`.
- The first corrected val-only asset preflight exposed and fixed a local variable shadowing error in `register_coco500.py`. The subsequent asset check passed for all 500 selected images and masks. The Detectron2 panoptic evaluator's thing/stuff category round-trip preflight also passed and wrote `models/experiments/eovsam/adapter/run-manifest.json`.
- The first inference launch used the ConceptGraphs repository as working directory and failed because upstream code opens `./maft/data/datasets/coco_panoptic_with_prompt_eng.txt` relative to the current directory. The retry from the EOVSAM repo directory progressed past this issue but stopped importing `sam3.train.data.sam3_image_dataset` with `ModuleNotFoundError: No module named 'decord'`. A bounded diagnosis installed `decord==0.6.0` and `psutil==7.2.2`; the prescribed import preflight then passed. The one CUDA attempt loaded the checkpoint and started 500 batches, but failed on the first item with `KeyError: 'lvis_v1_val'`. Runtime metadata is in `models/experiments/eovsam/output/coco500-panoptic-run2/`; there are no predictions or metrics. Peak GPU memory was not sampled. Further execution requires a new reviewed diagnosis and contract.

## Outcome of the prior run2 attempt

Run2 was an adapter/runtime failure, not a negative model result. Its first input's dataset identity did not reach EOVSAM, so it produced no score. Run5 later passed the corrected COCO500 panoptic evaluator, as reported above. Run5 does not evaluate fusion in ConceptGraphs mapping; keep EOVSAM optional pending separate map-level acceptance.
