# EOVSAM COCO val2017 run status

**Benchmark state: one fresh dependency attempt is authorized but not started; inference remains blocked.** The repository and checkpoint are installed, and the frozen 500-image COCO panoptic manifest passes its integrity audit. No EOVSAM model was loaded, and no benchmark predictions or metrics were produced.

## Installed assets

- Repository: `models/experiments/eovsam/repo`, commit `3450761453da50c0f6a689516b03e6aed826c926`.
- Checkpoint: `HF_HOME` snapshot of `HaominPeng/EOVSAM/eovsam.pth`, 7,694,435,929 bytes, SHA-256 `717ee1cf425396945fb38dbc6977262e115daf80230db4f5cf657004732ebda4`.
- SAM3 checkpoint dependency: local `facebook/sam3` snapshot, 3,450,062,241 bytes, SHA-256 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.
- Dataset: COCO val2017 panoptic, 500 selected images, manifest SHA-256 `512abb4eabfcadf8ee4818a1dfb2beaef4763679bbf127119ddbdc7b4348af62`. The accepted data-only run extracted all 500 source PNGs and verified ZIP-member hashes and image dimensions. Extracted-manifest SHA-256: `0eb28b44a543e2748a0262030823aabdacba085250d5a159c7334b5382ed1620`. The v6 evaluator category round-trip remains unrun.

## Environment work and failures

- Created `models/experiments/envs/eovsam` with Python 3.10.12.
- Installed PyTorch 2.5.1+cu124, torchvision 0.20.1+cu124, torchaudio 2.5.1+cu124, and the repository's Python requirements.
- Detectron2 v0.6 installation failed twice. The first build selected host CUDA 13.2, which mismatched Torch CUDA 12.4. The one repair attempt used CUDA 12.8 but rejected detected architecture `12.0` under the pinned Torch/toolchain. These attempts remain exhausted; their logs are preserved.
- A fresh root diagnosis was independently accepted for one distinct attempt using the official PyTorch 2.7.1/torchvision 0.22.1/torchaudio 2.7.1 cu128 package set and official Detectron2 source with CUDA 12.8. PyTorch 2.7 adds Blackwell and cu128 support. The existing val-only config does not load the registered train datasets; the adapter's train-assets guard will be narrowed to authentic validation inputs without creating placeholders. This route is unproven and must stop at the next build/import failure. Diagnosis and independent adjudication: `.agentic_workflow/workspace/pipeline-experiments/eovsam-fresh-diagnosis.md` and `eovsam-fresh-diagnosis-adjudication.md`.
- At 2026-11-15 19:45 UTC, `release-local` invoked from sandbox context returned `deep_cleanup=cleanup_failed` and `fast_cleanup=cleanup_failed`; no GPU work followed. A host-side controller status read at 19:49 UTC showed no loaded models or active sessions and 1,115 MiB aggregate device use. The host release retry is no earlier than 20:45 UTC. The earlier SAM import-gate incident is documented in `sam3-import-gate-incident-20261115.md` and does not indicate EOVSAM inference. No EOVSAM model/checkpoint was loaded and no inference ran.

## Next eligible action

**Disposition:** assets and dataset preparation complete; one newly accepted CUDA 12.8/PyTorch cu128 dependency attempt remains. First pass the successful host release/controller gate. Freeze official Detectron2 source commit and package versions before building in a fresh environment. Stop on any second compatibility failure; do not lower the target architecture or use CPU. Then run the actual val evaluator preflight and, only if it passes, the 500-image CUDA evaluation. Keep official ADE20K evaluation out of this COCO-val result.
