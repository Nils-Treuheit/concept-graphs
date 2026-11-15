# EOVSAM fusion feasibility report

**Status date:** 2026-11-15
**Benchmark:** Not run. A fresh bounded dependency attempt is now justified by a distinct official PyTorch/Detectron2 compatibility path, but the required host release gate has not yet passed. No model score is claimed.

## Finding

The experiment assets are now installed. The official EOVSAM repository is at `models/experiments/eovsam/repo` (commit `3450761453da50c0f6a689516b03e6aed826c926`); checkpoint `HaominPeng/EOVSAM` is cached in `HF_HOME` at revision `51a9003a32aa87254675e492f94087eaa55c4f6b` (7,694,435,929 bytes; SHA-256 `717ee1cf425396945fb38dbc6977262e115daf80230db4f5cf657004732ebda4`). The local SAM3 dependency is cached as well. These installs do not establish runtime compatibility or benchmark quality.

The preserved isolated Python 3.10.12 environment contains Torch 2.5.1+cu124 and repository requirements. Detectron2 v0.6 did not build: CUDA 13.2 mismatched Torch, while the CUDA 12.8 retry rejected host architecture 12.0. A fresh diagnosis was independently accepted for exactly one new attempt using PyTorch 2.7.1, torchvision 0.22.1 and torchaudio 2.7.1 from the cu128 index with official Detectron2 source and CUDA 12.8. PyTorch 2.7 adds Blackwell support and ships cu128 wheels; Detectron2 advises matching Torch/torchvision/CUDA and building from source when GPU architecture support is missing ([PyTorch 2.7 release](https://pytorch.org/blog/pytorch-2-7/), [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/), [Detectron2 install guide](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md)). The new environment is not yet installed, and no compatibility is claimed. See [`eovsam_run_status_20261115.md`](eovsam_run_status_20261115.md) for the retry plan and failure evidence.

## Evaluation protocol

Keep fusion disabled by default and write outputs under a unique experiment suffix. On the same held-out labeled frames as the accepted YOLOE baseline, prompt EOVSAM with the unique categories in YOLOE detections. Match predictions one-to-one by category and mask overlap. For high-agreement pairs, compare the better individual mask with the union mask. Make unmatched-mask handling and confidence weighting explicit config options. First precision-oriented condition discards unmatched EOVSAM masks. Verify instance separation before evaluating fusion.

- **Dataset/split:** labeled held-out frames from the accepted detector benchmark; freeze split and manifest hash before execution. No split was executed.
- **Metrics:** mask IoU and boundary quality, class precision/recall, instance-separation errors, projected point-cloud contamination, and map association if instance-ID ground truth exists.
- **Runtime/resources:** per-frame and total wall time, peak allocated/reserved GPU memory, model residency, failed prompts, and OOMs. **Not measured.**
- **Failures/blocker:** Detectron2 environment unresolved; the accepted fresh attempt is pending a successful host release/controller gate. The frozen 500-panoptic-PNG extraction is complete, but evaluator mapping round-trip and inference remain pending. No model loaded and no predictions exist.
- **Artifacts:** source/checkpoint and passing COCO500 source-data audit are installed. The adapter and run report are tracked separately. No masks or benchmark scores exist.

Protocol source: `MODERNIZED_PIPELINE_IMPLEMENTATION_PLAN.md`, “Later experiment A: optional EOVSAM fusion.”
