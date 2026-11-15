# SAM 3.1 mask-refinement experiment

**Status:** the official source and checkpoint are installed, and a 100-item paired COCO validation proxy is frozen. Inference has not run. The GPU release attempt made from the sandbox failed; no model was started.

## Installed model and implementation

- **Repository:** `models/experiments/sam3/repo`, official commit `eef6839bcfc3458dd52ea80535dc6bd62aea86b1`.
- **Checkpoint:** Hugging Face cache snapshot `models--facebook--sam3.1/snapshots/daa63191845a41281374e725f4c9e51c7a824460/sam3.1_multiplex.pt`, 3,502,755,717 bytes, SHA-256 `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`.
- **Environment:** Python 3.12.13, PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128, NumPy 1.26.4, timm 1.0.31, einops 0.8.2, pycocotools 2.0.11, psutil 7.2.2, editable official SAM3 source.
- **Runner:** `models/experiments/sam3/adapter/benchmark_template_refinement.py`; it imports Torch/SAM only after manifest, asset, source revision, and checkpoint checks. The runner requires CUDA and has no CPU fallback.

## Frozen paired evaluation input

This is a conditional segmentation refinement proxy, not map-template evaluation, detector AP, or map acceptance. It uses the same fixed 500-image COCO val2017 sample as the four-arm YOLOE report. From the 487 images with eligible same-category matches, 100 unique images were selected with seed 31. Each image contributes one greedily matched detection/ground-truth pair with bbox IoU at least 0.5. The selection does not use mask IoU. SAM receives the YOLOE-predicted box, while the unmodified YOLOE predicted mask is the baseline and the COCO instance mask is the target.

- **Manifest:** `models/experiments/sam3/data/coco-yoloe-l-seg-matched100-v1/manifest.json`.
- **Manifest SHA-256:** `bf7dafe3e4de7ee7e1d8b58589594bb7ce0901157d0ade29b35334bb29237654`.
- **Input counts:** 100 images, 100 detector prompts, 100 YOLOE masks, 100 ground-truth masks. All 300 referenced assets were SHA-256 verified; no asset path traverses a symlink.
- **Source data:** YOLOE-26-L-seg prediction run `scripts/pipeline_model_reports/data/yoloe26/rerun_20261115_1912/yoloe-26l-seg.json`; exact COCO subset labels are in its scorer folder.
- **Preparation utility:** `models/experiments/sam3/adapter/prepare_coco_refinement_manifest.py`.

## Preflight blocker

The SAM runner resolves the user-supplied checkpoint path before checking that its filename is the official `sam3.1_multiplex.pt`. Hugging Face snapshots expose that filename as a symlink to a hash-named blob. The runner consequently sees the blob filename and rejects the valid cached checkpoint. The required behavior is to validate the visible official snapshot path while hashing and loading its resolved file. This narrow runner fix is pending.

The release command attempted at 2026-11-15 19:45 UTC returned `deep_cleanup=cleanup_failed` and `fast_cleanup=cleanup_failed` when invoked in sandbox context. A subsequent host-side read-only controller check showed no loaded model or active session and controller-reported aggregate device use of 1,115 MiB. This was the required model-control status check, not a direct `nvidia-smi` or model-specific residency probe. The host release retry is scheduled no earlier than 20:45 UTC. No SAM import, checkpoint load, or inference occurred.

## Planned metrics and unmeasured fields

The paired primary metric is mean mask IoU against the fixed COCO ground-truth mask, before and after refinement, with paired delta. The runner also records per-item prompt latency, SAM score, candidate count, and selected-box overlap. **Mask IoU, runtime, throughput, memory, and failure rates are not measured yet.** The paired set conditions on a correctly localized same-category detection; it cannot measure SAM's effect on detector recall or full COCO AP.

## Artifacts

- Frozen inputs: `models/experiments/sam3/data/coco-yoloe-l-seg-matched100-v1/`.
- Runner and protocol: `models/experiments/sam3/adapter/benchmark_template_refinement.py` and `models/experiments/sam3/adapter/README.md`.
- Model repository and checkpoint: paths above; additional asset inventory is in `scripts/pipeline_model_reports/data/optional-assets/asset-hashes.json`.
- **Result files:** not produced; inference is blocked pending runner preflight correction and successful host release/controller gate.
