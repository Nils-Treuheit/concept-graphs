# SAM 3.1 mask-refinement experiment

**Result:** the full 100-item paired experiment completed on the RTX 5090. SAM 3.1 reduced aggregate mask IoU, so template-mask refinement should remain disabled.

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

## Final run

Run 6 used the frozen manifest without changes, SAM repository commit `eef6839bcfc3458dd52ea80535dc6bd62aea86b1`, the checkpoint SHA above, CUDA 0, and explicit PyTorch SDPA (`torch.nn.functional.scaled_dot_product_attention`). The default Flash Attention 3 backend was unavailable because `flash_attn_interface` was not installed. The report therefore applies specifically to the SDPA backend.

| Measure | Result |
|---|---:|
| Prompts processed / failures | 100 / 0 |
| Prompts with selected mask / no candidate | 73 / 27 |
| Baseline mean mask IoU | 0.774529 |
| Refined mean mask IoU | 0.589640 |
| Paired mean IoU change | -0.184888 |
| Median prompt latency | 0.07506 s |
| Model load / full run wall time | 11.611 s / 39.633 s |
| Peak allocated / reserved CUDA memory | 5.64 / 6.09 GiB |
| Whole-device baseline / peak memory | 1,634 / 8,006 MiB |

All 100 rows were scored, including 27 aligned all-zero SAM outputs with no candidate. Row-level CSV recomputation matches the JSON mean delta. The aggregate result is substantially worse even though the selected-mask subset improves slightly; excluding missing predictions would bias the result. The 27 empty outputs are counted as zero IoU.

Earlier attempts are retained in sibling run directories: run 1 stopped at repository ownership validation; run 2 exposed 64 derived RoPE buffers rejected by strict checkpoint loading; run 3 exposed an upstream multiplex-session API mismatch; run 4 lacked Flash Attention 3; run 5 completed only 73 rows before empty-output handling was fixed. These attempts are diagnostic history, not benchmark scores.

The paired set conditions on a correctly localized same-category detection. It cannot measure detector recall, full COCO AP, or full pipeline map acceptance. Runtime and memory describe this single 100-prompt run and SDPA configuration.

## Artifacts

- Frozen inputs: `models/experiments/sam3/data/coco-yoloe-l-seg-matched100-v1/`.
- Runner and protocol: `models/experiments/sam3/adapter/benchmark_template_refinement.py` and `models/experiments/sam3/adapter/README.md`.
- Model repository and checkpoint: paths above; additional asset inventory is in `scripts/pipeline_model_reports/data/optional-assets/asset-hashes.json`.
- **Final result:** `models/experiments/sam3/results/coco-yoloe-l-seg-matched100-20261115-run6-sdpa/results.json` and `per_object.csv`; per-item masks are in the sibling `masks/` directory. Run 1–5 logs and partial outputs remain in the parent `results/` folder.
