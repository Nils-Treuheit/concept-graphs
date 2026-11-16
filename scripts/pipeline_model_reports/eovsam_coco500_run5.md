# EOVSAM COCO500 panoptic run5

**Result:** EOVSAM+SAM3 scored **PQ 51.9266 / SQ 80.2159 / RQ 62.9975** on the frozen 500-image COCO val2017 subset. The corrected run processed all 500 images and exited successfully.

## Configuration

- Official EOVSAM source: `models/experiments/eovsam/repo`, commit `3450761453da50c0f6a689516b03e6aed826c926`.
- EOVSAM checkpoint revision `51a9003a32aa87254675e492f94087eaa55c4f6b`, 7,694,435,929 bytes, SHA-256 `717ee1cf425396945fb38dbc6977262e115daf80230db4f5cf657004732ebda4`.
- SAM3 checkpoint revision `3c879f39826c281e95690f02c7821c4de09afae7`, SHA-256 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.
- Detectron2 commit `247691a1c292654867deb35cabb27f18d75a1363`; panopticapi commit `7bb4655548f98f3fedc07bf37e9040a992b054b0`.
- Split: 500 COCO val2017 images, seed 42, 133 categories. Manifest SHA-256 `512abb4eabfcadf8ee4818a1dfb2beaef4763679bbf127119ddbdc7b4348af62`.
- Input adapter config SHA-256 `5679481d660827cf912a7ca10f8fd6603aa1bce004ebe828fe296497ed2137e8`. Generated run5 config SHA-256 `673b299c3e33259c8129705e27168bae5c3d4a7c8ebc5ebd0942e4dad5815f9d`; generated run4 config differed only in `OUTPUT_DIR`.
- Python 3.10.12, PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, Detectron2 0.6, CUDA 12.8.
- Hardware: RTX 5090, one GPU. The exact command and environment are recorded in the runner metadata.

## Results

| Metric | All | Things | Stuff |
|---|---:|---:|---:|
| PQ | 51.9266 | 57.9940 | 42.8828 |
| SQ | 80.2159 | 80.4961 | 79.7983 |
| RQ | 62.9975 | 69.5764 | 53.1911 |

The evaluator returned `evaluator_roundtrip_passed`. Root recomputation confirmed 500 unique prediction image IDs and 500 annotation IDs matching the frozen GT subset. Category records matched exactly. No inference traceback or per-image failure occurred.

## Runtime and resources

- Total runner wall time: 79.91 seconds.
- Detectron2 inference: 42.3313 seconds, including 38 seconds pure compute.
- Peak host RSS: 9,493,048 KiB.
- GPU sampling: 81 one-second samples; peak total GPU memory 14,288 MiB; peak utilization 100%.
- After cleanup: RTX 5090 at 1,140 MiB / 32,607 MiB and 0% utilization; no compute applications.
- `release-local` succeeded after the run; controller and OpenCode sessions were idle.

## Scoring repair and run-to-run comparison

Run4 produced predictions for the frozen 500 images, but PanopticAPI received the full 5,000-image GT annotation set. It stopped at unsampled image 139, so run4 has no valid PQ score. Run5 performed a fresh inference with the same model, checkpoint, sample, evaluator, and normalized config; only `OUTPUT_DIR` differed in the generated config. The evaluator GT JSON was filtered to the frozen 500 IDs. Run4 predictions were not reused.

The run4 and run5 prediction JSON SHA-256 values are `40db2e946af22f2b23d76ba64087ab528dfc0fb095480a2559e407765cc7ec3c` and `e92404977fe2e1aa8fe4997b7ed30c7aa523ea12edd8b25e06de90fc5dc5edc9`. Both contain the same ID set and category records; five per-image annotation rows differ, with segment counts changing for three images. Array order also differs. Since run4 did not score, no quality difference can be claimed. These are the run5 scores for the fresh run5 predictions only.

The adapter manifest was restored to its frozen pre-run SHA-256 `0075b38e7afb6a1b3569891291b8989041c65ffce8144800dbdcfa3a4788a3fc`. The produced manifest SHA-256 is `d167df5eb52a2f9de7ca19ea3a57c938e56745f58d821c361e92744b3aff7c95`.

## Artifacts

- [Run5 output folder](../../models/experiments/eovsam/output/coco500-panoptic-run5/)
- [Full console log](../../models/experiments/eovsam/output/coco500-panoptic-run5/console.log)
- [Structured run log](../../models/experiments/eovsam/output/coco500-panoptic-run5/log.txt)
- [Predictions JSON](../../models/experiments/eovsam/output/coco500-panoptic-run5/inference/predictions.json)
- [GPU samples](../../models/experiments/eovsam/output/coco500-panoptic-run5/gpu-samples.csv)
- Root recomputation: `.agentic_workflow/workspace/pipeline-experiments/eovsam-run5-root-recompute-v4.json`.

This is a fixed-subset panoptic evaluation, not a full COCO val2017 result. It does not evaluate ConceptGraphs relation propagation, map association, graph recovery, or caption-to-track consistency. It does not establish EOVSAM as the production detector. Earlier setup and adapter failures remain documented in [the EOVSAM fusion experiment history](eovsam_fusion.md).
