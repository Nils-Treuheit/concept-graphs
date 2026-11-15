# YOLOE-26 detection-only L/M checkpoint status

## Requested comparison

The requested candidates are pretrained YOLOE-26-L and YOLOE-26-M detection-only checkpoints for the FG-CLIP2 padded-proposal path. No benchmark score is reported for these exact candidates because their checkpoints are unavailable.

## Checkpoint evidence

The official Ultralytics YOLOE-26 documentation lists released checkpoints with `-seg` and `-seg-pf` suffixes. The official asset resolver likewise generates only those two suffixes for YOLOE-26. It does not list `yoloe-26l.pt` or `yoloe-26m.pt` as assets. Neither unsuffixed file is present under this project's `weights/` or `models/` directories.

Available official L/M weights are segmentation checkpoints:

| Size | Official checkpoint | Status |
|---|---|---|
| L | [`yoloe-26l-seg.pt`](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26l-seg.pt) | Present under `weights/`; copied into `models/` for this task |
| M | [`yoloe-26m-seg.pt`](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26m-seg.pt) | Present under `weights/`; copied into `models/` for this task |

The copied model files are local working assets and are excluded from this report commit. Their SHA-256 values are:

| File | Size | SHA-256 |
|---|---:|---|
| `models/yoloe-26l-seg.pt` | 79,417,413 bytes | `a612d2d505f24e14d87ec82d688b823b6cb600646664f16125ce6c84ce360da9` |
| `models/yoloe-26m-seg.pt` | 70,443,185 bytes | `585f5ec9028fd358035da8d860c27c56be285a795cba2076fba536a4391c2c83` |

These copies duplicate the files already in `weights/`; no new download was required. The segmentation checkpoints include mask and prompt components, so their architecture is not equivalent to a detect-only checkpoint.

## What can be measured with available weights

The existing detector benchmark records box AP@0.5 for the segmentation checkpoints, and the existing proposal pipeline can discard masks and produce rectangular map masks while FG-CLIP2 embeds padded boxes. A rerun of that path with `models/yoloe-26l-seg.pt` and `models/yoloe-26m-seg.pt` would measure box-only outputs from segmentation weights. Such a result must not be labeled as a detection-only checkpoint benchmark. The exact requested pretrained detect-only comparison remains unavailable until matching official weights are released or supplied.

## Evidence sources

- [Official YOLOE documentation](https://docs.ultralytics.com/models/yoloe/): released YOLOE-26 checkpoints are segmentation variants.
- [Official Ultralytics asset resolver](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/downloads.py): YOLOE-26 asset names are generated with `-seg` and `-seg-pf` suffixes.
- Local original files: `weights/yoloe-26l-seg.pt`, `weights/yoloe-26m-seg.pt`.
- Local copies: `models/yoloe-26l-seg.pt`, `models/yoloe-26m-seg.pt`.
