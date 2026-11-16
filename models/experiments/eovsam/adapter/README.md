# EOVSAM COCO500 panoptic adapter

This experiment adapter evaluates the installed EOVSAM checkpoint on the
frozen 500-image COCO val2017 subset. It keeps the upstream repository and
annotation JSON unchanged. The base config is the official
`repo/configs/eval_panoptic.yaml`; the overlay changes the validation dataset
and disables semantic scoring while preserving panoptic scoring.

## Status

The frozen 500-image val subset and category round-trip preflights pass. The
isolated CUDA 12.8 environment includes Detectron2 0.6, `decord==0.6.0`, and
`psutil==7.2.2`; its SAM3 import preflight passes. The 2026-11-16 CUDA run
loaded the checkpoint but stopped on the first inference batch with
`KeyError: 'lvis_v1_val'`. EOVSAM inferred a dataset key from the image path
instead of receiving the adapter's registered dataset name. No prediction or
score was produced. The metadata diagnosis and v2 correction contract passed
independent review. The patch-only implementation is pending, and any run3
requires source review plus a fresh host release/controller gate.

The adapter loads the exact upstream `openseg_classes.py` file directly to
obtain prompt categories and reproduces the ordered maps from upstream
`register_coco_panoptic_annos_semseg.get_metadata()` without importing the
`maft` package registration hooks.

After a new diagnosis and contract are independently accepted, run any
authorized retry from the EOVSAM repository directory and use a new output
directory. The failed run2 artifacts are preserved at
`models/experiments/eovsam/output/coco500-panoptic-run2`.

```bash
cd models/experiments/eovsam/repo
/mnt/HDD1/Project_Code/concept-graphs/models/experiments/envs/eovsam-cu128-attempt1/bin/python /mnt/HDD1/Project_Code/concept-graphs/models/experiments/eovsam/adapter/evaluate.py run --num-gpus 1 --eval-only --config-file /mnt/HDD1/Project_Code/concept-graphs/models/experiments/eovsam/adapter/eovsam_coco500.yaml OUTPUT_DIR /mnt/HDD1/Project_Code/concept-graphs/models/experiments/eovsam/output/coco500-panoptic-run3 MODEL.WEIGHTS /mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub/models--HaominPeng--EOVSAM/snapshots/51a9003a32aa87254675e492f94087eaa55c4f6b/eovsam.pth
```

The checkpoint path points to the existing HF_HOME snapshot; no 7.7 GB copy
is made. If this cache path changes, resolve it from `HF_HOME` and the snapshot
revision recorded in the experiment report. The run manifest is written to
`run-manifest.json`; output metrics and logs belong under a unique `OUTPUT_DIR`.
