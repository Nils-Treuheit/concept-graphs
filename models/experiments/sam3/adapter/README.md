# SAM 3.1 template-mask refinement benchmark

This adapter evaluates the official SAM 3.1 multiplex video predictor on fixed
representative object views. Each source image becomes a lossless, one-frame
PNG directory. A fixed source-pixel box is submitted as a positive normalized
XYWH visual prompt at frame zero. The selected SAM mask is compared with the
baseline template mask and independent ground-truth mask.

## Frozen manifest

The UTF-8 JSON manifest uses schema `sam31-template-refinement-v1`:

```json
{
  "schema": "sam31-template-refinement-v1",
  "dataset": "<dataset-name>",
  "split": "<held-out split>",
  "sam3_repo_commit": "eef6839bcfc3458dd52ea80535dc6bd62aea86b1",
  "items": [
    {
      "sample_id": "unique-safe-id",
      "image": "images/example.jpg",
      "bbox_xyxy": [12, 24, 180, 220],
      "base_mask": "base_masks/example.png",
      "ground_truth_mask": "gt_masks/example.png",
      "sha256": {
        "image": "<64 lowercase hex characters>",
        "base_mask": "<64 lowercase hex characters>",
        "ground_truth_mask": "<64 lowercase hex characters>"
      },
      "prompt_source": "fixed_template_box"
    }
  ]
}
```

Dataset and split must be nonempty. The SAM repository revision must match the
installed source. Each asset path must be an unambiguous relative path under the
manifest directory. Absolute paths, parent traversal and symlink traversal are
rejected. Each image and mask needs a SHA-256 in the manifest; every digest is
verified before importing Torch. Masks may be binary PNGs or numeric `.npy`
arrays. Nonzero values are foreground. Masks must match the source-image
dimensions. Boxes use source pixel coordinates `[x0, y0, x1, y1]` and are
converted to normalized XYWH.

## Invocation

Run in host context after `release-local` and the model-control ownership gate
succeed. Use the installed SAM 3 environment and a fresh output directory.

```bash
python models/experiments/sam3/adapter/benchmark_template_refinement.py \
  --manifest /path/to/frozen-manifest.json \
  --checkpoint /path/to/sam3.1_multiplex.pt \
  --repo models/experiments/sam3/repo \
  --output-dir /path/to/new-run \
  --device cuda:0
```

The runner checks the manifest, asset hashes, exact repository revision and
nonempty local checkpoint before importing Torch or SAM. It never downloads
weights. It rejects non-CUDA devices and checks that the selected GPU and
`nvidia-smi` are available before creating the output directory. Whole-device
GPU memory is sampled through model loading and inference. The official
`build_sam3_multiplex_video_predictor` runs with compilation and warmup off.
Each one-frame session receives one positive box prompt.

The selection rule matches `asset-research-sam31.md`: first retain returned
boxes with positive IoU against the requested normalized prompt box, then
select the highest model score among those candidates. The runner records total
candidate count, selected overlap and selected score. Missing boxes, no
positive-overlap candidates, invalid coordinates, shape mismatches or output
misalignment fail closed.

The official builder's stdout, stderr and Python logging are saved in
`predictor_builder.log`. The runner instruments the actual `load_state_dict`
calls without changing their behavior and saves full missing/unexpected key
lists in `checkpoint_load_diagnostics.json`. Dataset scoring begins only if
exactly one composite `Sam3MultiplexTrackingWithInteractivity` checkpoint load
occurred with no missing or unexpected keys. Build exceptions and incompatible
weights therefore fail before the per-object loop. Per-item and cleanup
failures are recorded, and any such failure produces a nonzero exit status.

## Artifacts and metrics

- `results.json`: manifest, asset and checkpoint hashes; source revision; GPU,
  driver and framework metadata; model-load and total runtime; per-process
  allocated/reserved memory peaks; whole-device baseline/peak/delta memory;
  metric means and failure details.
- `predictor_builder.log`: captured official builder stdout, stderr and logs.
- `checkpoint_load_diagnostics.json`: each actual loader call and full
  missing/unexpected key lists; compatibility gate result is represented by
  successful progression into dataset scoring.
- `per_object.csv`: baseline and refined mask IoU, paired delta, prompt
  latency, item time, SAM score, candidate count, selected box IoU and ID.
- `masks/<sample_id>.npz`: predicted binary refinement mask.
- `failures/<sample_id>.txt`: traceback for failed items.

The primary paired metric is per-object mask IoU against fixed ground truth.
The baseline is the exact input template mask. Report baseline IoU, refined IoU
and `refined_iou - base_iou`. Retrieval effects require a separate frozen
query set and embedding harness; this adapter does not recalculate embeddings.
