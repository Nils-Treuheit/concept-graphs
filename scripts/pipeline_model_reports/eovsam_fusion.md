# EOVSAM fusion feasibility report

**Status date:** 2026-11-15
**Benchmark:** Not run. No GPU/model job was started.

## Finding

The project plan defers EOVSAM until the YOLOE-only pipeline passes acceptance. No EOVSAM implementation, installed dependency, or checkpoint was found in this checkout or the inspected local model cache. This is an unavailable-method result, not a negative accuracy result.

## Evaluation protocol

Keep fusion disabled by default and write outputs under a unique experiment suffix. On the same held-out labeled frames as the accepted YOLOE baseline, prompt EOVSAM with the unique categories in YOLOE detections. Match predictions one-to-one by category and mask overlap. For high-agreement pairs, compare the better individual mask with the union mask. Make unmatched-mask handling and confidence weighting explicit config options. First precision-oriented condition discards unmatched EOVSAM masks. Verify instance separation before evaluating fusion.

- **Dataset/split:** labeled held-out frames from the accepted detector benchmark; freeze split and manifest hash before execution. No split was executed.
- **Metrics:** mask IoU and boundary quality, class precision/recall, instance-separation errors, projected point-cloud contamination, and map association if instance-ID ground truth exists.
- **Runtime/resources:** per-frame and total wall time, peak allocated/reserved GPU memory, model residency, failed prompts, and OOMs. **Not measured.**
- **Failures/blocker:** no local source, dependency, or checkpoint; instance-separated category outputs have not been verified.
- **Artifacts:** no configs, predictions, masks, or benchmark outputs generated. Intended report path: `scripts/pipeline_model_reports/eovsam_fusion.md`.

Protocol source: `MODERNIZED_PIPELINE_IMPLEMENTATION_PLAN.md`, “Later experiment A: optional EOVSAM fusion.”
