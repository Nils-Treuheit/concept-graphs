# ObjEmbed object-region embedding feasibility report

**Status date:** 2026-11-15
**Benchmark:** Not run. No GPU/model job was started.

## Finding

The project plan treats ObjEmbed as a separate object-region experiment. It uses its own proposal and object-scoring path. No ObjEmbed integration, installed dependency, or checkpoint was found in this checkout or the inspected local model cache. Pooling current YOLOE crops would not evaluate ObjEmbed's intended path. This is an unavailable-method result, not a negative accuracy result.

## Evaluation protocol

When the official code and weights are available, run the method's object proposal/scoring path on the same held-out images and object queries as deployed encoders. Preserve its proposal features and scoring. Compare with SigLIP2 So400m and FG-CLIP2 So400m on fixed query sets and the same split. Evaluate multi-view map association only if the official path produces comparable object features.

- **Dataset/split:** same labeled object/query split as template retrieval; ReplicaSSG held-out test scans may support cross-view evaluation if GT object regions can be processed with the official method. No split was executed.
- **Metrics:** per-category macro AP, Recall@K, object attribution, valid cross-view identity association/graph-edge recovery, throughput, and peak GPU memory.
- **Runtime/resources:** wall time, throughput, and peak allocated/reserved GPU memory. **Not measured.**
- **Failures/blocker:** no method-specific source, dependency, or weights located. The required proposal/scoring path is absent.
- **Artifacts:** no model outputs or benchmark artifacts generated. Intended report path: `scripts/pipeline_model_reports/objembed.md`.

Protocol source: `MODERNIZED_PIPELINE_IMPLEMENTATION_PLAN.md`, “Further embedding research” and “Evaluation plan.”
