# SAM 3.1 template-mask refinement feasibility report

**Status date:** 2026-11-15
**Benchmark:** Not run. No GPU/model job was started.

## Finding

The project plan reserves SAM 3.1 for selected representative object views after map consolidation; it is not a per-frame detector. The local Hugging Face cache contains a checkpoint blob, but there is no complete runnable project integration or standard Transformers inference package in this checkout. Therefore no benchmark was run.

- **Local checkpoint blob:** `/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub/models--facebook--sam3.1/blobs/0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6` (3,502,755,717 bytes).
- **Cache snapshot entry:** `models--facebook--sam3.1/snapshots/daa63191845a41281374e725f4c9e51c7a824460/` contains symlinks for `sam3.1_multiplex.pt` and `config.json`; this is checkpoint/config material, not a complete Transformers-loadable package.
- **Model-card/code status:** the [facebook/sam3.1 model card](https://huggingface.co/facebook/sam3.1/blob/main/README.md) identifies the repository as checkpoint-only, says it has no Hugging Face Transformers integration, and points to the [official SAM 3 code](https://github.com/facebookresearch/sam3). The checkpoint is gated under the model-card license terms. No SAM 3 integration was found in this project.

## Evaluation protocol

Freeze consolidated objects, representative RGB views, and visual box/point prompts. Keep each map category label fixed. Compare original template masks with SAM-refined masks using the same held-out queries and fixed retrieval backend. Score against independently labeled template masks, then recompute only the corresponding mask-aware embeddings. Keep this post-consolidation stage separately configurable and disabled by default.

- **Dataset/split:** held-out representative object views and the project query set; record split, object/view selection, and prompt manifest. No split was executed.
- **Metrics:** template mask IoU/boundary quality, retrieval Recall@K and macro AP, object attribution, added refinement/embedding latency, and peak GPU memory.
- **Runtime/resources:** **Not measured.**
- **Failures/blocker:** checkpoint cached, but no runnable project inference integration or complete standard Transformers package was available. No prompts, masks, embeddings, or predictions were produced.
- **Artifacts:** checkpoint/config only at the local HF cache paths above; no experiment outputs. Intended report path: `scripts/pipeline_model_reports/sam31_template_refinement.md`.

Protocol source: `MODERNIZED_PIPELINE_IMPLEMENTATION_PLAN.md`, “Later experiment B: optional SAM 3.1 template-mask refinement.”
