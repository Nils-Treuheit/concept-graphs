# T-REN region-token embedding feasibility report

**Status date:** 2026-11-15; official checkpoint access rechecked at 2026-11-15 22:28 UTC.
**Benchmark:** Not run. Repository and region-encoder checkpoint are installed, but the required official DINOv3 text/vision dependencies remain unavailable.

## Finding

The official repository is installed at `models/experiments/t_ren/repo` (commit `0817b0d015a4462756248825eee0c3da5aada37a`); the region encoder checkpoint is cached from `savyak2/T-REN` at revision `cfc2ea88a4f8e7679df43a8ecbb4a99701329cd8` (1,455,981,684 bytes; SHA-256 `7f85132237001d52176226b3fd6ebde8ecfc08569ae196d56deeb5b052bf9b1f`). The official DINOv3 code repository is installed at `models/experiments/t_ren/dinov3-repo` (commit `6876159a11b4df116f30f667f8c9888617df0751`). The required official DINOv3 `dino.txt` vision/text head and backbone files are absent. I rechecked the exact upstream URLs from the installed DINOv3 and T-REN documentation; both returned HTTP 403 on 2026-11-15. No unofficial mirror or substitute has been used.

- Backbone: `https://dl.fbaipublicfiles.com/dinov3/dinov3_vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`
- Vision/text head: `https://dl.fbaipublicfiles.com/dinov3/dinov3_vitl16/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth`

Pooling standard image or crop embeddings would not evaluate T-REN. No inference or accuracy measurement has run.

## Evaluation protocol

Use deterministic points sampled from held-out object masks and the official region-token extraction/text-alignment path. Compare against the same fixed encoder and query control split. Measure single-view object retrieval first; evaluate cross-view map association only after a stable object-feature path is established.

- **Dataset/split:** same held-out labeled object/query set as template retrieval. Record point sampling and mask-to-prompt mapping. No split was executed.
- **Metrics:** per-category macro AP, Recall@K, object attribution, valid cross-view identity association, latency, throughput, and peak GPU memory.
- **Runtime/resources:** **Not measured.**
- **Failures/blocker:** missing official DINOv3 backbone `.pth` plus dino.txt head download access; method environment, prompts, and output manifest are also pending.
- **Artifacts:** source and T-REN weights installed; no prompts, embeddings, predictions, or benchmark scores exist.

Protocol source: `MODERNIZED_PIPELINE_IMPLEMENTATION_PLAN.md`, “Further embedding research” and “Evaluation plan.”
