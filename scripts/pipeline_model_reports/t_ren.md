# T-REN region-token embedding feasibility report

**Status date:** 2026-11-16; official download URLs last checked on 2026-11-15.
**Benchmark:** Not run. The required official DINOv3 DINOtxt head files are unavailable; cached visual backbone snapshots do not include the text-aligned head. No inference or accuracy measurement has run.

## Finding

The official repository is installed at `models/experiments/t_ren/repo` (commit `0817b0d015a4462756248825eee0c3da5aada37a`); the region encoder checkpoint is cached from `savyak2/T-REN` at revision `cfc2ea88a4f8e7679df43a8ecbb4a99701329cd8` (1,455,981,684 bytes; SHA-256 `7f85132237001d52176226b3fd6ebde8ecfc08569ae196d56deeb5b052bf9b1f`). The official DINOv3 code repository is installed at `models/experiments/t_ren/dinov3-repo` (commit `6876159a11b4df116f30f667f8c9888617df0751`).

### Cached visual backbone snapshots

Three visual backbone HF snapshots are present, but they do not supply the required DINOtxt head:

- ViT-S/16, revision `114c1379950215c8b35dfcd4e90a5c251dde0d32`: `model.safetensors`, 86,406,384 bytes, SHA-256 `4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d`.
- ViT-B/16, revision `5931719e67bbdb9737e363e781fb0c67687896bc`: `model.safetensors`, 342,662,192 bytes, SHA-256 `9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b`.
- ViT-L/16, revision `ea8dc2863c51be0a264bab82070e3e8836b02d51`: `model.safetensors`, 1,212,559,808 bytes, SHA-256 `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`.

These are visual backbone HF snapshots, not T-REN's expected `.pth` assets. T-REN supports only `dinov3_vitl16` and requires `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` (backbone) and `dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth` (DINOtxt head). The cached files contain no text-aligned head and are not in the expected `.pth` format.

### Download blocker and access

No official-email URLs have been supplied in this task, and email receipt has not been confirmed. No inference or accuracy run is possible.

Official HTTPS download URLs:
- Backbone: `https://dl.fbaipublicfiles.com/dinov3/dinov3_vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`
- DINOtxt head: `https://dl.fbaipublicfiles.com/dinov3/dinov3_vitl16/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth`

### Downloader instructions

A downloader script is available at `scripts/download_tren_dinov3_assets.py`. It accepts official HTTPS `dl.fbaipublicfiles.com` URLs only, verifies the first eight SHA-256 hex digits, and writes weights outside the Git worktree.

Requirements:
- A private mode-600 file outside the repository containing exactly two lines:
  ```
  backbone=<URL>
  dinotxt=<URL>
  ```
- An output directory outside the Git worktree.

Command:
```
python3 scripts/download_tren_dinov3_assets.py --url-file <private-file> --output-dir <outside-repo-directory>
```

### Links

- Meta DINOv3 access page: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/
- DINOv3 README: https://github.com/facebookresearch/dinov3/blob/main/README.md
- T-REN README: https://github.com/savya08/T-REN

Pooling standard image or crop embeddings would not evaluate T-REN.

## Evaluation protocol

Use deterministic points sampled from held-out object masks and the official region-token extraction/text-alignment path. Compare against the same fixed encoder and query control split. Measure single-view object retrieval first; evaluate cross-view map association only after a stable object-feature path is established.

- **Dataset/split:** same held-out labeled object/query set as template retrieval. Record point sampling and mask-to-prompt mapping. No split was executed.
- **Metrics:** per-category macro AP, Recall@K, object attribution, valid cross-view identity association, latency, throughput, and peak GPU memory.
- **Runtime/resources:** **Not measured.**
- **Failures/blocker:** missing official DINOv3 backbone `.pth` plus DINOtxt head download access; method environment, prompts, and output manifest are also pending.
- **Artifacts:** source and T-REN weights installed; no prompts, embeddings, predictions, or benchmark scores exist.

Protocol source: `MODERNIZED_PIPELINE_IMPLEMENTATION_PLAN.md`, "Further embedding research" and "Evaluation plan."
