# T-REN region-token embedding status

**Status:** repository and T-REN checkpoint are installed. No inference or accuracy benchmark can run because T-REN also requires two official DINOv3 assets that currently return HTTP 403.

T-REN source commit `0817b0d015a4462756248825eee0c3da5aada37a` is under `models/experiments/t_ren/repo`. The `savyak2/T-REN` checkpoint revision `cfc2ea88a4f8e7679df43a8ecbb4a99701329cd8` is in HF_HOME. Its size is 1,455,981,684 bytes and SHA-256 is `7f85132237001d52176226b3fd6ebde8ecfc08569ae196d56deeb5b052bf9b1f`.

Required DINOv3 files, as named by upstream T-REN, are the ViT-L/16 backbone and its text-aligned vision/text head. Direct HEAD requests to both exact official download URLs returned HTTP 403 on 2026-11-15. No unofficial weights or standard-crop substitutes were used. Thus no retrieval, identity, runtime, throughput, or GPU-memory metric is reported.

The intended method is point-prompted region-token extraction followed by text alignment. It cannot be evaluated by pooling generic image or crop embeddings. See the [full status and exact URLs](../../../../scripts/pipeline_model_reports/t_ren.md).
