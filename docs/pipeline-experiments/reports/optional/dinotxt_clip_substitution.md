# DINOv3 + dino.txt ViT-L/16 substitution experiment

**Status:** asset gate passed; GT-mask-isolated crop category retrieval measured and reviewed (16 Nov 2026). See [detailed result](dinotxt_global_retrieval_test5.md). Broader Stage 1 (cross-view identity / graph diagnostics) and Stage 2 (optional pipeline-slot acceptance) remain explicitly pending. The candidate is a direct shared-space image/text encoder for the SigLIP2 mask-isolated slot. It is distinct from DINOde's ODE text transport and T-REN's point-conditioned region tokens.

## Why this is a valid candidate

Meta's official DINOv3 release includes the `dinov3_vitl16_dinotxt_tet1280d20h24l` PyTorch Hub model and a zero-shot segmentation notebook. Its factory freezes the pretrained distilled ViT-L/16 backbone and loads a learned two-block vision head plus a trained text encoder. The API exposes normalized image and text features in a shared 2048-dimensional space and scores them with cosine similarity multiplied by a learned logit scale. The official image feature combines a class-token branch with mean-pooled patch-token features. Use this complete pretrained model path; raw DINOv3 CLS features alone are not text-aligned and are not a substitute.

The official loader is `torch.hub.load(<local-dinov3-repo>, "dinov3_vitl16_dinotxt_tet1280d20h24l", source="local", weights=<dinotxt-head>, backbone_weights=<vitl16-backbone>, bpe_path_or_url=<local-vocabulary>)`. Use the official tokenizer, image transform, `encode_image`, `encode_text`, and logit scale. Freeze all weights. Do not train or fine-tune on ReplicaSSG.

## Asset and run gate

Before initialization, pin the DINOv3 repository commit and record the license/access basis for local use. Verify the full SHA-256 values and official filename prefixes for the ViT-L/16 LVD-1689M backbone and dino.txt head. Keep both assets outside Git and disable network fallback during inference.

A local audit on 16 November 2026 verified all three assets under `models/experiments/t_ren/weights/dinov3/`:

- **Backbone:** `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` — 1,213,021,279 bytes, SHA-256 `e73f0550f00e5eafe9712177da027e5f34df3de5250b56370bcb9acaa23dab9c` (matches `8aa4cbdd` prefix).
- **Head + text encoder:** `dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth` — 2,253,936,683 bytes, SHA-256 `a442d8f52a3a7ad715bf6b7d8117fb3a84d54249389b0a13f6956cd0d2eca4f0` (matches `a442d8f5` prefix).
- **BPE vocabulary:** `bpe_simple_vocab_16e6.txt.gz` — 1,356,917 bytes, SHA-256 `924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a`.
- **Source commit:** `6876159a11b4df116f30f667f8c9888617df0751` (exact match).

The asset gate is passed. The GT-mask-isolated crop category retrieval benchmark ran on the five-scene held-out split (2,204 views, 49 categories) with all metrics measured. See the detailed result report for full numbers. Broader Stage 1 (cross-view identity and graph-triplet diagnostics) and Stage 2 (optional pipeline-slot acceptance) remain explicitly pending.

## Stage 1: matched retrieval benchmark

GT-mask-isolated crop category retrieval (macro AP, macro top-1/top-5, macro query recall@1/5/10/50) is measured and reviewed on the five-scene held-out split (2,204 views, 49 categories). See the detailed result report.

Remaining Stage 1 diagnostics — cross-view identity top-1, MRR, recall@1/5/10 using leave-one-frame-out prototypes, graph-node identity accuracy, and oracle relation-triplet recall — are **explicitly pending**. They require the same frozen ReplicaSSG test subset, object IDs, and score code, but have not yet been computed for dino.txt.

All other Stage 1 requirements are documented: frozen five scenes with 6 excluded scenes lacking local RGB/poses, official transform and prompt, descriptive SigLIP2 and FG-CLIP2 controls, asset hashes, transform/prompt definitions, throughput, CUDA/host memory, cache size, cache namespace, and release-local workflow compliance.

## Stage 2: optional pipeline-slot acceptance

**Explicitly pending.** Stage 1 crop category retrieval has a measured result, but broader identity/graph diagnostics remain incomplete. Only after all Stage 1 work is reviewed should an opt-in dino.txt encoder configuration be added. Changes would affect only the mask-isolated encoder: detector, prompts, masks, VLM, frame sample, mapping thresholds, and downstream stages stay fixed. Build a fresh dino.txt map and cache; never combine dino.txt and CLIP embeddings in one map. Measure map identity and association on labeled data, plus runtime, peak memory, and cache size. Keep SigLIP2 as the default unless paired quality and resource evidence supports switching. Do not claim VQA improvement from this study: the current ReplicaSSG split has no aligned natural-language QA labels. Test VQA separately once a labeled question set and fixed answer-evidence protocol exist.

## Separation from related experiments

- **Visual-only DINOv3** has no text-aligned shared space; its earlier association benchmark is not a dino.txt result.
- **DINOde** transports a CLIP text embedding into frozen DINOv3 features with a learned ODE. It is a different model and retrieval path.
- **T-REN** uses point prompts and learned region tokens. Its runner and scores must remain a separate region-token experiment.

## Primary sources

- [Official DINOv3 README and dino.txt zero-shot model](https://github.com/facebookresearch/dinov3/blob/main/README.md)
- [Official dino.txt hub factory](https://github.com/facebookresearch/dinov3/blob/main/dinov3/hub/dinotxt.py)
- [Official DINOTxt API](https://github.com/facebookresearch/dinov3/blob/main/dinov3/eval/text/dinotxt_model.py)
