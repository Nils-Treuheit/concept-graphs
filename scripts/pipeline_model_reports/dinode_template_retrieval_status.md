# DINOde template text-retrieval status

**Status:** The adapter source is implemented and source-reviewed at `scripts/pipeline_model_reports/dinode_replica_test5.py`. No DINOde benchmark has run, and the run is blocked by the missing pinned `preprocessor_config.json` at the HF snapshot path (`/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub/models--facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots/ea8dc2863c51be0a264bab82070e3e8836b02d51/preprocessor_config.json`). The user can run `python3 scripts/download_dinov3_processor_config.py`, which caches and verifies that single file under HDD-backed `HF_HOME`. Upstream licensing or use terms for the DINOde repository remain unverified. As a project precaution, keep assets and outputs local until those terms are clarified.

## Required model path

The official DINOde repository is installed at `models/experiments/dinode/repo` (commit `0a2c5182c44107fdd8c786b2192fcc3c5e5bebd1`). The official ODE checkpoint from `HaominPeng/DINOde` is installed at `models/experiments/dinode/checkpoints/eccv26_dinode_coco_stuff.pth` (214,589,829 bytes; SHA-256 `494354fe148a278a99c8186a59d9fa5ef14cf6254e312195eb08cac69cac1377`). The official DINOv3 ViT-L/16 snapshot and OpenCLIP `ViT-L-14` / `laion2b_s32b_b82k` assets are cached. The OpenCLIP model is installed in HF_HOME at revision `ffcbdd7da3d52f9fd7bb15e78ab6815e1a87ef60`; the `open_clip_pytorch_model.bin` SHA-256 is `5ddb47339f44e4fd9cace3d3960d38af1b51a25857440cfae90afc44706d7e2b`. The official [LAION model card](https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K) gives the OpenCLIP loading path and lists MIT. It states that deployment is out of scope and recommends thorough in-domain testing before constrained non-deployed use. The planned local evaluation is non-deployed and will remain local. The DINOde repository does not declare its own license. The project owner has authorized this local experiment, but that authorization does not establish upstream terms or broader redistribution rights. The DINOde adapter source is implemented and reviewed; no ConceptGraphs result has been measured.

A visual-only DINOv3 association run is reported separately in [`dinov3_visual_association_test5.md`](dinov3_visual_association_test5.md). It cannot substitute for DINOde because it cannot encode a text query in the same feature space.

## Proposed held-out comparison

Use the locally available overlap of ReplicaSSG `test_scans.txt`: `office_1`, `office_3`, `office_4`, `room_1`, and `room_2` (250 sampled frames). Compare DINOde with the same SigLIP2 So400m mask-isolated and FG-CLIP2 So400m padded retrieval controls. Freeze mapped template views, natural-language queries, candidate gallery, answerer, prompt and output schema. Keep each encoder's cache separate.

Report category/object Recall@1/5, macro AP, cross-view identity, subject/template attribution, answer correctness, relation direction, evidence support, abstention, and end-to-end query latency. Record ODE/query encoding, retrieval and answer timing separately, plus cache size, warm/cold timing, peak CUDA memory, exact code/checkpoint revisions and failures. The current matched retrieval comparison is [`template_retrieval_test5.md`](template_retrieval_test5.md); it contains no VQA result.

**Decision:** proceed with the locally authorized experiment after the missing `preprocessor_config.json` is downloaded. The adapter source and GPU launch gate are ready. Keep the unresolved upstream-terms notice with the result. As a project precaution, do not redistribute assets or outputs while those terms remain unclear. The current no-score status is not a negative benchmark finding.

## Current upstream check

On 2026-11-16, the [official DINOde repository](https://github.com/yoon307/DINOde)
listed its source, README, requirements, and model files without a root
`LICENSE` file, and its README did not state a DINOde license. The project
owner later explicitly authorized a local experiment. This does not establish
upstream licensing terms. Inference is still pending, so this item remains
unscored rather than a model failure.
