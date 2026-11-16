# DINOde template text-retrieval status

**Status:** installed, but local project policy keeps inference on hold pending clarification of applicable terms or permission. No ConceptGraphs DINOde retrieval accuracy, answer accuracy, latency, or resource measurements exist.

## Required model path

The official DINOde repository is installed at `models/experiments/dinode/repo` (commit `0a2c5182c44107fdd8c786b2192fcc3c5e5bebd1`). The official ODE checkpoint from `HaominPeng/DINOde` is installed at `models/experiments/dinode/checkpoints/eccv26_dinode_coco_stuff.pth` (214,589,829 bytes; SHA-256 `494354fe148a278a99c8186a59d9fa5ef14cf6254e312195eb08cac69cac1377`). DINOv3 ViT-L/16 and OpenCLIP assets are cached. The repository does not state an applicable DINOde license, and no separate project permission is recorded. Under local project policy, inference stays on hold until terms or permission are clarified. Asset acquisition is recorded separately from that decision.

A visual-only DINOv3 association run is reported separately in [`dinov3_visual_association_test5.md`](dinov3_visual_association_test5.md). It cannot substitute for DINOde because it cannot encode a text query in the same feature space.

## Proposed held-out comparison

Use the locally available overlap of ReplicaSSG `test_scans.txt`: `office_1`, `office_3`, `office_4`, `room_1`, and `room_2` (250 sampled frames). Compare DINOde with the same SigLIP2 So400m mask-isolated and FG-CLIP2 So400m padded retrieval controls. Freeze mapped template views, natural-language queries, candidate gallery, answerer, prompt and output schema. Keep each encoder's cache separate.

Report category/object Recall@1/5, macro AP, cross-view identity, subject/template attribution, answer correctness, relation direction, evidence support, abstention, and end-to-end query latency. Record ODE/query encoding, retrieval and answer timing separately, plus cache size, warm/cold timing, peak CUDA memory, exact code/checkpoint revisions and failures. The current matched retrieval comparison is [`template_retrieval_test5.md`](template_retrieval_test5.md); it contains no VQA result.

**Decision:** keep installed assets but do not execute until applicable terms or permission are clarified under local project policy. This is a policy blocker, not a negative benchmark finding.

## Current upstream check

On 2026-11-16, the [official DINOde repository](https://github.com/yoon307/DINOde)
still listed its source, README, requirements, and model files without a root
`LICENSE` file, and its README did not mention a DINOde license. The local
project evidence contains no separate permission record. Therefore no
inference was started under the project's review policy; this item remains
unscored rather than a model failure.
