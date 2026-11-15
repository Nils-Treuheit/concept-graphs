# DINOde template text-retrieval status

**Status:** not run. No ConceptGraphs DINOde retrieval accuracy, answer accuracy, latency, or resource measurements exist.

## Required model path

The planned DINOde arm transports text features into a frozen DINOv3 visual feature space using a learned ODE flow. It requires the official DINOde implementation, a compatible OpenCLIP text encoder, DINOv3 weights, and the learned ODE/checkpoint assets. The local asset audit found the DINOv3 ViT-L/16 weights but did not find the DINOde repository or ODE/text-transport checkpoint. No runnable DINOde integration is present in this checkout.

A visual-only DINOv3 association run is reported separately in [`dinov3_visual_association_test5.md`](dinov3_visual_association_test5.md). It cannot substitute for DINOde because it cannot encode a text query in the same feature space.

## Proposed held-out comparison

Use the locally available overlap of ReplicaSSG `test_scans.txt`: `office_1`, `office_3`, `office_4`, `room_1`, and `room_2` (250 sampled frames). Compare DINOde with the same SigLIP2 So400m mask-isolated and FG-CLIP2 So400m padded retrieval controls. Freeze mapped template views, natural-language queries, candidate gallery, answerer, prompt and output schema. Keep each encoder's cache separate.

Report category/object Recall@1/5, macro AP, cross-view identity, subject/template attribution, answer correctness, relation direction, evidence support, abstention, and end-to-end query latency. Record ODE/query encoding, retrieval and answer timing separately, plus cache size, warm/cold timing, peak CUDA memory, exact code/checkpoint revisions and failures. The current matched retrieval comparison is [`template_retrieval_test5.md`](template_retrieval_test5.md); it contains no VQA result.

**Decision:** defer DINOde until official code and learned checkpoint assets are available and compatible with this environment. This is a feasibility blocker, not a negative benchmark finding.
