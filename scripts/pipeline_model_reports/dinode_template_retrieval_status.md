# DINOde template text-retrieval status

**Status:** Completed. The DINOde mask-isolated crop retrieval benchmark
(test5) finished at 2026-11-17T03:33:42.167032Z. Macro one-vs-rest AP is
0.0368157099, observation top-1 accuracy is 0.0526315789, and top-5 accuracy
is 0.2568058076 across 2,204 held-out views on five scenes. The full result is
reported in [`dinode_template_retrieval_test5.md`](dinode_template_retrieval_test5.md).

## Model paths used in test5

The official DINOde repository is installed at `models/experiments/dinode/repo`
(commit `0a2c5182c44107fdd8c786b2192fcc3c5e5bebd1`). The official ODE
checkpoint from `HaominPeng/DINOde` is installed at
`models/experiments/dinode/checkpoints/eccv26_dinode_coco_stuff.pth` (214,589,829
bytes; SHA-256
`494354fe148a278a99c8186a59d9fa5ef14cf6254e312195eb08cac69cac1377`). The
official DINOv3 ViT-L/16 snapshot and OpenCLIP `ViT-L-14` /
`laion2b_s32b_b82k` assets are cached. The OpenCLIP model is installed in
HF_HOME at revision `ffcbdd7da3d52f9fd7bb15e78ab6815e1a87ef60`; the
`open_clip_pytorch_model.bin` SHA-256 is
`5ddb47339f44e4fd9cace3d3960d38af1b51a25857440cfae90afc44706d7e2b`. The
official [LAION model card](https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K)
gives the OpenCLIP loading path and lists MIT. It states that deployment is out
of scope and recommends thorough in-domain testing before constrained
non-deployed use. This completed result was produced in a local,
non-deployed evaluation; assets and outputs remain local. The DINOde repository does not declare its own license. The project owner
has authorized this local experiment, but that authorization does not establish
upstream terms or broader redistribution rights.

A visual-only DINOv3 association run is reported separately in
[`dinov3_visual_association_test5.md`](dinov3_visual_association_test5.md). It
cannot substitute for DINOde because it cannot encode a text query in the same
feature space.

## Result summary

The completed test5 run used the same five-scene split as the matched held-out
comparison: `office_1`, `office_3`, `office_4`, `room_1`, and `room_2` (250
sampled frames, 2,204 GT object views, 49 categories). The result is a
descriptive, local measurement; no significance claim is made. The result
documents the protocol, inputs, model configuration and hashes, per-scene
metrics, control comparison with caveats, runtime, GPU resources, environment,
prior failures, durable artifact locations, and licensing/use limitations.

**Comparison caveat:** matched aggregate controls (SigLIP2 So400m AP 0.0313,
FG-CLIP2 So400m AP 0.0403) have no per-view scores, exact crop pixel
equivalence is unproven, and no paired significance test exists. The comparison
is descriptive only.

## Licensing and redistribution

DINOde's upstream license remains undeclared. The project owner authorized this
local experiment; that authorization does not establish upstream terms or
redistribution rights. Keep assets and outputs local as a project precaution.
No deployment suitability is inferred from this result.
