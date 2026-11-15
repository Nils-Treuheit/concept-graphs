# Held-out Replica template-retrieval and object-association comparison

**Run date:** 15 November 2026. **Status:** retrieval and visual object-association component experiment completed. **End-to-end template-matching VQA was not run.**

## Scope

This experiment compares the selected SigLIP2 and FG-CLIP2 retrieval/association paths with visual-only DINOv3. It uses ReplicaSSG ground-truth masks rather than detector or map outputs. It evaluates text-to-category retrieval and multi-view identity matching. It does not evaluate VLM question answering, detected-map retrieval, predicate generation, DINOde, or end-to-end pipeline acceptance.

## Dataset and split

The source split is ReplicaSSG `files/test_scans.txt` (SHA-256 `b7f8e16ad992562c76192dbfc96566dc1fdc4ab129a14bc1585e87badeb0dd33`). Eleven held-out scans are listed. Only five have local Replica RGB sequences and camera trajectories: `office_1`, `office_3`, `office_4`, `room_1`, and `room_2`. Six scans were excluded because local RGB or pose files were missing: `apartment_1`, `apartment_2`, `hotel_0`, `frl_apartment_3`, `frl_apartment_4`, and `frl_apartment_5`.

The matched run sampled 50 evenly spaced frames per available scene: 250 frames, 2,204 visible ground-truth object views, and 49 object categories. Masks were raycast from Replica semantic-mesh instance IDs at the RGB camera poses. Category and relationship labels came from `objects.json` and `relationships.json`. The five selected scenes contain 91 unique annotated relationship tuples. The graph metric scores 49 visible relation observations because both endpoints must be visible in a query frame and have identity prototypes available. This is an oracle-proposal split; detector quality and predicted object tracks are outside scope.

Each scored retrieval query uses `a photo of a {category}` against the complete object-view bank. Association metrics leave out the query frame when forming each object's visual prototype. Scene-graph metrics hold ReplicaSSG predicates fixed and measure the effect of identity assignment errors on graph nodes and relation triples.

## Exact configurations and results

All runs used CUDA 0, batch size 16, the same five scenes, frame sampling, object-view bank, and two input conditions. The SigLIP2 cache snapshot was `e8e487298228002f3d8a82e0cd5c8ea9c567f57f`; the FG-CLIP2 snapshot was `d57d30fe94a107dd6a2610eb4e9a135004d823a4`. These revisions were verified from local cache paths because the run JSON omitted them. The padded crop used a 20-pixel context. The mask-isolated crop filled pixels outside the ground-truth mask with RGB `[128,128,128]`, matching the stored processor mean `[0.5,0.5,0.5]` for both CLIP models.

| Encoder and feature path | Text mAP, padded / isolated | Identity top-1, padded / isolated | Identity MRR, padded / isolated | Identity R@5, padded / isolated | Oracle graph node accuracy, padded / isolated | Oracle relation-triplet recall, padded / isolated | Encode seconds, padded / isolated | Peak allocated / reserved CUDA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SigLIP2 So400m, shared crop encoder | 0.0226 / 0.0313 | 0.2395 / **0.3966** | **0.3910** / **0.5373** | **0.5652** / **0.7057** | 0.2395 / **0.3966** | 0.0408 / **0.3878** | 17.98 / 18.77 | 2.471 / 2.650 GiB |
| FG-CLIP2 So400m, shared crop encoder | **0.0247** / **0.0403** | **0.2446** / 0.3791 | 0.3834 / 0.5207 | 0.5357 / 0.6891 | 0.2446 / 0.3791 | **0.0612** / 0.3265 | 26.31 / 27.38 | 6.710 / 7.742 GiB |
| FG-CLIP2 So400m, native full-frame RoIAlign | 0.0232 | 0.1681 | 0.3489 | 0.5675 | 0.1681 | 0.0408 | 3.65 | 6.710 / 7.742 GiB |
| DINOv3 ViT-L/16 CLS, FP32 visual-only | N/A | 0.2142 / 0.3800 | 0.3702 / 0.5239 | 0.5472 / 0.6978 | 0.2142 / 0.3800 | 0.0000 / 0.3265 | 16.02 / 16.85 | 1.319 / 1.402 GiB |

Category mAP and zero-shot category accuracy are not applicable to DINOv3 because it has no shared text-image space. It is not DINOde. The DINOv3 model ID was `facebook/dinov3-vitl16-pretrain-lvd1689m`, snapshot `ea8dc2863c51be0a264bab82070e3e8836b02d51`. The runner used final-layer `pooler_output` with CLS fallback, L2 normalization, 224×224 bicubic resize, ImageNet mean `[0.485,0.456,0.406]`, and standard deviation `[0.229,0.224,0.225]`. The cached snapshot lacked processor metadata, so the recorded preprocessing should be checked against the upstream model card before interpreting absolute scores.

SigLIP2 loaded in 19.14 s and encoded category text in 0.44 s. FG-CLIP2 loaded in 49.39 s and encoded text in 0.04 s for shared-crop categories; its native RoI condition used its own box-text path. DINOv3 loaded in 1.23 s. Per-process resident CUDA allocations were 2.138 GiB for SigLIP2, 2.204 GiB for FG-CLIP2, and 1.130 GiB for DINOv3. Memory figures exclude other processes. Whole-job wall time, host RAM, and energy were not recorded. The reported inference times cover image encoding for each condition; text encoding is reported separately above.

Graph denominators were 2,171 scored object observations and 49 scored relation observations for each condition. An edge observation is included only when both relation endpoints are visible in the frame and identity prototypes are available for both endpoints. There were 91 unique ReplicaSSG relationship tuples across the five selected scenes. The reported graph score therefore does not provide predicate-generation recall. The explicit run artifacts include all metrics and settings:

- [Matched SigLIP2 and FG-CLIP2 JSON](../../data/replica-test-retrieval-controls.json)
- [DINOv3 FP32 JSON](../../data/dinov3-replica-test5-fp32.json)
- [DINOv3 corrected runner](../../data/replica_dinov3_benchmark.py)

## Interpretation

Mask isolation improved cross-view identity retrieval and downstream graph identity for all three methods on this subset. SigLIP2 So400m produced the strongest isolated-mask identity top-1 and oracle relation-triplet recall. FG-CLIP2 had the highest category-text mAP in this comparison, while its native RoIAlign path was weaker on the identity metrics and used the highest peak allocation. DINOv3 FP32 approached the CLIP-family identity results with lower recorded memory, but did not improve the selected SigLIP2 mapping feature on this split.

The evidence supports keeping SigLIP2 So400m as the map-association feature for the evaluated five-scene subset. FG-CLIP2 remains relevant for the separate padded-proposal retrieval feature slot; category mAP alone did not make it the best visual identity feature. No threshold tuning was attempted because the experiment isolates encoder embeddings with oracle masks.

An earlier, nonmatched component benchmark used eight ReplicaSSG scenes, 400 frames, and 3,545 object views. There, SigLIP2 So400m mask-isolated identity top-1 was 0.4015 and oracle relation-triplet recall was 0.3026. Those values are context only because its scene set and denominators differ from this held-out five-scene experiment. See [`feature_encoders.md`](../feature_encoders.md) and the original [comparison report](../../data/baseline-five-model-comparison-20261029.md).

## Failed initial DINOv3 attempt

The first DINOv3 attempt used FP16 weights and inputs. Its saved association scores are invalid and must not be compared: the run produced 0.02994 top-1 for both crop conditions, while a follow-up CUDA diagnostic found NaN cosine similarity for all 40 paired real crops. Non-finite features caused `argmax` to collapse to candidate index zero. The diagnostic artifact is [dinov3-input-diagnostic.json](../../data/dinov3-input-diagnostic.json); the invalid run is [dinov3-replica-test5.json](../../data/dinov3-replica-test5-invalid-fp16.json). The corrected runner uses FP32 and rejects non-finite embeddings before scoring. Corrected FP32 results above are the only DINOv3 values used in the interpretation.

## VQA status and limits

This is a retrieval and cross-view association experiment, not template-matching VQA. ReplicaSSG provides category and relationship labels but no reference object captions or held-out natural-language VQA set. There are no ConceptGraphs maps or fixed answer-evidence packets for these five scenes. The generic image-VQA panel and per-frame annotation comparison are separate roles and are not used as VQA results here. The VQA readiness assessment and executable follow-up protocol are in [`template_vqa_query_vlm_status.md`](template_vqa_query_vlm_status.md). End-to-end answer correctness, attribution, evidence support, abstention, and answer latency remain unmeasured.
