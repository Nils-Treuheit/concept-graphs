# Optional and substitution experiment evaluation

**Branch:** `pipeline-model-substitution-experiments`
**Evidence date:** 15 November 2026
**Evaluation owner:** root agent

## Per-frame VLM substitutions

All four candidates used the same 40 Replica `room0` frames, cached YOLOE detections, object labels, prompts, and output checks. Qwen3.8 completed 40/40 relation and caption lists, accounted for 763/763 caption IDs, and serves as the provisional annotation baseline. Its agreement advantage is self-referential because the cached Qwen3.8 outputs are the reference. It does not establish semantic accuracy.

LFM3-VL NVFP4 returned strict relation lists for 36/40 frames and strict caption lists for 28/40, covering 526/763 caption IDs. Qwen2.5-VL 7B produced strict relation outputs for 38/40 and fence-aware caption outputs for 40/40, covering 762/763 IDs. LLaVA OneVision Qwen2 7B completed all requests but produced no outputs accepted by the required list schema. Therefore Qwen3.8 remains the safest tested default for format reliability and object-ID coverage; the comparison does not establish an overall production winner. The four individual reports retain exact checkpoint, configuration, metrics, resource scope, failures and raw responses.

## Template retrieval and visual association

The previous eight-scene component benchmark reported SigLIP2 mask-isolated identity top-1 0.4015 and oracle relation-triplet recall 0.3026. The matched held-out ReplicaSSG run uses five available test scenes and different denominators, so these earlier values are context, not a direct paired delta.

On the new matched five-scene split, SigLIP2 So400m with ground-truth mask-isolated crops reached identity top-1 0.3966 and oracle graph triplet recall 0.3878. FG-CLIP2 So400m reached 0.3791 and 0.3265 on those measures, while its category macro AP was higher at 0.0403 versus SigLIP2's 0.0313. DINOv3 ViT-L/16 FP32 reached 0.3800 identity top-1 and 0.3265 triplet recall. Its visual-only path is not DINOde and does not answer text queries.

Keep SigLIP2 as the map-association feature for the measured subset and FG-CLIP2 for its existing padded-proposal retrieval role. This experiment used oracle masks and fixed predicates, so it does not validate predicted detections, generated relations, map tracks, or end-to-end template VQA. The first DINOv3 FP16 run is invalid: a CUDA diagnostic found non-finite features for all 40 diagnostic crop pairs. Its metrics are withdrawn; only the finite-checked FP32 result is evaluated.

## Query-time VQA readiness

No ConceptGraphs template-matching VQA benchmark was run. The five held-out scenes have no generated ConceptGraphs maps, frozen question manifest, or evidence packets. The existing generic direct-image VQA panel and 40-frame per-frame annotation comparison test different roles. Query-time accuracy, attribution, relation direction, evidence support, abstention, schema validity, latency, and GPU use are not measured for Qwen3.8, Qwen3-VL 8B Instruct, Qwen3-VL 30B Thinking, or Qwen3-VL 8B Thinking. Retain Qwen3.8 as a query baseline candidate only; do not name a query-time winner.

## Optional object and mask methods

EOVSAM fusion, ObjEmbed and T-REN lack method-specific local code and checkpoints. SAM 3.1 has a cached checkpoint blob, but no complete runnable integration was found. These are feasibility blockers, not negative benchmark results. No accuracy, runtime or resource comparison was produced.

The prior YOLOE-L and YOLOE-M run reported custom polygon AP50 0.5881 and 0.5805, respectively. It omitted per-instance predictions and the configured local checkpoints are absent. Official COCO segmentation AP, AP50, AP75, AR, and per-category metrics remain unavailable. Keep the existing L model choice provisional under its prior custom metric; this audit supplies no native-mask evidence to change it.

## Map-level pipeline acceptance

The original 40-frame `room0` map had 199 cached 2D relation tuples and zero 3D edges. The reviewed mapper correction now serializes 32 edges from those cached tuples; all 64 endpoints resolve in the 626-object map. This confirms relation propagation for the fixed CUDA run. It does not establish relation precision or recall, and each surviving edge has support count one.

The broader fixed-configuration run covers 200 of 2,000 frames at stride 10. It serialized 30 edges with resolvable endpoints from 1,109 cached relation tuples, but map aggregation and weak-edge pruning prevent a recall interpretation. The map contains 2,794 tracks and 3,185 observations from 4,502 detections. Of 211 repeated tracks, 102 (48.3%) contain differing observed class IDs; 31 tracks contain duplicate frame IDs. Per-observation caption names differ from final track classes on 144/3,185 observations (4.52%). No instance ground truth is available, so identity precision/recall and caption accuracy remain unmeasured. Thresholds were not tuned. The 200-frame result is 10% sampled coverage, not full-scene or multi-scene acceptance.

The report therefore supports accepting the relation-flow fix while keeping map-level association open. The output JSONs and mapper source change are committed separately from component-model selection. The full 443 MB PCD remains at its documented local artifact path; compact object and edge JSONs are committed.

## Caption consolidation substitutions

The existing 40-track sample contains five multi-caption and 35 single-caption tracks. It compares each candidate with cached Qwen3.8 captions, so word F1 and exact match are reference agreement rather than semantic accuracy. Gemma 4 12B produced the highest 40-track word F1 (0.966) and exact match (60.0%); it scored 0.893 word F1 and one exact match on five multi-caption tracks. Qwen3.5 Q4 was fastest at 0.2009 seconds per response, with 0.946 word F1 and 50.0% exact match. Qwen3.5 Q8 measured 0.3002 seconds, 0.940 word F1 and 42.5% exact match. All three consolidator candidates were valid on the 40-track sample.

Keep Gemma as the current configured consolidator on the available pilot, but do not call it a semantic winner. The multi-caption subset is small and the source map includes inconsistent track captions. Separate per-model reports retain the run settings and raw JSON. The Qwen3.8 reference has no measured candidate inference latency.

## Promotion decision

Promote the reports and raw artifacts as evidence for the next feature branch. Do not promote model-selection or threshold changes based on these optional evaluations. Keep the core model choices fixed. The relation-flow correction is accepted by the saved CUDA map evidence, while object association still lacks identity ground truth. The retrieval result supports the existing SigLIP2/FG-CLIP2 role split on the available oracle-mask subset. Query VQA, native masks and optional methods remain open or blocked. The report page labels these evidence classes separately.

No tests were added or run. GPU measurements cited here came from real CUDA runs in host context; feasibility-only checks made no model call.
