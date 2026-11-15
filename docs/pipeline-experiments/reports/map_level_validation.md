# Map-level pipeline validation

**Status:** 3D relation propagation is confirmed on CUDA, and a 200-frame stride-10 `room0` mapping run is complete. The larger map retains relations but exposes fragmented object tracks and caption/class disagreement. No instance ground truth is available, so map-level identity accuracy remains unmeasured. Component model selection is reported separately.

## 3D relation propagation

The original 40-frame Replica `room0` sample contains 199 cached 2D relation tuples. Its saved map had zero 3D edges. Source inspection found three relevant mapper defects: first-frame relations bypassed `process_edges`; filtered relation rows could be shifted when point-cloud construction removed detections; and the edge caller used the post-merge object count, so new observations could receive invalid map indices.

The correction maps relation label suffixes through filtered detection labels and each retained detection's `mask_idx`. It captures the object count before merging and processes first-frame edges in both real-time mapper paths. The relation-index correction received a root static review and independent read-only adjudication. No relation thresholds or stale-edge pruning behavior were changed.

The accepted integration test initially reached the real mapper in host CUDA context but stopped at its required startup guard: it found 29.65 GiB free against a 30 GiB minimum. We preserved the failure and retried after GPU headroom improved. The successful host CUDA retry processed all 40 frames in about 29 seconds, mapped 626 objects, and serialized 32 edges from the 199 cached relations. All 64 edge endpoint references resolved to map objects, and all edge types were present in the output relation vocabulary. This confirms 2D-to-3D propagation and serialization for the fixed run configuration. It does not measure relation precision or recall.

Every output edge has `num_detections=1`. In this implementation the field counts support for one exact ordered map-object pair. It increments when the same pair recurs; merge index remapping sums counts, while the existing pruning rule can remove weak edges. A value of one therefore means no surviving repeated mapped pair was observed in this output, not that the field is a presence flag.

### Source-to-map mismatch trace

An independent read-only trace examined the closest available per-observation history: an earlier PCD from the same source detections and mapper configuration. In source frame `001700` (loop index 34), the cached relation connects row 16 (`potted plant`) to row 12 (`coffee table`). The earlier PCD maps row 16 to object 556 (`potted plant`) and row 12 to object 325 (`couch`). The retry edge JSON contains the same object IDs and tags as `potted plant on top of couch`.

The retry itself used `save_pcd=false`, so it does not preserve per-observation provenance. This source-to-map trace is strong evidence of an association or class mismatch, but it is not conclusive for the retry run. The earlier PCD also showed four plant/table observations mapping to four different object pairs. Across the prior PCD, 146 relation tuples mapped to 146 unique object pairs; 53 tuples had no surviving PCD row. These observations are consistent with fragmented tracks preventing repeated-edge support accumulation. They do not identify whether each lost or reassigned observation resulted from semantic detection error, spatial association, projection filtering, or pruning.

## Map-level association evidence

The completed 200-frame stride-10 map contains 2,794 final tracks and 3,185 retained observations from 4,502 cached detections. Track lengths are 2,583 singletons; 146 length two; 25 length three; 19 length four; seven length five; six length six; three length seven; two length eight; two length eleven; and one length twenty-nine. Thus 211 tracks have more than one observation. The 1,317-observation difference spans RGB-D projection, filtering, denoising, and map processing; retained artifacts do not provide per-stage counters, so it cannot be assigned to one cause.

Across the 211 multi-observation tracks, 102 (48.3%) contain differing observed class IDs, and 31 tracks contain duplicate frame IDs. Of 3,185 retained per-observation captions, 144 (4.52%) have a caption `name` different from the final track `class_name`; these affect 102 tracks. Caption consolidation was disabled for this map, so consolidated object captions are empty by design. The per-frame caption records remain in the PCD. These are consistency diagnostics, not identity or semantic accuracy scores.

The final edge JSON has 30 directed map edges: 29 `on top of`, one `under`. Every endpoint resolves to an object, and each endpoint tag matches the object JSON. Support counts are 19 edges with one observation, nine with two, one with three, and one with five. The 1,109 cached 2D relation tuples are not a 1:1 target for the 30 map edges: mapping aggregates exact ordered object pairs and prunes weak edges. Therefore no relation recall or graph recovery score is reported. Available `room0` data has RGB-D, trajectory, and mesh files but no ready-to-use per-frame instance-ID annotations. No matching threshold was tuned without an evaluable identity set.

The separate 40-frame retry saved JSON but no PCD observation history. The 200-frame stride-10 map now saves its full PCD, object JSON, edge JSON, and all-frame object snapshots, enabling the descriptive analysis above. It is 10% sampled coverage of the 2,000-frame sequence, not full-scene coverage.

## Broader end-to-end coverage

The original stride-50 cache samples 40 of 2,000 `room0` frames (2%). The corrected stride-10 run covers 200 sampled frames, or 10% of that sequence, on the same scene. It completed detection, per-frame annotation, and CUDA mapping. This is broader end-to-end coverage, but not full-scene validation and not an additional-scene replication.

Retrieval-backend and query-time VLM evaluations remain outside the validated core path. The matched per-frame VLM shortlist and separate per-model reports are available under `scripts/pipeline_model_reports/`; they measure output agreement and schema reliability, not map-level acceptance.

## Configuration and artifacts

The original 40-frame cache is `s_modernized_yoloe26l_stride50`, generated with YOLOE-26-L-seg, SigLIP2 So400m masked features (`google/siglip2-so400m-patch14-384`), and FG-CLIP2 So400m proposal features (`qihoo360/fg-clip2-so400m`). Its mapper uses CUDA, `match_method=sim_sum`, `spatial_sim_type=overlap`, `sim_threshold=1.2`, `phys_bias=0.0`, `mask_area_threshold=25`, `mask_conf_threshold=0.25`, and `max_bbox_area_ratio=0.5`.

| Work item | Dataset and configuration | Runtime / resources | Result | Artifact |
|---|---|---|---|---|
| Cached-data audit | Replica `room0`, 40 frames, stride 50 | CPU file audit; no model loaded | 909 masks, 199 2D tuples, original saved map 626 objects / 634 observations / 0 edges | `/mnt/HDD1/Project_Data/public_datasets/Replica/room0/exps/s_modernized_yoloe26l_stride50/detections`; `/mnt/HDD1/Project_Data/public_datasets/Replica/room0/exps/r_modernized_yoloe26l_stride50/pcd_r_modernized_yoloe26l_stride50.pkl.gz` |
| Initial relation retry | Same 40-frame cache, CUDA, `make_edges=false`, JSON save | Failed before frame 0 at 29.65 GiB free; 30 GiB required | Resource guard correctly prevented execution | `.agentic_workflow/workspace/room0-relation-propagation-test/mapper.log` |
| Relation propagation retry | Same 40-frame cache, CUDA, `make_edges=false`, JSON save, PCD save disabled | About 29 s frame loop; startup logged 1.58 GiB used / 30.26 GiB free | 626 map objects, 32 relation edges; all endpoint references resolve | `.agentic_workflow/workspace/room0-relation-propagation-test/data/Replica/room0/exps/r_room0_relation_propagation_acceptance_retry/` |
| Stride-10 detection | Replica `room0`, 200 sampled frames, fixed YOLOE-26-L + SigLIP2 So400m + FG-CLIP2 So400m | 4:00; peak logged GPU use about 4.88 GiB total | Completed; 200 caches contain 4,502 detections | `.agentic_workflow/workspace/pipeline-experiments/detection-room0-stride10-so400m.log`; cache under `.agentic_workflow/workspace/room0-relation-propagation-test/data/Replica/room0/exps/s_modernized_yoloe26l_stride10_siglip2so400m/` |
| Stride-10 Qwen3.8 annotation | Same 200 frames and detections; managed Qwen3.8 UD Q4_K_XL MTP4, 131,072 context, 4,096 output cap; xhigh reasoning | Segmented host runs; exact aggregate wall time unavailable. Logs record ~23.9 GiB model VRAM and about 25.5 GiB total GPU use. Three longer invocations stopped when model-control reported the model unloaded; completed frames were preserved and resumed in smaller chunks. | Completed; all 200 caches contain edges and captions: 1,109 relation tuples and 3,606 caption records | `.agentic_workflow/workspace/pipeline-experiments/vlm-room0-stride10-*.log`; detector cache path above |
| Stride-10 CUDA mapping | Same 200 annotated caches; CUDA, overlap spatial similarity, `sim_sum`, threshold 1.2, `phys_bias=0`, mask area 25, confidence .25, max bbox ratio .5; denoise/filter/merge every five frames; `make_edges=false`, consolidation disabled, PCD/JSON/all-frame snapshots enabled | 30:12 frame loop; logged peak about 14.58 GiB total GPU use (13.00 GiB mapper process); no mapping error | Completed; 2,794 map objects, 3,185 observations, 30 serialized edges; endpoint references and tags resolve | `.agentic_workflow/workspace/pipeline-experiments/mapper-room0-stride10-qwen38-pcd.log`; output under `.agentic_workflow/workspace/room0-relation-propagation-test/data/Replica/room0/exps/r_room0_stride10_siglip2so400m_qwen38_pcd/` |

The 200-frame mapper's successful log and output artifacts are retained. Its frame loop took 30:12; the full command was run in host CUDA context after releasing this workflow's runner and confirming model-control idle. Peak memory is from mapper logs. The earlier 40-frame propagation retry still lacks its successful log, so that run's approximate runtime and startup memory remain transcript-reported.

## Committed compact artifacts

The following compact outputs are committed with this report. The full 200-frame PCD remains at the local output path listed above and is not duplicated in Git because it is approximately 443 MB.

- Original zero-edge 40-frame map: [object JSON](../data/map_level/baseline40_objects.json), [edge JSON](../data/map_level/baseline40_edges_zero.json), and [configuration](../data/map_level/baseline40_config.json).
- Corrected 40-frame relation propagation: [object JSON](../data/map_level/propagation_retry40_objects.json) and [edge JSON](../data/map_level/propagation_retry40_edges.json). Its successful console log was not retained; runtime remains transcript-reported.
- Broader 200-frame map: [object JSON](../data/map_level/stride10_200_objects.json) and [edge JSON](../data/map_level/stride10_200_edges.json). Its full detector, annotation, and mapper logs remain at the local paths listed in the table above.
- Initial resource-guard result: `.agentic_workflow/workspace/room0-relation-propagation-test/mapper.log`.

## Separate OpenCode implementation attempt

The local OpenCode/NInfer Qwen3.8 implementation handoff used the configured system route and repository host context. Its provider repeatedly returned `ECONNREFUSED`; no model-authored patch was produced. The owned session was closed and its managed model released before GPU experiments continued. Evidence: `.agentic_workflow/workspace/pipeline-experiments/implementation-qwen38-cfb0dfe32e4546ba8f78ae728a605c53.session.jsonl` and `/home/nils/.local/share/model-unload/ninfer-qwen3.8-27b-nvfp4.log`.
