# Room0 map association audit: mask-association benchmark

**Date:** 2026-11-17
**Status:** Completed -- mask association scored; identity top-1 unestimable (zero eligible).

## Overview

A CPU-only evaluator was executed against the saved 200-frame stride-10 `room0` map PCD to assess how consistently the map associates its observations to ReplicaSSG semantic instance IDs. The first attempt used a fail-closed guard on duplicate `(track_id, source_frame)` observation groups. That approach was superseded by a logical-OR mask-union aggregation, which groups each duplicate group, unions all mask pixels with `np.logical_or`, and produces one prediction per unique `(track_id, source_frame)` unit. The run completed end-to-end: mesh loading, raycasting, Hungarian IoU matching, aggregate scoring, identity prediction, and five visual overlays were all produced.

Unioning each duplicate group preserved all mask pixels across the 39 extra components; no prediction was discarded. The aggregated result had 3,146 unique track-frame prediction units from 3,185 raw mask rows. The IoU threshold of 0.50 produced 11 matches. Identity prediction found zero eligible predictions, so top-1 is JSON null with status `not_estimable_zero_eligible`. Coverage and coverage-adjusted top-1 are both zero.

## Configuration

| Item | Value |
|---|---|
| Branch | `pipeline-model-substitution-experiments` |
| Candidate | `scripts/pipeline_model_reports/benchmarks/room0_map_track_identity.py` |
| Candidate SHA-256 | `8a039304fd2e1c5af964772a452446530eec03ab911e9cecd13784f7d5a6403c` |
| Python | Python 3 (project `.venv`: NumPy 2.5.4, SciPy 1.18.1, Open3D 0.20.0, Pillow 13.0.0, psutil 7.2.2) |
| I/O threshold | IoU ≥ 0.50 for one-to-one matching via `scipy.optimize.linear_sum_assignment` |
| Overlay frames | 0, 40, 530, 1060, 1550 |
| Execution context | Host CPU only; no model loaded, no GPU, no network call, no `/tmp` artifacts |

## Input artifacts

| Artifact | Path | Size | SHA-256 |
|---|---|---|---|
| Map PCD | `.agentic_workflow/workspace/room0-relation-propagation-test/data/Replica/room0/exps/r_room0_stride10_siglip2so400m_qwen38_pcd/pcd_r_room0_stride10_siglip2so400m_qwen38_pcd.pkl.gz` | ~443 MB | `32c91c66ffb2deea030630b4ca560abbe53e51de76e0a89916ee7555048392d9` |
| PLY mesh | `/mnt/HDD1/Project_Data/public_datasets/Replica_benchmark/replica_v1/room_0/habitat/mesh_semantic.ply` | 43,890,939 B | `104cc7fe28bee178537e8034859e8f121f89a166ea37132b4574e10050947d98` |
| Trajectory | `/mnt/HDD1/Project_Data/public_datasets/Replica/room0/traj.txt` | 811,033 B | `5692a60076e426b403dc64025ff38c443606a506704545ee192d4dbadedaaca2` |
| Intrinsics | `/mnt/HDD1/Project_Data/public_datasets/Replica/cam_params.json` | — | `ffbefdeb53755e1490d6936dffdee253510e68d5e443e1ba91e60202addc1a0a` |
| Annotations | `/mnt/HDD1/Project_Data/public_datasets/Replica_benchmark/ReplicaSSG/files/objects.json` | — | `3bceb3d838c7992ec7099dc4b46b49346b4961154376bff5748b1d1b7c5e96d5` |

## Measured result

One saved Replica room0 map on 200 source frames:

| Metric | Value |
|---|---|
| Map tracks | 2,794 |
| Raw mask rows | 3,185 |
| Unique track-frame units | 3,146 |
| Duplicate groups / extra components | 36 groups / 39 extra (preserved by union) |
| One-to-one Hungarian matches (IoU ≥ 0.50) | 11 |
| Macro precision | 0.0038289714671293617 |
| Macro recall | 0.004484734556211202 |
| Macro F1 | 0.0038090027240922386 |
| Micro precision | 0.0034965034965034965 |
| Micro recall | 0.0038596491228070177 |
| Mean matched IoU | 0.589284051110483 |
| Median matched IoU | 0.5487957573530834 |
| Identity eligible | 0 |
| Identity top-1 | JSON null |
| Identity status | `not_estimable_zero_eligible` |
| Coverage | 0 |
| Coverage-adjusted top-1 | 0 |
| Runtime | 39.5304 s |
| Sampled RSS | 9,167,642,624 bytes (8,742.95 MiB) |
| Unmatched predictions | 3,135 |
| Unmatched GT occurrences | 2,839 |
| Unmatched unique GT IDs | 71 |
| Matched-IoU standard deviation | 0.0887473772191468 |
| Minimum matched IoU | 0.5009539349011698 |
| Maximum matched IoU | 0.798864508829353 |
| Frames with empty GT/prediction denominator | 0 |
| Identity tie count | 0 |

Fragmentation: among the GT IDs appearing in matched pairs, six IDs were seen; three were matched to multiple map tracks; maximum tracks per matched GT ID was three. Exact GT-ID-to-track counts: `22: 3`, `21: 1`, `64: 1`, `50: 2`, `30: 3`, `63: 1`.

Mask association completed and produced precision/recall/F1/IoU metrics. Identity prediction is unestimable because zero matched track-frame units had prior-frame GT history to support majority-vote prediction.

## Superseded failure

The first implementation used a fail-closed guard that stopped evaluation on the 36 duplicate `(track_id, source_frame)` groups (39 extra records). The guard was removed in favor of a logical-OR mask-union strategy: each duplicate group is collapsed to a single prediction by unioning all component masks with `np.logical_or`, preserving every mask pixel from every component. The union aggregation step recorded 3,146 unique track-frame units from 3,185 raw rows without discarding any mask content.

## Failures

None. The completed run produced all artifacts without errors.

## Artifacts and hashes

| Artifact | Path | SHA-256 |
|---|---|---|
| Result JSON | `models/experiments/t_ren/data/map_assoc_room0/room0_identity_result.json` | `0ac0af125e4faaff270f61cd97974593a4dfd1e1bc5d5764655e7d06f35f522c` |
| Map PCD (input) | `.agentic_workflow/workspace/room0-relation-propagation-test/data/Replica/room0/exps/r_room0_stride10_siglip2so400m_qwen38_pcd/pcd_r_room0_stride10_siglip2so400m_qwen38_pcd.pkl.gz` | `32c91c66ffb2deea030630b4ca560abbe53e51de76e0a89916ee7555048392d9` |
| Tracked evaluator | `scripts/pipeline_model_reports/benchmarks/room0_map_track_identity.py` | `8a039304fd2e1c5af964772a452446530eec03ab911e9cecd13784f7d5a6403c` |

### Overlay images

| Frame | Path | SHA-256 |
|---|---|---|
| 0 | `.agentic_workflow/workspace/pipeline-experiments/map-association-eval-20261116/outputs/overlays/room0_identity_0000.png` | `c72fac4f81bc0280d4f344a63d9b57729edb6245d6a94c90eeacb9ea3dc486ff` |
| 40 | `.agentic_workflow/workspace/pipeline-experiments/map-association-eval-20261116/outputs/overlays/room0_identity_0040.png` | `6b88231dd68f00bede4367ffec91c557c6bdf0b65e8bf96f7421238a2918746b` |
| 530 | `.agentic_workflow/workspace/pipeline-experiments/map-association-eval-20261116/outputs/overlays/room0_identity_0530.png` | `6324a057919b0de2d33f27ed6170ee4db9c0e13d36f59d6fb0008bd3c1bf00b4` |
| 1060 | `.agentic_workflow/workspace/pipeline-experiments/map-association-eval-20261116/outputs/overlays/room0_identity_1060.png` | `0aabc6457c07d94ee4cd8aef0821a71d34aefc023cf5385d7f9acedd6d02bd97` |
| 1550 | `.agentic_workflow/workspace/pipeline-experiments/map-association-eval-20261116/outputs/overlays/room0_identity_1550.png` | `97068eb9dce2443d31e5daaaba713011cfcc7f8e42d941727d0187889b1bbf32` |

## Limitations

1. This audit evaluates a single saved `room0` map (200 sampled frames, 10% of the 2,000-frame sequence). Results do not establish full-scene coverage or performance on other scenes.
2. The duplicate join count (36 groups, 39 extra records) does not identify the root cause: it could arise from duplicate detection entries in the cache, mapper-level re-projection duplicates, or an observation-creation path that stores multiple mask records per `(track, frame)`. The union aggregation preserved all mask pixels; the evaluator does not analyze mask content overlap between duplicates.
3. Identity top-1 is unestimable due to zero eligible predictions. The evaluation contract did not require identity scoring for this first map benchmark.
4. The evaluator passed strict JSON output, aggregate re-computation verification, and fresh Qwen result adjudication.
