# Room0 map association audit: duplicate join blocker

**Date:** 2026-11-17
**Status:** Blocked -- no scoring metrics produced.

## Overview

A CPU-only evaluator was executed against the saved 200-frame stride-10 `room0` map PCD to assess how consistently the map associates its observations to ReplicaSSG semantic instance IDs. The run terminated at the planned fail-closed guard: it found duplicate `(track_id, source_frame)` observation groups before reaching semantic mesh loading, mask rendering, Hungarian matching, or identity scoring. No IoU threshold, precision, recall, or identity metric is available.

## Configuration

| Item | Value |
|---|---|
| Branch | `pipeline-model-substitution-experiments` |
| Candidate | `.agentic_workflow/workspace/pipeline-experiments/map-association-eval-20261116/outputs/scripts/pipeline_model_reports/benchmarks/room0_map_track_identity.py` |
| Candidate SHA-256 | `04f0340dc3297c380632422b107c3976e70086bd79efae36e953747ca39b8ee5` |
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
| Annotations | `/mnt/HDD1/Project_Data/public_datasets/Replica_benchmark/ReplicaSSG/files/objects.json` | — | `3bceb3d838c7992ec7099dc4b46b49346b4961154376bff5748b1d7c5e96d5` |

## Actual metrics state

**No scoring metrics were produced.** The evaluator's fail-closed guard triggered before reaching the rendering, matching, or identity prediction stages. The result JSON contains `metrics_produced: false` and `scope_statement: "PCD validation failed; no metrics produced."`

The intended metrics -- per-frame mask precision, recall, F1; matched-observation IoU statistics; per-observation identity top-1 and coverage-adjusted top-1; and per-scene GT-object fragmentation counts -- were never computed. The planned IoU threshold of 0.50 was not evaluated.

## Exact failure

The PCD validation step found **36 duplicate `(track_id, source_frame)` groups** containing **39 extra records** among the 3,185 total observations and 2,794 tracks. The accepted evaluation protocol requires unique observation joins (each map observation should join to exactly one source frame), so the evaluator stopped before any scoring.

All 36 duplicate groups are recorded in the result JSON under `config.duplicate_join_groups_detail`. Notable cases include:

- Track `d295723d-5094-4e9f-aa4b-597d46fc35e3` across six groups (frames 1490, 1500, 1510, 1520, 1540, 1550): one group (frame 1500) has count 3 and the other five have count 2, totaling seven extra records.
- Track `125212fc-75e1-468a-ae9c-58a0c6366144` at frame 320: count 4, adding three extras.
- Remaining 29 groups have count 2 each, adding 29 extras.

The fail-closed error message was:

```
Duplicate observation joins found: 36 groups, 39 extra keys. Metrics prevented.
```

## Runtime and resources

| Metric | Value |
|---|---|
| Elapsed time | 10.640 s |
| Sampled RSS | 6,206,808,064 bytes (~5.78 GiB) |
| GPU memory | Not measured (CPU-only run) |

The RSS field in the result JSON (`peak_rss_bytes`) samples current RSS at the point of failure; it is not a tracked peak over the full run. It should be described as sampled RSS.

## Artifacts and hashes

| Artifact | Path | SHA-256 |
|---|---|---|
| Result JSON | `models/experiments/t_ren/data/map_assoc_room0/room0_identity_result.json` | `e8e939c51be68ba37206cc53820c35ea0f75218eac6e3ff4ae328a140f9eedd1` |
| PCD (input) | `.agentic_workflow/workspace/room0-relation-propagation-test/data/Replica/room0/exps/r_room0_stride10_siglip2so400m_qwen38_pcd/pcd_r_room0_stride10_siglip2so400m_qwen38_pcd.pkl.gz` | `32c91c66ffb2deea030630b4ca560abbe53e51de76e0a89916ee7555048392d9` |
| Candidate script | `.agentic_workflow/workspace/pipeline-experiments/map-association-eval-20261116/outputs/scripts/pipeline_model_reports/benchmarks/room0_map_track_identity.py` | `04f0340dc3297c380632422b107c3976e70086bd79efae36e953747ca39b8ee5` |

No overlay artifacts were written. The PCD remains at its existing local HDD path; it was not copied into Git or `/tmp`.

## Limitations

1. This audit evaluates a single saved `room0` map (200 sampled frames, 10% of the 2,000-frame sequence). Results do not establish full-scene coverage or performance on other scenes.
2. The duplicate join count (36 groups, 39 extra records) does not identify the root cause: it could arise from duplicate detection entries in the cache, mapper-level re-projection duplicates, or an observation-creation path that stores multiple mask records per `(track, frame)`. The evaluator only reports the counts; it does not analyze mask content overlap between duplicates.
3. No overlay images were generated because the run terminated before rendering. Root-specified overlay inspection frames (0, 40, 530, 1060, 1550) cannot be produced until the join issue is resolved.
4. The evaluator candidate passed `py_compile` and received independent adjudication approval for the fail-closed run and result structure. The RSS naming caveat (sampled current RSS vs. tracked peak) was noted.

## Next step

Inspect whether the repeated `(track_id, source_frame)` rows in the PCD can be normalized -- for example, by merging duplicate masks with `np.logical_or` per group, or by selecting a deterministic single record -- without violating the accepted evaluation protocol, which requires unique observation joins. If normalization is protocol-compliant, re-run the evaluator on the deduplicated PCD. If the duplicates are structurally intended and cannot be normalized under the protocol, document the constraint and adjust the evaluation contract.

This step does not invent scores or assume that deduplication will yield a passable IoU threshold. It is a narrow data-integrity check before any further scoring effort.
