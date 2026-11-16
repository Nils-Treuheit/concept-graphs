# DINOtxt ReplicaSSG test-5 cross-view graph-node identity and oracle predicate triplet recall — measured result

**Status:** measured and reviewed (Stage 1 identity and graph diagnostics). Stage 2 (optional pipeline-slot acceptance) remains explicitly pending.

- **Runner:** `scripts/pipeline_model_reports/benchmarks/dinotxt_replica_test5_identity.py` (candidate v5, SHA-256 `7ef2eae19ddda64a18b14ebb870db4eaaeb91865611101827252ed8ebe6c9ed6`)
- **Result:** `models/experiments/t_ren/data/dinotxt_test5/dinotxt_test5_identity_20261116T211846Z_3124965.json` (3,257,400 bytes)

**Evaluated:** 16 November 2026; independent recomputation from `per_query_ranks` and frozen `relationships.json` matched every stored value exactly.

---

## Protocol and scope

CPU-only leave-one-frame-out retrieval of object identity from cached 2048-d FP32-normalised DINOv3/dino.txt embeddings on the frozen ReplicaSSG test-5 split. Five scenes, 49 categories, 2,204 validated views, 2,171 eligible queries, 33 excluded. Identity is `(scene, int(object_id))`. For each query observation all views sharing the same frame string are excluded. Remaining normalised embeddings are averaged per object node, L2-normalised, and ranked by FP32 cosine similarity with deterministic tie-break (descending cosine, scene name, integer object ID). Graph-node identity accuracy and oracle predicate triplet recall are computed from the eligible per-query ranks and the `predicted_node_by_query` map. Predicates are frozen; predicate inference is not measured.

This is cached-crop analysis, not map association or pipeline-slot acceptance.

---

## Model and feature provenance

- **Embeddings:** cached 2048-d FP32-normalised DINOv3/dino.txt embeddings (NPZ + sidecar JSON). These were produced by a prior GPU encoder pass; this analysis consumes only the cached files on CPU.
- **Input sources:** frozen crop manifest, `ReplicaSSG/files/objects.json`, `ReplicaSSG/files/relationships.json`, and the global score JSON.
- **Runner SHA-256:** `7ef2eae19ddda64a18b14ebb870db4eaaeb91865611101827252ed8ebe6c9ed6` (candidate v5).

---

## Scene and view counts

| Scene | Views |
|---|---:|
| office_1 | 338 |
| office_3 | 553 |
| office_4 | 402 |
| room_1 | 505 |
| room_2 | 406 |
| **Total** | **2,204** |

Eligible queries: 2,171. Excluded: 33 (identity unavailable or insufficient candidates).

---

## Primary retrieval metrics (eligible queries, 2,171)

| Metric | Value |
|---|---:|
| Top-1 / R@1 | 0.1953017043 |
| MRR | 0.3393284171 |
| Recall@5 | 0.5002303086 |
| Recall@10 | 0.6642100415 |

## Coverage-adjusted metrics (all 2,204 views)

| Metric | Value |
|---|---:|
| Top-1 / R@1 | 0.1923774955 |
| MRR | 0.3342477284 |
| Recall@5 | 0.4927404719 |
| Recall@10 | 0.6542649728 |

---

## Graph-node identity

Graph-node identity accuracy is the fraction of eligible object observations where the highest-scoring leave-one-frame-out candidate matches the true node ID.

- **Overall:** 424 / 2,171 = 0.1953017043

### Per-scene

| Scene | Observations | Correct | Accuracy |
|---|---:|---:|---:|
| office_1 | 331 | 84 | 0.2537764350 |
| office_3 | 546 | 104 | 0.1904761905 |
| office_4 | 398 | 80 | 0.2010050251 |
| room_1 | 503 | 92 | 0.1829025845 |
| room_2 | 393 | 64 | 0.1628498728 |

Excluded queries: 33 total — 33 `identity_unavailable`, 0 `insufficient_candidates`.

Full per-scene counts are stored in the result JSON under `graph_node_identity_metrics.per_scene`.

---

## Oracle predicate triplet recall

This metric measures the fraction of observable frozen edges where both predicted endpoints are supported in the frame's shared leave-one-frame-out candidate set, have a non-null prediction, and match their ground-truth node IDs. **Predicates are taken directly from `relationships.json` and are not generated or evaluated.** The result 0 / 49 = 0.0 does not measure predicate inference capability.

- **Overall:** 0 / 49 = 0.0

### Per-scene edge observations

| Scene | Observations | Correct | Recall |
|---|---:|---:|---:|
| office_1 | 0 | 0 | 0.0 |
| office_3 | 0 | 0 | 0.0 |
| office_4 | 9 | 0 | 0.0 |
| room_1 | 5 | 0 | 0.0 |
| room_2 | 35 | 0 | 0.0 |

The 49-edge denominator was validated independently: the comparator's `if oid in prototypes` gate excludes one endpoint (object 52 in `office_4`, frame `frame-0775`, edge `(52, "near", 28)`) whose ground-truth node lacks a true-node prototype in that frame. The corrected denominator of 49 matches both mask-isolated control benchmarks.

---

## Superseded diagnostic run (50-edge)

An earlier v4 run (`dinotxt_test5_identity_20261116T210457Z_3092669.json`) produced a 0 / 50 support-unfiltered oracle edge result. The one extra observation in `office_4`, frame `frame-0775`, was caused by an endpoint lacking a true-node prototype in the frame's leave-one-frame-out candidate set. The v5 candidate preserves the comparator's support gate and excludes that endpoint from edge scoring. The earlier JSON remains intact on disk and is superseded for matched-edge comparison.

---

## Comparison controls

The frozen SigLIP2 So400m and FG-CLIP2 So400m mask-isolated controls each score 2,171 graph-node observations and 49 edge observations.

| Model | Graph-node accuracy | Oracle triplet recall |
|---|---:|---:|
| SigLIP2 So400m | 0.3965914325 | 0.3877551020 |
| FG-CLIP2 So400m | 0.3790879779 | 0.3265306122 |
| DINOtxt (this run) | 0.1953017043 | 0.0 |

Aggregate controls have no paired per-view predictions. Comparison is descriptive only; no paired deltas or significance claims are made.

---

## Runtime and resources

| Phase | Time (s) |
|---|---:|
| Total wall | 2.454 |
| Prototype computation | 1.757 |
| Ranking | 0.130 |

Peak host RSS: 1,267,652 KB. Output file size: 3,257,400 bytes.

- **Python:** 3.10.12
- **NumPy:** 1.26.4
- **Platform:** Linux 6.8.0-138-generic
- **Resource scope:** CPU-only analysis of cached features. Qwen residency for adjudication was not used by the benchmark; runner code labels no model or GPU operation.

---

## Input hashes

| Input | SHA-256 |
|---|---|
| Scores JSON | `4923c4c0b8f5329675a309a9ae4977fd46851151be01a754d91c6a544cda672e` |
| Crop manifest | `314f3f656394014dba4015f95c4c4e654ea8a4be6a68a702f10b7c92e14ecc15` |
| ReplicaSSG objects | `3bceb3d838c7992ec7099dc4b46b49346b4961154376bff5748b1d1b7c5e96d5` |
| ReplicaSSG relationships | `11fd5a4b2fca5c654f0d5429fd381f174fa8668840532866c1efeb150442e240` |
| Embeddings NPZ | `bf0021862f5477d961372c51907e8a504c92daf5d2247f4ede1fb6575e185c63` |
| Embedding sidecar JSON | `3665b653cd556d7c175832195055bebb00615f79c07395e4e0519221eb5203b3` |

All six hashes were independently recomputed and matched.

---

## Internal consistency

The runner performs seven flags of internal-consistency verification before metrics output:

1. `query_prototypes` keys equal all resolved queries.
2. `query_support` keys equal all resolved queries.
3. `query_candidate_counts` keys equal all resolved queries.
4. Every candidate count scalar equals the prototype map size.
5. Every prototype is finite and L2-normalised to within 1e-6.
6. All candidates are scene-local.
7. True candidate is present in the prototype map if and only if it has alternate-frame support.

All checks passed on the v5 run.

---

## Artifacts

- **v5 result JSON:** `models/experiments/t_ren/data/dinotxt_test5/dinotxt_test5_identity_20261116T211846Z_3124965.json`
- **Superseded v4 result (intact):** `models/experiments/t_ren/data/dinotxt_test5/dinotxt_test5_identity_20261116T210457Z_3092669.json`
- **Runner (candidate v5):** `scripts/pipeline_model_reports/benchmarks/dinotxt_replica_test5_identity.py`

---

## Limitations

1. Cached-crop CPU analysis only. No model load, GPU operation, or new input generation.
2. Leave-one-frame-out retrieval of graph-node identity is not map association or pipeline-slot acceptance.
3. Oracle predicate triplet recall uses frozen `relationships.json` predicates; predicate generation is not evaluated.
4. The v4 support-unfiltered 50-edge diagnostic must not be merged with the corrected 49-edge result.
5. Controls are unpaired aggregates; no per-view comparison is possible.
