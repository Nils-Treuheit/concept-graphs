# Caption consolidation reference: Qwen3.8 27B

**Role:** cached reference captions, not a newly timed consolidator candidate.
**Result status:** no independent semantic accuracy or query latency measurement.

## Configuration and data

- Model: `qwen3.8:27b-mtp4-ud_q4_K_XL-128k`, UD Q4_K_XL, MTP4.
- Dataset: Replica `room0`; the source map contains 626 tracks. The paired comparison samples 40 tracks, comprising five multi-caption tracks and 35 single-caption tracks.
- Map SHA-256: `0b0c6f03db84270f21866e03ae84a394fe9f13d81d203ff3499bbd783a7fa502`.
- Baseline provenance: cached outputs from the original 40-frame Qwen3.8 mapping, with four blank groups replaced by a prior real-GPU rerun using reasoning effort `none`.
- Candidate inference settings: not applicable to the cached reference. Baseline latency and GPU memory were not measured in the paired comparison.

## Comparison outcome

All candidate word-F1 and exact-match scores are agreement with these Qwen3.8 captions. They do not measure factual accuracy. The five multi-caption tracks contain known inconsistent source captions, which limits their value as semantic references.

| Requested measure | Result |
|---|---:|
| Reference outputs | Cached; independent validation not performed in this comparison |
| Mean inference latency | Not measured |
| Peak GPU memory | Not measured |
| Benchmark role | Reference only, not a candidate winner |

## Evidence

- [Cached Qwen3.8 reference JSON](data/room0-qwen3.8-baseline-40.json)
- [40-track sample manifest](data/room0-40-case-sample.json)
- [Cross-candidate benchmark summary](caption_consolidation.md)
