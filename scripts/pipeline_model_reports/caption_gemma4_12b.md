# Caption consolidation candidate: Gemma 4 12B QAT UD-Q4_K_XL

**Benchmark status:** real CUDA pilot completed. The result supports the configured consolidator as a lexical-agreement choice, not as a proven semantic accuracy winner.

## Configuration and data

- Model: Gemma 4 12B QAT UD-Q4_K_XL; endpoint alias `edge-compatible-gemma4-12b`.
- Backend: managed edge-compatible llama.cpp OpenAI-compatible server on the local RTX 5090; one model at a time.
- Generation: 256-token output cap; reasoning off.
- Dataset: Replica `room0`, same cached 40-frame map and Qwen3.8 reference as the other candidates. The paired sample contains five multi-caption tracks and 35 single-caption tracks.
- Map SHA-256: `0b0c6f03db84270f21866e03ae84a394fe9f13d81d203ff3499bbd783a7fa502`.
- Exact server build/version, load time, sampling settings, and peak GPU memory were not captured. Latency is mean request time after model load.

## Results against the Qwen3.8 reference

| Sample | Valid outputs | Mean latency | Word F1 | Exact match |
|---|---:|---:|---:|---:|
| Failure-focused four groups | 4/4 | 0.3178 s | 0.903 | Not reported |
| Paired 40-track sample | 40/40 | 0.3020 s | **0.966** | **60.0%** |
| Five multi-caption tracks within 40 | 5/5 included | Not separately timed | **0.893** | **1/5** |

Gemma led both lexical-agreement measures on the paired 40-track sample and the small failure-focused pilot. Five multi-caption tracks are too few for a broad semantic conclusion, and source captions include known track inconsistencies. The result justifies retaining the current Gemma consolidator provisionally; it does not establish factual accuracy against independent references.

## Evidence

- [Four-group result](data/room0-gemma4-12b-caption-consolidation.json)
- [Paired 40-track result](data/room0-gemma4-12b-caption-consolidation-40.json)
- [Sample manifest](data/room0-40-case-sample.json)
- [Cross-candidate benchmark summary](caption_consolidation.md)
