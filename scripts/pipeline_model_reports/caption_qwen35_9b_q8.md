# Caption consolidation candidate: Qwen3.5 9B UD-Q8_K_XL

**Benchmark status:** real CUDA pilot completed. Agreement is measured against cached Qwen3.8 captions, not ground truth.

## Configuration and data

- Model: Qwen3.5 9B UD-Q8_K_XL; endpoint alias `edge-compatible-qwen35-9b-q8`.
- Backend: managed edge-compatible llama.cpp OpenAI-compatible server on the local RTX 5090; one model at a time.
- Generation: 256-token output cap; reasoning off.
- Dataset: Replica `room0`, same cached 40-frame map and Qwen3.8 reference as the other candidates. The 40-track sample contains five multi-caption tracks and 35 single-caption tracks.
- Map SHA-256: `0b0c6f03db84270f21866e03ae84a394fe9f13d81d203ff3499bbd783a7fa502`.
- Exact server build/version, load time, sampling settings, and peak GPU memory were not captured. Latency is mean request time after load.

## Results against the Qwen3.8 reference

| Sample | Valid outputs | Mean latency | Word F1 | Exact match |
|---|---:|---:|---:|---:|
| Failure-focused four groups | 4/4 | 0.3605 s | 0.769 | Not reported |
| Paired 40-track sample | 40/40 | 0.3002 s | 0.940 | 42.5% |
| Five multi-caption tracks within 40 | 5/5 included | Not separately timed | 0.754 | 0/5 |

Qwen3.5 Q8 was slower than Qwen3.5 Q4 and had lower 40-track lexical agreement than Gemma 4 12B. Its small four-group sample scored higher than Q4, but it is too small and reference-dependent to establish a semantic advantage. All five multi-caption outputs differed exactly from the Qwen3.8 reference.

## Evidence

- [Four-group result](data/room0-qwen35-9b-q8-caption-consolidation.json)
- [Paired 40-track result](data/room0-qwen35-9b-q8-caption-consolidation-40.json)
- [Sample manifest](data/room0-40-case-sample.json)
- [Cross-candidate benchmark summary](caption_consolidation.md)
