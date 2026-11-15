# Caption consolidation candidate: Qwen3.5 9B UD-Q4_K_XL

**Benchmark status:** real CUDA pilot completed. This is a lexical-agreement and output-validity result, not an independent caption-accuracy score.

## Configuration and data

- Model: Qwen3.5 9B UD-Q4_K_XL; endpoint alias `edge-compatible-qwen35-9b-q4`.
- Backend: managed edge-compatible llama.cpp OpenAI-compatible server on the local RTX 5090; one model at a time.
- Generation: 256-token output cap. Accepted pilot used reasoning off. The initial default-reasoning attempt is reported separately as a configuration failure.
- Dataset: Replica `room0`, same cached 40-frame map and Qwen3.8 reference as the other candidates. The paired sample contains 40 tracks: five multi-caption and 35 single-caption tracks.
- Map SHA-256: `0b0c6f03db84270f21866e03ae84a394fe9f13d81d203ff3499bbd783a7fa502`.
- Exact provider sampling, server build/version, model load time, and peak memory were not captured. Reported latency is mean request time after model load.

## Results against the Qwen3.8 reference

| Sample | Valid outputs | Mean latency | Word F1 | Exact match |
|---|---:|---:|---:|---:|
| Failure-focused four groups, reasoning off | 4/4 | 0.2513 s | 0.741 | Not reported |
| Paired 40-track sample | 40/40 | 0.2009 s | 0.946 | 50.0% |
| Five multi-caption tracks within 40 | 5/5 included | Not separately timed | 0.730 | 0/5 |

The first four-group attempt with default server reasoning produced 0/4 valid results at 2.7905 s mean response time. The response budget was spent on hidden reasoning. The reasoning-off run fixed format validity; the initial result is a configuration failure, not an accuracy comparison.

The 40-track score is dominated by 35 singleton tracks. Qwen3.5 Q4 was fastest among accepted candidates, but it did not lead lexical agreement. The baseline is Qwen3.8 itself, so these scores do not establish semantic superiority.

## Evidence

- [Four-group default-reasoning attempt](../data/room0-qwen35-9b-q4-caption-consolidation.json)
- [Four-group reasoning-off result](../data/room0-qwen35-9b-q4-caption-consolidation-no-reasoning.json)
- [Paired 40-track result](../data/room0-qwen35-9b-q4-caption-consolidation-40.json)
- [Sample manifest](../data/room0-40-case-sample.json)
- [Cross-candidate benchmark summary](caption_consolidation.md)
