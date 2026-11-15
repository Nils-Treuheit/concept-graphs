# LFM3-VL-2B official Transformers BF16 benchmark

This is a report of an existing run, completed 2026-11-12. No model was run for this report.

## Configuration and sample

| Setting | Value |
|---|---|
| Checkpoint | `LiquidAI/LFM3-VL-2B` |
| Revision | `23387beda43c60310348e83e25f78f33c3140d3a` |
| Precision | BF16, official Transformers checkpoint |
| Backend | Transformers 5.22.1, `AutoModelForImageTextToText`, CUDA 13.0 |
| Configured context | 8,192 tokens |
| Generation | temperature 0.1, top-p 1.0, top-k 50, min-p 0.0, repetition penalty 1.1, maximum 4,096 new tokens, seed 3407 |
| Evaluation | Fixed 825-example shared suite; serial, one image per request; sample seed 3407 |

The run passed its real-image AI2D and TextVQA preflights. Each of the eight tasks completed all requested items.

## Results

| Task | Items | Score | Mean generation time | Output tokens/s | Parse errors |
|---|---:|---:|---:|---:|---:|
| AI2D | 100 | 93.0% | 6.00 s | 91.24 | 0 |
| TextVQA / OCR | 100 | 81.0% | 4.39 s | 91.56 | 0 |
| VQA truth | 25 | 83.6% | 4.03 s | 92.07 | 0 |
| VQA counting | 25 | 83.2% | 11.04 s | 91.97 | 0 |
| VQA description | 25 | 60.4% | 2.05 s | 91.55 | 0 |
| VQA affordance | 25 | 8.0% | 8.24 s | 91.60 | 0 |
| VQA negation | 25 | 64.0% | 11.88 s | 91.89 | 1 |
| GRAB-Lite graph analysis | 500 | 33.4% | 23.49 s | 90.74 | 0 |

The equally weighted eight-task full-suite macro was **63.325%**. The six-category VQA panel macro was **63.367%**. The raw run records 69 GRAB-Lite outputs ending at the 4,096-token limit. These were length-limited responses, not failed requests; all 500 GRAB-Lite items are recorded complete. One VQA negation output had a parse error.

## Comparison and interpretation

| Candidate and route | Completed | Full-suite macro | VQA panel macro |
|---|---:|---:|---:|
| Qwen3.8 27B, configured Ollama MTP4 UD-Q4_K_XL route | 825/825 | **63.5375%** | 65.75% |
| LFM3-VL-2B official Transformers BF16 | 825/825 | 63.325% | 63.367% |
| LFM3-VL-2B GGUF Q4_K_M | 825/825 | 55.725% | 55.367% |
| LFM3-VL-2B GGUF NVFP4 | 825/825 | 58.5375% | 57.083% |

The official BF16 LFM3 run outscored its two measured GGUF variants on the full-suite macro. It was 0.2125 percentage points below the configured Qwen3.8 route. Qwen3.8 is the highest complete full-suite score in this local-foundation comparison, but this candidate set does not establish an overall winner across VLMs or a production winner. These results use a general image-question benchmark, not ConceptGraphs annotation or mapping inputs.

## Runtime, memory, and run status

Per-task means and output throughput are shown above. The raw run does not record an overall wall-clock duration. Summing each task's mean generation time across its examples gives approximately 13,712.56 seconds of request generation; this is not end-to-end wall time and excludes non-generation overhead.

The aggregate report records **14.638 GiB peak whole-device VRAM** for this run. The raw JSON records a count of 26,662 memory samples, but no sample series, and the run-level memory-method field is unset. The aggregate report describes whole-device `nvidia-smi` sampling every 0.5 seconds, including the desktop baseline, and notes that task-level peaks were not separated. The aggregate report and `data.json` disagree on LFM3 NVFP4 peak memory (4.90 versus 5.981 GiB), so this report leaves cross-model memory comparison out of the score table.

The first attempt failed before inference because `AutoModelForMultimodalLM` did not recognize the custom LFM3-VL config; it produced zero GPU outputs. The run then switched to the official model-card `AutoModelForImageTextToText` route. The second attempt passed the preflights and completed the full suite on the RTX 5090. No requests are marked failed in the completed run.

## Artifact paths

- Aggregate report: `/mnt/HDD1/Project_Code/VLM_benchmark/local_foundation_Bench/report.md`
- Aggregate score and latency data: `/mnt/HDD1/Project_Code/VLM_benchmark/local_foundation_Bench/data.json`
- Candidate status and revision: `/mnt/HDD1/Project_Code/VLM_benchmark/local_foundation_Bench/candidate_inventory.json`
- Raw item prompts, outputs, scores, per-item timings, and the memory-sample count: `/mnt/HDD1/Project_Code/VLM_benchmark/local_foundation_Bench/results/runs/lfm3_refresh_20261112/transformers-original/results.json`
- Run log: `/mnt/HDD1/Project_Code/VLM_benchmark/local_foundation_Bench/results/runs/lfm3_refresh_20261112/transformers-original/run.log`
- Attempt history: `/mnt/HDD1/Project_Code/VLM_benchmark/local_foundation_Bench/results/runs/lfm3_refresh_20261112/transformers-original/attempts.json`
- Upstream revision/file inventory: `/mnt/HDD1/Project_Code/VLM_benchmark/local_foundation_Bench/results/runs/lfm3_refresh_20261112/transformers-original/upstream.json`
