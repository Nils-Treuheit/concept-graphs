# Edge-compatible ConceptGraphs route

This route keeps the current ConceptGraphs mapping and prompts. It changes the
local model server behind the existing LLM and VLM clients to one selected
GGUF model. Only one candidate model is served at a time.

The selected files include each model's `mmproj-F16.gguf`, so relation and
object-caption requests can use the same annotated images as the current
Qwen3.8 route. The text consolidation stage uses that same candidate model.

## Download

Preview one model first:

```bash
.venv/bin/python scripts/edge-compatible/download_models.py --models qwen35-9b-q4
```

Download it:

```bash
.venv/bin/python scripts/edge-compatible/download_models.py --models qwen35-9b-q4 --download
```

Available model IDs are `qwen35-9b-q4`, `qwen35-9b-q8`, `gemma4-12b`, and
`gemma4-26b-a4b`. `--models all` downloads about 43 GB. The script prints the
selection and required disk space before downloading. It records the exact Hub
revision in `manifest.json` and leaves existing files untouched.

The default destination is
`/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/edge-compatible-gguf`.
Set `CG_EDGE_MODEL_DIR` or pass `--output-dir` to change it.

## Start one candidate

Build a private CUDA-enabled `llama-server` if one is not already available:

```bash
scripts/edge-compatible/build_llama_cpp_cuda.sh
```

The build requires `git`, CMake, and the CUDA toolkit. It does not install
system-wide. Set `LLAMA_CPP_SOURCE_DIR` to choose its checkout and
`LLAMA_BUILD_JOBS` to limit compilation parallelism. Then start one candidate:

```bash
LLAMA_SERVER_BIN=/path/to/llama-server scripts/edge-compatible/serve_model.sh qwen35-9b-q4
```

The server runs on `127.0.0.1:20262`, requests every model layer and the image
projector on CUDA, uses a 16K context, and refuses a CPU-only llama.cpp build.
The launcher first calls the configured agentic workflow's `release-local`
command, then checks CUDA availability. Stop the server with Ctrl-C before
starting another candidate. Set `EDGE_COMPATIBLE_CONTEXT_SIZE` to change the
context budget.

## Run the pipeline

In a second terminal, use the existing mapping path through the edge-compatible
profile:

```bash
scripts/run_pipeline_edge_compatible.sh --scene room0 --stride 50
```

It writes to its own edge-compatible detection and mapping experiments. The
first run performs detections; later runs can reuse that candidate's cache.
Pipeline arguments such as `--no-consolidation` pass through unchanged.

## Run the paired Replica benchmark

The saved `s_detections_stride50` outputs and `r_mapping_stride50` map provide
the Qwen3.8 reference. The benchmark samples five annotated frames and five
objects by default, sending the candidate the same prompts and inputs. It
measures relation agreement, caption ID coverage and text overlap, response
latency, output validity, and caption-consolidation agreement.

```bash
.venv/bin/python -m conceptgraph.scripts.benchmark_edge_compatible \
  --scene-root /mnt/HDD1/Project_Data/public_datasets/Replica/room0 \
  --detections-exp-suffix s_detections_stride50 \
  --mapping-exp-suffix r_mapping_stride50 \
  --candidate-id qwen35-9b-q4 \
  --model edge-compatible \
  --frames 5 \
  --objects 5
```

Results are written under
`.agentic_workflow/workspace/edge-compatible-benchmark/results/`. The report
labels the baseline outputs as cached Qwen3.8 responses. Agreement is not
ground-truth accuracy; inspect the saved raw responses before choosing a model.
Use a different `--candidate-id` for each model so result files stay separate.

## Candidate sizes and model sources

| ID | Weight file | Approximate weights plus projector |
|---|---|---:|
| `qwen35-9b-q4` | Qwen3.5 9B UD-Q4_K_XL | 6.9 GB |
| `qwen35-9b-q8` | Qwen3.5 9B UD-Q8_K_XL | 13.9 GB |
| `gemma4-12b` | Gemma 4 12B QAT UD-Q4_K_XL | 6.9 GB |
| `gemma4-26b-a4b` | Gemma 4 26B-A4B QAT UD-Q4_K_XL | 15.4 GB |

Model cards: [Qwen3.5 9B GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF),
[Gemma 4 12B QAT GGUF](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF),
and [Gemma 4 26B-A4B QAT GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF).
The server uses llama.cpp's [OpenAI-compatible API and multimodal projector
support](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
