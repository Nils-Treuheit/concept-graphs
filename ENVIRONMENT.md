# Environment setup (uv)

This project uses a **uv** virtual environment (`.venv`, Python 3.12) instead of the
conda `environment.yml`. The environment is tuned for this machine:

| | |
|---|---|
| GPU | NVIDIA RTX 5090, 32 GB, `sm_120` (Blackwell) |
| CPU | AMD Ryzen 9 7900 (Zen 4, AVX2, no AVX-512) |
| driver | CUDA 13.2 |
| toolkits | `/usr/local/cuda-12.8`, `/usr/local/cuda-13.2` (the latter is used) |

> `/usr/bin/nvcc` is Ubuntu's CUDA 11.5 wrapper — always point the build at
> `/usr/local/cuda-13.2`.

## Build flags

All builds use the flags defined in `~/.bashrc`:

```bash
export CUDA_HOME=/usr/local/cuda-13.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=4          # 24 jobs would need 6-8 GB of RAM each
export NVCC_THREADS=1
```

`CMAKE_CUDA_COMPILER` / `CUDACXX` are set explicitly to `$CUDA_HOME/bin/nvcc`,
because CMake otherwise picks up the CUDA 11.5 wrapper in `/usr/bin`.

## Install / sync

```bash
uv sync
```

`uv sync` rebuilds the two vendored extensions if their sources changed
(~9 min for faiss, ~20 min for pytorch3d, both with 4 jobs). To rebuild one of
them explicitly:

```bash
CUDA_HOME=/usr/local/cuda-13.2 \
TORCH_CUDA_ARCH_LIST="12.0" \
MAX_JOBS=4 \
NVCC_THREADS=1 \
CMAKE_BUILD_PARALLEL_LEVEL=4 \
CUDACXX=$CUDA_HOME/bin/nvcc \
CMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
uv sync --reinstall-package faiss-gpu-cu12   # or pytorch3d
```

## What is in the environment

| package | version | notes |
|---|---|---|
| `torch` | 2.14.1+cu130 | from the `pytorch-cu130` index; arch list `sm_75 … sm_120` |
| `torchvision` | 0.29.1+cu130 | from the `pytorch-cu130` index |
| `torchaudio` | 2.11.0+cu130 | from the `pytorch-cu130` index |
| `torchcodec` | 0.17.0 | audio backend for torchaudio >= 2.9 |
| `triton` | 3.8.0 | comes with torch |
| `pytorch3d` | 0.7.9 | built from source, `third_party/pytorch3d` |
| `faiss-gpu-cu12` | 1.15.1 | built from source, `third_party/faiss` |
| `nvidia-*` | cu13 line | CUDA 13 runtime wheels, single `nvidia/cu13/lib` directory |

The CUDA 12 `nvidia-*-cu12` wheels are **not** used any more: the PyTorch cu130
build and the locally built faiss both link against the CUDA 13 runtime.

### Why pytorch3d and faiss are built from source

* **pytorch3d 0.7.9** does not compile against torch 2.14 (C++17 vs. the C++20
  that ATen now requires) and does not know the CUDA 13 visibility flags. Both
  fixes from upstream `main` were backported — see
  [`third_party/pytorch3d/README.md`](third_party/pytorch3d/README.md).
* **faiss**: the PyPI wheel `faiss-gpu-cu12` only ships `sm_70`/`sm_80` SASS and
  fails on the RTX 5090 with `CUDA error 209`. It is rebuilt for `sm_120` and
  needs a patched loader for the CUDA 13 wheel layout — see
  [`third_party/faiss/README.md`](third_party/faiss/README.md).

## Local models (no cloud API)

The `openai` dependency has been removed: every former OpenAI call now goes to
a single model that runs on this machine.  Both clients of the tiny
OpenAI-compatible client (`conceptgraph/utils/llm_client.py`) use the same
local route:

| step | model | server | client |
|---|---|---|---|
| text (caption consolidation, scene-graph bookkeeping) | `qwen3.8:27b-mtp4-ud_q4_K_XL-128k` | Ollama (`:20261`) | `get_llm_client()` |
| vision (annotated frame → relations / captions) | `qwen3.8:27b-mtp4-ud_q4_K_XL-128k` | Ollama (`:20261`) | `get_vlm_client()` |

### Qwen3.8 (Ollama) - the shared local model

```bash
ollama serve                       # OLLAMA_HOST=127.0.0.1:20261, OLLAMA_MODELS=/mnt/HDD1/ollama_models
ollama list                        # qwen3.8:27b-mtp4-ud_q4_K_XL-128k
```

The tag encodes the configuration: **UD** (Unsloth-Dynamic Q4_K_XL quantisation),
**MTP4** (`draft_num_predict = 4`, speculative decoding) and **128k**
(`num_ctx = 131072`).  The model is a native VLM with *thinking* levels
`false | low | medium | xhigh`; the pipeline runs it with `xhigh`, which the
client sends as `reasoning_effort: "xhigh"` to Ollama's OpenAI-compatible
endpoint (`/v1/chat/completions`).  The model serves both the text and the
vision requests. The staged launcher keeps it resident for the complete VLM
annotation pass, unloads it before mapping, then loads it again for final
caption consolidation.

### Configuration

| variable | default | meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://127.0.0.1:20261/v1` | Ollama endpoint (text model) |
| `LLM_MODEL` | `qwen3.8:27b-mtp4-ud_q4_K_XL-128k` | Ollama model tag |
| `LLM_REASONING_EFFORT` | `xhigh` | thinking level of the text model |
| `LLM_TIMEOUT` | `600` | request timeout in seconds |
| `VLM_BASE_URL` | `http://127.0.0.1:20261/v1` | Ollama endpoint (vision model) |
| `VLM_MODEL` | `qwen3.8:27b-mtp4-ud_q4_K_XL-128k` | Ollama model tag |
| `VLM_REASONING_EFFORT` | `xhigh` | thinking level of the vision model |
| `VLM_TIMEOUT` | `600` | request timeout in seconds |
| `VLM_MAX_TOKENS` | `4096` | output-token cap for vision requests |

The launcher only fills in these defaults when a variable is not already set,
so explicit environment overrides always take precedence.

See `env_vars.bash.template`.

## The local staged pipeline

The supported local command runs one GPU model phase at a time:

```bash
scripts/run_pipeline_local.sh --scene room0 --stride 50
```

The launcher runs YOLO detection, SAM-L masks, and CLIP crop embeddings as
separate scene-wide passes. It releases each model before loading the next.
Then it runs local Qwen VLM annotations, cached RGB-D mapping with GPU Faiss,
and caption consolidation. The VLM and text model unload between their stages.

Use `--no-consolidation` to skip the final text-model stage. Other options are
`--scene <id>`, `--stride <n>`, `--data-root <dir>`, `--dataset <name>` and
`--dataset-config <path>`. The launcher composes the dataset root after parsing.
Mapping starts after neural weights have been released, leaving VRAM available
to geometry and GPU Faiss. The launcher checks configured `/models` endpoints
and aborts if they do not answer. Results are written below
`<dataset_root>/<scene>/exps/r_mapping_stride<STRIDE>`.

The mapping process keeps exact GPU Faiss indexes for unchanged object point
clouds resident across overlap calls. A geometry hash catches in-place point
changes, while stable object IDs let an updated cloud replace its prior index.
The active-index payload limit is 8 GiB by default; override it with
`CONCEPTGRAPH_FAISS_GPU_CACHE_BYTES` (bytes). A map whose simultaneously needed
indexes exceed this limit fails before GPU upload. The limit excludes Faiss
index overhead, query batches and temporary tensors, so it is not a peak-VRAM
cap. Indexes for deleted objects are reclaimed by LRU eviction or when mapping
exits. Only changed or uncached map points are uploaded to build indexes;
query points still cross from existing Open3D/NumPy storage each invocation.

### LLaVA in the Hugging Face format

The LLaVA checkpoints of the local hub (`llava-hf/llava-v1.6-mistral-7b-hf`,
`llava-hf/llava-onevision-qwen2-7b-ov-hf`, ...) are stored in the
*transformers* format, the vendored `conceptgraph/llava/llava_model.py`
expects the *original* LLaVA release format.  `conceptgraph/llava/llava_hf_model.py`
provides a drop-in replacement (`LLaVAHFChat`) with the same API;
`build_scenegraph_cfslam.py` uses it through `load_llava_chat()`, which falls
back to the original loader if a checkpoint is not in the HF format.

## Verification

```bash
.venv/bin/python scripts/verify_env.py
```

The script exercises every installed package on the CPU **and** the GPU
(65 checks: matmuls, convolutions, autograd, autocast, a Triton kernel, the
pytorch3d CUDA ops and rasteriser, the faiss GPU indices, the audio I/O stack,
the CUDA driver bindings, ...) and prints a PASS/FAIL report. It exits with
status 1 if anything fails.
