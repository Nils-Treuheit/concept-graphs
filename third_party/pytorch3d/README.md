# Vendored PyTorch3D

Source: https://github.com/facebookresearch/pytorch3d @ tag `v0.7.9` (commit `33824be3`)

Only `setup.py`, `pytorch3d/` and `projects/` are vendored (the build inputs); `docs/`,
`tests/`, `dev/` and `website/` are omitted.

## Local modifications to `setup.py`

Upstream `v0.7.9` cannot be built against the PyTorch version used in this project
(`torch 2.14.1+cu130`) or with a CUDA 13 toolkit, so two fixes from upstream `main`
were backported:

1. **C++20** — `torch/ATen/ATen.h` errors out with
   `#error C++20 or later compatible compiler is required to use ATen` when the
   extension is compiled as C++17, which is what upstream `v0.7.9` requests.
   `-std=c++17` was changed to `-std=c++20` for both the C++ and the NVCC
   compiler arguments.

2. **CUDA 13 flags** — starting with CUDA 13, `__global__` function visibility
   changed, which breaks the `pulsar` kernels. The flags added upstream in `main`
   are used:
   `--device-entity-has-hidden-visibility=false` and `-static-global-template-stub=false`.

## Building

The extension is compiled against the CUDA toolkit found in `CUDA_HOME` and the
architecture list in `TORCH_CUDA_ARCH_LIST` (this machine has an RTX 5090, i.e.
`sm_120`):

```bash
CUDA_HOME=/usr/local/cuda-13.2 \
PATH=$CUDA_HOME/bin:$PATH \
FORCE_CUDA=1 \
CUB_HOME=$CUDA_HOME/include \
TORCH_CUDA_ARCH_LIST="12.0" \
MAX_JOBS=4 \
NVCC_THREADS=1 \
CMAKE_BUILD_PARALLEL_LEVEL=4 \
uv sync --reinstall-package pytorch3d
```

The flags above are the ones defined in `~/.bashrc` (`MAX_JOBS=4` because every
nvcc job wants 6-8 GB of RAM). The build takes ~20 minutes with 4 jobs.

`pyproject.toml` redirects the build requirement `torch==2.14.1` to the
`pytorch-cu130` index, so the extension is compiled against the same torch
build that is installed at runtime.
