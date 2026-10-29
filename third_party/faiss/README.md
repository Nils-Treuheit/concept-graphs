# Vendored Faiss

Source: https://github.com/facebookresearch/faiss @ tag `v1.15.1` (MIT licence)

Only the library sources are vendored (`CMakeLists.txt`, `cmake/`, `contrib/`, `faiss/`);
`tests/`, `benchs/`, `demos/`, `c_api/` and `misc/` are omitted.

## Why it is built from source

The PyPI wheel `faiss-gpu-cu12` (1.14.1.post1) only contains SASS for `sm_70`
and `sm_80`; on an RTX 5090 (`sm_120`) every GPU query fails with
`CUDA error 209` ("no kernel image is available for execution on the device").
The wheel is therefore rebuilt here with `CMAKE_CUDA_ARCHITECTURES=120`.

`faiss/python/__init__.py` and `faiss/python/CMakeLists.txt` carry two local
patches (see below) that make the CUDA 13 build loadable from the CUDA 13
`nvidia-*` wheels that PyTorch already pulls in.

## Local modifications

1. **`faiss/python/CMakeLists.txt`** — writes an extra marker file
   `_cuda13_build.py` next to the existing `_gpu_build.py` when the extension
   is compiled with CUDA >= 13. Upstream only ships such a marker with the
   cuVS wheels.

2. **`faiss/python/__init__.py`** — the loader now distinguishes three cases:

   | marker files                | wheels loaded from              | cuVS stack |
   |-----------------------------|---------------------------------|------------|
   | `_cuvs_build.py`            | `nvidia/cu13/lib/`               | required   |
   | `_cuda13_build.py`          | `nvidia/cu13/lib/`               | not needed |
   | neither (upstream behaviour) | `nvidia/<component>/lib/` (cu12)| not needed |

   The CUDA 13 `nvidia-*` wheels ship every shared library in a single PEP 420
   namespace directory (`nvidia/cu13/lib/`), so the loader has to preload
   `libcudart.so.13`, `libnvJitLink.so.13`, `libcublas.so.13`,
   `libcublasLt.so.13` and `libcurand.so.10` from there. Without the patch the
   plain CUDA 12 code path is taken and the `libcudart.so.13` dependency of
   `libfaiss.so` cannot be resolved.

## Build configuration (`pyproject.toml`)

| setting                    | value                         | why                                            |
|----------------------------|-------------------------------|------------------------------------------------|
| `CMAKE_CUDA_ARCHITECTURES` | `120`                         | RTX 5090 (`sm_120`), see `TORCH_CUDA_ARCH_LIST` in `~/.bashrc` |
| `CMAKE_CUDA_COMPILER`      | `/usr/local/cuda-13.2/bin/nvcc` | `/usr/bin/nvcc` is Ubuntu's CUDA 11.5 wrapper  |
| `FAISS_OPT_LEVEL`          | `avx2`                        | host is an AMD Zen 4 (AVX2, no AVX-512)        |
| `FAISS_USE_LTO`            | `OFF`                         | keeps build time and memory usage down         |
| `build.tool-args`          | `["-j4"]`                     | `MAX_JOBS=4` from `~/.bashrc` (RAM limit)      |
| `CMAKE_CUDA_FLAGS_RELEASE` | `--threads 1`                 | `NVCC_THREADS=1` from `~/.bashrc`              |

## Building

```bash
CUDA_HOME=/usr/local/cuda-13.2 \
PATH=$CUDA_HOME/bin:$PATH \
TORCH_CUDA_ARCH_LIST="12.0" \
MAX_JOBS=4 \
NVCC_THREADS=1 \
CMAKE_BUILD_PARALLEL_LEVEL=4 \
CUDACXX=$CUDA_HOME/bin/nvcc \
CMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
uv sync --reinstall-package faiss-gpu-cu12
```

The build takes ~9 minutes with 4 jobs and produces ~56 MB of wheel.

## Result

`faiss/libfaiss.so` contains 182 `sm_120` cubins (plus the matching
`compute_120` PTX for forward compatibility), and
`GpuIndexFlatL2`, `GpuIndexFlatIP` and `GpuIndexIVFFlat` all run on the GPU.
