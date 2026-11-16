# External build dependencies

The full FAISS and PyTorch3D source trees are external Git checkouts, ignored by
the ConceptGraphs repository. The project tracks only upstream references, a
small CUDA compatibility patch, and the FAISS local build configuration.

On a fresh checkout, prepare the pinned source trees before running `uv sync`:

```bash
bash scripts/bootstrap_third_party.sh
uv sync
```

The bootstrap checks out upstream tags at fixed commit hashes. It refuses to
replace an existing non-Git directory or overwrite a different local build
configuration. Existing local files are left untouched on failure.

## FAISS

- Upstream: [facebookresearch/faiss](https://github.com/facebookresearch/faiss)
- Reference: tag `v1.15.1`, commit `75c755c5a1fc854756aaf3bd1131213aeb21d8d1`
- Local adaptation: add a CUDA 13 build marker and preload shared libraries
  from the CUDA 13 wheel layout for non-cuVS builds. The local build config
  targets the RTX 5090 (`sm_120`) and CUDA 13 runtime packages.
- Tracked adaptation: [`patches/faiss-cuda13.patch`](patches/faiss-cuda13.patch)
  and [`patches/faiss-pyproject.toml`](patches/faiss-pyproject.toml).

## PyTorch3D

- Upstream: [facebookresearch/pytorch3d](https://github.com/facebookresearch/pytorch3d)
- Reference: tag `v0.7.9`, commit `33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba`
- Local adaptation: compile the extension as C++20 for the project PyTorch
  build and add CUDA 13 visibility flags for the Pulsar kernels.
- Tracked adaptation: [`patches/pytorch3d-cuda13.patch`](patches/pytorch3d-cuda13.patch).

These adaptations are small enough to maintain as patches against upstream.
If they grow beyond targeted compatibility changes, move them to a maintained
ConceptGraphs fork and pin that fork here.
