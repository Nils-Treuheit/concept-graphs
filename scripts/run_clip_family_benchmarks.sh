#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-${repo_root}/.venv/bin/python}"
executor="${CODEX_EXECUTOR:-/home/nils/Project_Code/codex_opencode_pipeline/src/executor.py}"
batch_size="${BATCH_SIZE:-4}"

if [[ ! -x "${python_bin}" ]]; then
  echo "Python executable not found: ${python_bin}" >&2
  exit 2
fi
if [[ ! -f "${executor}" ]]; then
  echo "Model-release executor not found: ${executor}" >&2
  echo "Set CODEX_EXECUTOR to the workflow checkout's src/executor.py." >&2
  exit 2
fi

# Fail before starting the longer COCO run if the large Replica archive or its
# semantic instance assets have not finished extracting.
cd "${repo_root}"
"${python_bin}" -m conceptgraph.scripts.replica_clip_benchmark --check-data-only
"${python_bin}" -m conceptgraph.scripts.clip_component --check-data-only

"${python_bin}" "${executor}" release-local
"${python_bin}" -m conceptgraph.scripts.clip_component --batch-size "${batch_size}"

# Release between jobs so the next benchmark starts without stale model state.
"${python_bin}" "${executor}" release-local
"${python_bin}" -m conceptgraph.scripts.replica_clip_benchmark --batch-size "${batch_size}"
