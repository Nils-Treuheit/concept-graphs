#!/usr/bin/env bash
# Build a private CUDA-enabled llama.cpp server without installing system-wide.
set -euo pipefail

SOURCE_DIR="${LLAMA_CPP_SOURCE_DIR:-/home/nils/.local/src/llama.cpp}"
WORKFLOW_ROOT="${CODEX_OPENCODE_PIPELINE:-/home/nils/Project_Code/codex_opencode_pipeline}"
JOBS="${LLAMA_BUILD_JOBS:-4}"

for command_name in git cmake nvcc; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Required build tool is missing: $command_name" >&2
        exit 2
    fi
done
if [[ ! -f "$WORKFLOW_ROOT/src/executor.py" ]]; then
    echo "Agentic workflow executor not found at $WORKFLOW_ROOT/src/executor.py" >&2
    echo "Set CODEX_OPENCODE_PIPELINE to the workflow checkout." >&2
    exit 2
fi

mkdir -p "$(dirname "$SOURCE_DIR")"
if [[ -d "$SOURCE_DIR/.git" ]]; then
    echo "Using existing llama.cpp checkout: $SOURCE_DIR"
elif [[ ! -e "$SOURCE_DIR" ]]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$SOURCE_DIR"
elif [[ -n "$(find "$SOURCE_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to use a non-empty directory without Git metadata: $SOURCE_DIR" >&2
    exit 2
else
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$SOURCE_DIR"
fi

if [[ -n "${LLAMA_CPP_REF:-}" ]]; then
    git -C "$SOURCE_DIR" fetch --depth 1 origin "$LLAMA_CPP_REF"
    git -C "$SOURCE_DIR" checkout --detach FETCH_HEAD
fi

cmake -S "$SOURCE_DIR" -B "$SOURCE_DIR/build" \
    -DGGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "$SOURCE_DIR/build" --config Release \
    --parallel "$JOBS" --target llama-server

SERVER_BIN="$SOURCE_DIR/build/bin/llama-server"
if [[ ! -x "$SERVER_BIN" ]]; then
    echo "Build completed without an executable at $SERVER_BIN" >&2
    exit 1
fi

(cd "$WORKFLOW_ROOT" && python3 src/executor.py release-local)
DEVICE_LIST="$("$SERVER_BIN" --list-devices 2>&1)" || {
    echo "$DEVICE_LIST" >&2
    exit 1
}
printf '%s\n' "$DEVICE_LIST"
if ! grep -Eiq 'cuda|nvidia' <<<"$DEVICE_LIST"; then
    echo "Build does not expose CUDA; refusing to recommend CPU inference." >&2
    exit 2
fi
printf '\nCUDA llama-server is ready.\n'
printf 'LLAMA_SERVER_BIN=%q scripts/edge-compatible/serve_model.sh qwen35-9b-q4\n' "$SERVER_BIN"
printf 'llama.cpp revision: %s\n' "$(git -C "$SOURCE_DIR" rev-parse HEAD)"
