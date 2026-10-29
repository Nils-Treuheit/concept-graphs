#!/usr/bin/env bash
# Serve one selected GGUF model with all model layers and its image projector
# offloaded to CUDA. Stop it with Ctrl-C before launching another model.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_DIR="${CG_EDGE_MODEL_DIR:-/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/edge-compatible-gguf}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$(command -v llama-server || true)}"
WORKFLOW_ROOT="${CODEX_OPENCODE_PIPELINE:-/home/nils/Project_Code/codex_opencode_pipeline}"
PORT="${EDGE_COMPATIBLE_PORT:-20262}"
CTX_SIZE="${EDGE_COMPATIBLE_CONTEXT_SIZE:-16384}"

usage() {
    printf 'Usage: %s {qwen35-9b-q4|qwen35-9b-q8|gemma4-12b|gemma4-26b-a4b}\n' "$0"
}

if [[ $# -ne 1 ]]; then usage >&2; exit 2; fi
MODEL_ID="$1"
case "$MODEL_ID" in
    qwen35-9b-q4) MODEL_FILE="Qwen3.5-9B-UD-Q4_K_XL.gguf" ;;
    qwen35-9b-q8) MODEL_FILE="Qwen3.5-9B-UD-Q8_K_XL.gguf" ;;
    gemma4-12b) MODEL_FILE="gemma-4-12B-it-qat-UD-Q4_K_XL.gguf" ;;
    gemma4-26b-a4b) MODEL_FILE="gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf" ;;
    *) usage >&2; exit 2 ;;
esac

MODEL_PATH="${MODEL_DIR}/${MODEL_ID}/${MODEL_FILE}"
MMPROJ_PATH="${MODEL_DIR}/${MODEL_ID}/mmproj-F16.gguf"
for path in "$MODEL_PATH" "$MMPROJ_PATH"; do
    if [[ ! -s "$path" ]]; then
        echo "Missing model file: $path" >&2
        echo "Download it with scripts/edge-compatible/download_models.py first." >&2
        exit 2
    fi
done
if [[ -z "$LLAMA_SERVER_BIN" || ! -x "$LLAMA_SERVER_BIN" ]]; then
    echo "CUDA-enabled llama-server was not found. Set LLAMA_SERVER_BIN to its executable." >&2
    exit 2
fi
if [[ ! -f "$WORKFLOW_ROOT/src/executor.py" ]]; then
    echo "Agentic workflow executor not found at $WORKFLOW_ROOT/src/executor.py" >&2
    echo "Set CODEX_OPENCODE_PIPELINE to the workflow checkout." >&2
    exit 2
fi

# The project workflow requires unloading managed local models before probing
# CUDA or starting any local model job.
(cd "$WORKFLOW_ROOT" && python3 src/executor.py release-local)

DEVICE_LIST="$($LLAMA_SERVER_BIN --list-devices 2>&1)" || {
    echo "$DEVICE_LIST" >&2
    exit 1
}
printf '%s\n' "$DEVICE_LIST"
if ! grep -Eiq 'cuda|nvidia' <<<"$DEVICE_LIST"; then
    echo "This llama-server build exposes no CUDA device; refusing CPU inference." >&2
    exit 2
fi

echo "Starting $MODEL_ID on CUDA with a ${CTX_SIZE}-token context at port ${PORT}."
echo "All model layers and the multimodal projector are requested on GPU."
exec "$LLAMA_SERVER_BIN" \
    --model "$MODEL_PATH" \
    --mmproj "$MMPROJ_PATH" \
    --alias edge-compatible \
    --host 127.0.0.1 \
    --port "$PORT" \
    --n-gpu-layers all \
    --mmproj-offload \
    --fit off \
    --flash-attn on \
    --ctx-size "$CTX_SIZE" \
    --parallel 1
