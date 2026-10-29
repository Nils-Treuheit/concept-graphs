#!/usr/bin/env bash
# Serve the local vision-language model that replaces the former cloud VLM
# calls (scene-graph relations + object captions).
#
#   Qwen/Qwen3-VL-8B-Instruct   general purpose VLM
#   JosephZ/R1-SGG-7B           scene-graph specialised model (Qwen2-VL-7B)
#
# The pipeline talks to it through the OpenAI-compatible endpoint
# http://127.0.0.1:8000/v1 (see conceptgraph/utils/llm_client.py).
#
# NOTE: only one model fits into the 32 GB of the RTX 5090 at a time - stop
# the Ollama model first (`ollama stop qwen3.8:27b-mtp4-ud_q4_K_XL-128k`).
#
# Usage:  scripts/serve_qwen3_vl.sh [model] [port]
set -euo pipefail

MODEL="${1:-Qwen/Qwen3-VL-8B-Instruct}"
PORT="${2:-8000}"
MAX_LEN="${MAX_MODEL_LEN:-16384}"
# 0.80 leaves ~5 GB of the 32 GB card free so that the mapping stage can keep
# the CLIP model (and PyTorch3D) on the GPU while the VLM is serving.
GPU_UTIL="${GPU_MEMORY_UTILIZATION:-0.80}"

export HF_HOME="${HF_HOME:-/mnt/HDD1/unsloth_and_hugging_face_models/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

echo "Serving ${MODEL} on http://127.0.0.1:${PORT}/v1"
exec vllm serve "${MODEL}" \
    --served-model-name "${MODEL}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --max-model-len "${MAX_LEN}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --limit-mm-per-prompt '{"image": 1}'
