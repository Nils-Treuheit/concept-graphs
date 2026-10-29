#!/usr/bin/env bash
# Source this profile to route both former GPT call classes through the
# separately served edge-compatible multimodal GGUF model.
export LLM_BASE_URL="${EDGE_COMPATIBLE_BASE_URL:-http://127.0.0.1:20262/v1}"
export LLM_MODEL="${EDGE_COMPATIBLE_MODEL_ALIAS:-edge-compatible}"
export LLM_API_KEY="EMPTY"
export LLM_TIMEOUT="${EDGE_COMPATIBLE_TIMEOUT:-600}"
export LLM_REASONING_EFFORT=""
export LLM_JSON_MODE=1

export VLM_BASE_URL="$LLM_BASE_URL"
export VLM_MODEL="$LLM_MODEL"
export VLM_API_KEY=EMPTY
export VLM_TIMEOUT="$LLM_TIMEOUT"
export VLM_REASONING_EFFORT=""
export VLM_JSON_MODE=0
export VLM_MAX_TOKENS="${EDGE_COMPATIBLE_VLM_MAX_TOKENS:-4096}"
# llama-server does not use the Ollama unload route.
export VLM_UNLOAD_AFTER_FRAME=0
