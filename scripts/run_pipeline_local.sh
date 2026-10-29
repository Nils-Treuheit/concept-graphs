#!/usr/bin/env bash
# Run ConceptGraphs in GPU ownership stages: YOLO, SAM, CLIP, VLM annotation,
# then cached RGB-D mapping/FAISS, and finally optional text consolidation.
# The local model service is checked but never started by this script.
#
# Usage:
#   scripts/run_pipeline_local.sh [--scene room0] [--stride 50] [--no-consolidation]
#   scripts/run_pipeline_local.sh [--data-root DIR] [--dataset NAME]
#   scripts/run_pipeline_local.sh [--dataset-config PATH]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# configuration (defaults; explicit environment variables take precedence)
# ---------------------------------------------------------------------------
DATA_ROOT="${DATA_ROOT:-/mnt/HDD1/Project_Data/public_datasets}"
DATASET_NAME="${DATASET_NAME:-Replica}"
DATASET_CONFIG="${DATASET_CONFIG:-${REPO_ROOT}/conceptgraph/dataset/dataconfigs/replica/replica.yaml}"
SCENE="${SCENE:-room0}"
STRIDE="${STRIDE:-50}"
RUN_CONSOLIDATION=1

# Shared local Ollama Qwen3.8 route for both the LLM and the VLM clients.
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:20261/v1}"
LLM_MODEL="${LLM_MODEL:-qwen3.8:27b-mtp4-ud_q4_K_XL-128k}"
LLM_REASONING_EFFORT="${LLM_REASONING_EFFORT-xhigh}"
LLM_TIMEOUT="${LLM_TIMEOUT:-600}"
VLM_BASE_URL="${VLM_BASE_URL:-http://127.0.0.1:20261/v1}"
VLM_MODEL="${VLM_MODEL:-qwen3.8:27b-mtp4-ud_q4_K_XL-128k}"
VLM_REASONING_EFFORT="${VLM_REASONING_EFFORT-xhigh}"
VLM_TIMEOUT="${VLM_TIMEOUT:-600}"
VLM_MAX_TOKENS="${VLM_MAX_TOKENS:-4096}"

export HF_HOME="${HF_HOME:-/mnt/HDD1/unsloth_and_hugging_face_models/huggingface}"
export LLM_BASE_URL LLM_MODEL LLM_REASONING_EFFORT LLM_TIMEOUT
export VLM_BASE_URL VLM_MODEL VLM_REASONING_EFFORT VLM_TIMEOUT VLM_MAX_TOKENS
# Keep the VLM resident during its complete annotation pass. The annotation
# process unloads it once when that stage exits, before mapping starts.
export VLM_UNLOAD_AFTER_FRAME=0
export PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"

usage() {
    sed -n '2,16p' "$0"
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene) SCENE="$2"; shift 2 ;;
        --stride) STRIDE="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --dataset) DATASET_NAME="$2"; shift 2 ;;
        --dataset-config) DATASET_CONFIG="$2"; shift 2 ;;
        --no-consolidation) RUN_CONSOLIDATION=0; shift ;;
        -h|--help) usage 0 ;;
        *) echo "Unknown option: $1"; usage 1 ;;
    esac
done

# DATASET_ROOT is calculated after parsing so that --data-root/--dataset
# overrides take effect.
DATASET_ROOT="${DATA_ROOT}/${DATASET_NAME}"
DET_SUFFIX="${DETECTIONS_EXP_SUFFIX:-s_detections_stride${STRIDE}}"
MAP_SUFFIX="${MAP_EXP_SUFFIX:-r_mapping_stride${STRIDE}}"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------
# endpoint readiness (the launcher only checks; it never starts or stops
# model servers)
# ---------------------------------------------------------------------------
check_endpoint() {
    local url="$1"
    log "Checking model endpoint ${url}/models"
    if ! curl --fail -s --max-time 10 "${url}/models" > /dev/null; then
        echo "Error: model endpoint ${url} did not answer on /models; not starting the pipeline." >&2
        exit 1
    fi
}

check_endpoint "${LLM_BASE_URL}"
if [[ "${VLM_BASE_URL}" != "${LLM_BASE_URL}" ]]; then
    check_endpoint "${VLM_BASE_URL}"
fi

# ---------------------------------------------------------------------------
# Stage 1: sequential YOLO -> SAM-L -> CLIP passes. Each model is released
# before the next one loads; the resulting per-frame detections are cached.
# ---------------------------------------------------------------------------
log "Running sequential GPU detection stages (scene: ${SCENE}, stride: ${STRIDE})"
"${PYTHON}" -m conceptgraph.scripts.streamlined_detections \
    dataset_root="${DATASET_ROOT}" \
    dataset_config="${DATASET_CONFIG}" \
    scene_id="${SCENE}" \
    stride="${STRIDE}" \
    exp_suffix="${DET_SUFFIX}" \
    device=cuda

# Stage 2: Qwen VLM annotations run after all detector models have exited.
log "Annotating cached detections with the local VLM"
"${PYTHON}" -m conceptgraph.scripts.vlm_annotations \
    dataset_root="${DATASET_ROOT}" \
    dataset_config="${DATASET_CONFIG}" \
    scene_id="${SCENE}" \
    stride="${STRIDE}" \
    detections_exp_suffix="${DET_SUFFIX}"

# Stage 3: map cached detections. No neural weights are constructed here;
# CUDA remains available for geometry and GPU Faiss overlap search.
log "Mapping cached detections and RGB-D data"
"${PYTHON}" -m conceptgraph.slam.rerun_realtime_mapping \
    dataset_root="${DATASET_ROOT}" \
    dataset_config="${DATASET_CONFIG}" \
    scene_id="${SCENE}" \
    stride="${STRIDE}" \
    detections_exp_suffix="${DET_SUFFIX}" \
    exp_suffix="${MAP_SUFFIX}" \
    force_detection=false \
    make_edges=false \
    consolidate_captions=false

EXP_OUT_PATH="${DATASET_ROOT}/${SCENE}/exps/${MAP_SUFFIX}"
if [[ "${RUN_CONSOLIDATION}" == "1" ]]; then
    log "Consolidating object captions in ${EXP_OUT_PATH}"
    "${PYTHON}" -m conceptgraph.scripts.consolidate_captions_local \
        --exp_out_path "${EXP_OUT_PATH}" \
        --exp_suffix "${MAP_SUFFIX}"
    log "Pipeline and caption consolidation finished. Results in ${EXP_OUT_PATH}"
else
    log "Mapping finished; caption consolidation skipped (--no-consolidation). Results in ${EXP_OUT_PATH}"
fi
