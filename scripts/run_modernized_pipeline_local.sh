#!/usr/bin/env bash
# Run the isolated YOLOE-26 -> VLM annotation -> cached RGB-D mapping pipeline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-/mnt/HDD1/Project_Data/public_datasets}"
DATASET_NAME="${DATASET_NAME:-Replica}"
DATASET_CONFIG="${DATASET_CONFIG:-${REPO_ROOT}/conceptgraph/dataset/dataconfigs/replica/replica.yaml}"
SCENE="${SCENE:-room0}"
STRIDE="${STRIDE:-50}"
RUN_CONSOLIDATION=1
RUN_DETECTIONS=1

CONSOLIDATION_MODEL="${CONSOLIDATION_MODEL:-gemma4-12b}"
LLM_BASE_URL="${LLM_BASE_URL:-}"
LLM_MODEL="${LLM_MODEL:-}"
LLM_REASONING_EFFORT="${LLM_REASONING_EFFORT-}"
LLM_TIMEOUT="${LLM_TIMEOUT:-600}"
VLM_BASE_URL="${VLM_BASE_URL:-http://127.0.0.1:20261/v1}"
VLM_MODEL="${VLM_MODEL:-qwen3.8:27b-mtp4-ud_q4_K_XL-128k}"
VLM_REASONING_EFFORT="${VLM_REASONING_EFFORT-xhigh}"
VLM_TIMEOUT="${VLM_TIMEOUT:-600}"
VLM_MAX_TOKENS="${VLM_MAX_TOKENS:-4096}"
LLM_SERVER_PID=""

stop_llm_server() {
    if [[ -n "${LLM_SERVER_PID}" ]] && kill -0 "${LLM_SERVER_PID}" 2>/dev/null; then
        kill -TERM "${LLM_SERVER_PID}" 2>/dev/null || true
        wait "${LLM_SERVER_PID}" 2>/dev/null || true
        LLM_SERVER_PID=""
    fi
}
trap stop_llm_server EXIT

YOLOE_CHECKPOINT="${YOLOE_CHECKPOINT:-}"
INPUT_MODE="${INPUT_MODE:-masked}"
MASK_ENCODER="${MASK_ENCODER:-siglip2}"
MASK_MODEL_ID="${MASK_MODEL_ID:-google/siglip2-so400m-patch14-384}"
MASK_CHECKPOINT="${MASK_CHECKPOINT:-}"
BOX_ENCODER="${BOX_ENCODER:-fg_clip2}"
BOX_MODEL_ID="${BOX_MODEL_ID:-qihoo360/fg-clip2-so400m}"
BOX_CHECKPOINT="${BOX_CHECKPOINT:-}"

export HF_HOME="${HF_HOME:-/mnt/HDD1/unsloth_and_hugging_face_models/huggingface}"
export LLM_BASE_URL LLM_MODEL LLM_REASONING_EFFORT LLM_TIMEOUT
export VLM_BASE_URL VLM_MODEL VLM_REASONING_EFFORT VLM_TIMEOUT VLM_MAX_TOKENS
export VLM_KEEP_ALIVE="${VLM_KEEP_ALIVE:--1}"
export LLM_KEEP_ALIVE="${LLM_KEEP_ALIVE:--1}"
export VLM_UNLOAD_AFTER_FRAME=0
PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    echo "Error: the modernized pipeline requires this repository's Python environment at ${PYTHON}. Run uv sync first." >&2
    exit 1
fi

usage() {
    cat <<'EOF'
Usage: scripts/run_modernized_pipeline_local.sh [options]
  --scene NAME --stride N --data-root DIR --dataset NAME
  --dataset-config PATH --no-consolidation
  --consolidation-model gemma4-12b|qwen3.8
  --resume-from-caches
  --input-mode masked|region-proposal
  --yoloe-checkpoint PATH
  --mask-encoder siglip2|clip --mask-model-id ID --mask-checkpoint PATH
  --box-encoder fg_clip2|siglip2|clip --box-model-id ID --box-checkpoint PATH
  --vlm-model NAME

Use one override per run when preparing a component comparison. Each run must
use a distinct DETECTIONS_EXP_SUFFIX and MAP_EXP_SUFFIX.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene) SCENE="$2"; shift 2 ;;
        --stride) STRIDE="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --dataset) DATASET_NAME="$2"; shift 2 ;;
        --dataset-config) DATASET_CONFIG="$2"; shift 2 ;;
        --consolidation-model) CONSOLIDATION_MODEL="$2"; shift 2 ;;
        --input-mode) INPUT_MODE="$2"; shift 2 ;;
        --no-consolidation) RUN_CONSOLIDATION=0; shift ;;
        --resume-from-caches) RUN_DETECTIONS=0; shift ;;
        --yoloe-checkpoint) YOLOE_CHECKPOINT="$2"; shift 2 ;;
        --mask-encoder) MASK_ENCODER="$2"; shift 2 ;;
        --mask-model-id) MASK_MODEL_ID="$2"; shift 2 ;;
        --mask-checkpoint) MASK_CHECKPOINT="$2"; shift 2 ;;
        --box-encoder) BOX_ENCODER="$2"; shift 2 ;;
        --box-model-id) BOX_MODEL_ID="$2"; shift 2 ;;
        --box-checkpoint) BOX_CHECKPOINT="$2"; shift 2 ;;
        --vlm-model) VLM_MODEL="$2"; export VLM_MODEL; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

case "${CONSOLIDATION_MODEL}" in
    gemma4-12b)
        LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:20262/v1}"
        LLM_MODEL="${LLM_MODEL:-edge-compatible-gemma4-12b}"
        LLM_REASONING_EFFORT="${LLM_REASONING_EFFORT:-}"
        LLM_JSON_MODE=0
        LLM_EXTERNAL_SERVER=1
        ;;
    qwen3.8)
        LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:20261/v1}"
        LLM_MODEL="${LLM_MODEL:-qwen3.8:27b-mtp4-ud_q4_K_XL-128k}"
        LLM_REASONING_EFFORT="${LLM_REASONING_EFFORT:-xhigh}"
        LLM_JSON_MODE=1
        LLM_EXTERNAL_SERVER=0
        ;;
    *)
        echo "Error: --consolidation-model must be gemma4-12b or qwen3.8." >&2
        exit 2
        ;;
esac
export LLM_BASE_URL LLM_MODEL LLM_REASONING_EFFORT LLM_JSON_MODE LLM_EXTERNAL_SERVER

case "${INPUT_MODE}" in
    masked)
        DEFAULT_YOLOE_CHECKPOINT="yoloe-26l-seg.pt"
        ;;
    region-proposal)
        DEFAULT_YOLOE_CHECKPOINT="yoloe-26l.pt"
        ;;
    *)
        echo "Error: --input-mode must be masked or region-proposal." >&2
        exit 2
        ;;
esac
YOLOE_CHECKPOINT="${YOLOE_CHECKPOINT:-${DEFAULT_YOLOE_CHECKPOINT}}"
HYDRA_INPUT_MODE="${INPUT_MODE//-/_}"

DATASET_ROOT="${DATA_ROOT}/${DATASET_NAME}"
DET_SUFFIX="${DETECTIONS_EXP_SUFFIX:-s_modernized_${HYDRA_INPUT_MODE}_yoloe26l_stride${STRIDE}}"
MAP_SUFFIX="${MAP_EXP_SUFFIX:-r_modernized_${HYDRA_INPUT_MODE}_yoloe26l_stride${STRIDE}}"
log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

check_endpoint() {
    local url="$1"
    log "Checking model endpoint ${url}/models"
    if ! curl --fail -s --max-time 10 "${url}/models" > /dev/null; then
        echo "Error: model endpoint ${url} did not answer on /models; not starting the pipeline." >&2
        exit 1
    fi
}

if [[ "${CONSOLIDATION_MODEL}" == "qwen3.8" ]]; then
    check_endpoint "${LLM_BASE_URL}"
fi
check_endpoint "${VLM_BASE_URL}"

if [[ "${RUN_DETECTIONS}" == "1" ]]; then
    log "YOLOE-26 detection (scene: ${SCENE}, stride: ${STRIDE}, suffix: ${DET_SUFFIX})"
    "${PYTHON}" -m conceptgraph.scripts.modernized_detections \
        dataset_root="${DATASET_ROOT}" \
        dataset_config="${DATASET_CONFIG}" \
        scene_id="${SCENE}" \
        stride="${STRIDE}" \
        exp_suffix="${DET_SUFFIX}" \
        features.input_mode="${HYDRA_INPUT_MODE}" \
        model.yoloe_checkpoint="${YOLOE_CHECKPOINT}" \
        features.mask_encoder="${MASK_ENCODER}" \
        features.mask_model_id="${MASK_MODEL_ID}" \
        features.mask_checkpoint="${MASK_CHECKPOINT}" \
        features.box_encoder="${BOX_ENCODER}" \
        features.box_model_id="${BOX_MODEL_ID}" \
        features.box_checkpoint="${BOX_CHECKPOINT}" \
        device=cuda
else
    DET_CACHE_DIR="${DATASET_ROOT}/${SCENE}/exps/${DET_SUFFIX}/detections"
    if [[ ! -d "${DET_CACHE_DIR}" ]] || ! compgen -G "${DET_CACHE_DIR}/*.pkl.gz" > /dev/null; then
        echo "Error: no detection caches found under ${DET_CACHE_DIR}; cannot resume from caches." >&2
        exit 1
    fi
    EXPECTED_CACHE_COUNT="$("${PYTHON}" - "${DATASET_CONFIG}" "${DATASET_ROOT}" "${SCENE}" "${STRIDE}" <<'PY'
import sys
import contextlib
import io
import torch
from conceptgraph.dataset.datasets_common import get_dataset

dataset_config, dataset_root, scene, stride = sys.argv[1:]
with contextlib.redirect_stdout(io.StringIO()):
    dataset = get_dataset(
        dataconfig=dataset_config,
        start=0,
        end=-1,
        stride=int(stride),
        basedir=dataset_root,
        sequence=scene,
        desired_height=480,
        desired_width=640,
        device="cpu",
        dtype=torch.float,
    )
print(len(dataset.color_paths))
PY
    )"
    ACTUAL_CACHE_COUNT="$(find "${DET_CACHE_DIR}" -maxdepth 1 -type f -name '*.pkl.gz' | wc -l | tr -d '[:space:]')"
    if [[ "${ACTUAL_CACHE_COUNT}" != "${EXPECTED_CACHE_COUNT}" ]]; then
        echo "Error: cache resume requires one detection cache per sampled frame; found ${ACTUAL_CACHE_COUNT}, expected ${EXPECTED_CACHE_COUNT} in ${DET_CACHE_DIR}." >&2
        exit 1
    fi
    log "Resuming from existing detection caches (${DET_SUFFIX})"
fi

log "Annotating cached detections with ${VLM_MODEL}"
"${PYTHON}" -m conceptgraph.scripts.vlm_annotations \
    dataset_root="${DATASET_ROOT}" \
    dataset_config="${DATASET_CONFIG}" \
    scene_id="${SCENE}" \
    stride="${STRIDE}" \
    detections_exp_suffix="${DET_SUFFIX}"

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
    use_rerun=false \
    consolidate_captions=false

EXP_OUT_PATH="${DATASET_ROOT}/${SCENE}/exps/${MAP_SUFFIX}"
if [[ "${RUN_CONSOLIDATION}" == "1" ]]; then
    if [[ "${CONSOLIDATION_MODEL}" == "gemma4-12b" ]]; then
        log "Starting Gemma 4 12B Q4 for final caption consolidation"
        GEMMA_SERVER_LOG="${EXP_OUT_PATH}/gemma4-server.log"
        mkdir -p "${EXP_OUT_PATH}"
        CG_EDGE_MODEL_ALIAS="${LLM_MODEL}" \
            "${SCRIPT_DIR}/edge-compatible/serve_model.sh" gemma4-12b \
            >"${GEMMA_SERVER_LOG}" 2>&1 &
        LLM_SERVER_PID=$!
        server_ready=0
        for _ in $(seq 1 120); do
            if ! kill -0 "${LLM_SERVER_PID}" 2>/dev/null; then
                echo "Error: Gemma 4 server exited during startup. See ${GEMMA_SERVER_LOG}." >&2
                cat "${GEMMA_SERVER_LOG}" >&2
                exit 1
            fi
            if curl --fail -s --max-time 2 "${LLM_BASE_URL}/models" \
                | "${PYTHON}" -c 'import json,sys; expected=sys.argv[1]; data=json.load(sys.stdin).get("data", []); raise SystemExit(0 if any(row.get("id") == expected for row in data) else 1)' \
                    "${LLM_MODEL}"; then
                server_ready=1
                break
            fi
            sleep 1
        done
        if [[ "${server_ready}" != "1" ]]; then
            echo "Error: Gemma 4 server did not expose ${LLM_MODEL} within 120 seconds. See ${GEMMA_SERVER_LOG}." >&2
            exit 1
        fi
    fi
    check_endpoint "${LLM_BASE_URL}"
    log "Consolidating object captions in ${EXP_OUT_PATH}"
    "${PYTHON}" -m conceptgraph.scripts.consolidate_captions_local \
        --exp_out_path "${EXP_OUT_PATH}" \
        --exp_suffix "${MAP_SUFFIX}"
fi
log "Modernized pipeline finished. Detection output: ${DATASET_ROOT}/${SCENE}/exps/${DET_SUFFIX}; map: ${EXP_OUT_PATH}"
