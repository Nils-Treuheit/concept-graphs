#!/usr/bin/env bash
# Run paired full-pipeline arms while changing exactly one component setting.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
COMPONENT=""
BASELINE=""
CANDIDATE=""
SCENE="${SCENE:-room0}"
STRIDE="${STRIDE:-50}"
INPUT_MODE=""
DATA_ROOT="${DATA_ROOT:-/mnt/HDD1/Project_Data/public_datasets}"
DATASET_NAME="${DATASET_NAME:-Replica}"
DATASET_CONFIG="${DATASET_CONFIG:-${REPO_ROOT}/conceptgraph/dataset/dataconfigs/replica/replica.yaml}"

usage() {
    cat <<'EOF'
Usage: scripts/run_modernized_component_benchmark.sh --component PART --baseline VALUE --candidate VALUE [pipeline options]

PART is one of detector, mask-encoder, proposal-encoder, or vlm.
Values are local checkpoint paths for detector, model IDs for encoders, and
server model names for VLM. Each paired run changes only that component.
Pipeline options: --scene NAME --stride N --data-root DIR --dataset NAME
                 --dataset-config PATH --input-mode masked|region-proposal
                 --no-consolidation

Use the same held-out scene, prompts, runtime, and dataset settings for both
arms. Outputs use separate benchmark suffixes so annotation cannot overwrite
the other arm. Review scripts/MODERNIZED_COMPONENT_BENCHMARKS.md for scoring.
EOF
}

RUN_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --component) COMPONENT="$2"; shift 2 ;;
        --baseline) BASELINE="$2"; shift 2 ;;
        --candidate) CANDIDATE="$2"; shift 2 ;;
        --scene) SCENE="$2"; shift 2 ;;
        --stride) STRIDE="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --dataset) DATASET_NAME="$2"; shift 2 ;;
        --dataset-config) DATASET_CONFIG="$2"; shift 2 ;;
        --input-mode) INPUT_MODE="$2"; shift 2 ;;
        --no-consolidation) RUN_ARGS+=(--no-consolidation); shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [[ -z "${COMPONENT}" || -z "${BASELINE}" || -z "${CANDIDATE}" ]]; then
    echo "Error: --component, --baseline, and --candidate are required." >&2
    usage >&2
    exit 2
fi
case "${COMPONENT}" in
    detector|mask-encoder|proposal-encoder|vlm) ;;
    *) echo "Unsupported component: ${COMPONENT}" >&2; usage >&2; exit 2 ;;
esac

if [[ -z "${INPUT_MODE}" ]]; then
    if [[ "${COMPONENT}" == "proposal-encoder" ]]; then
        INPUT_MODE="region-proposal"
    else
        INPUT_MODE="masked"
    fi
fi
case "${INPUT_MODE}" in
    masked|region-proposal) ;;
    *) echo "Unsupported input mode: ${INPUT_MODE}" >&2; usage >&2; exit 2 ;;
esac

run_arm() {
    local arm="$1"
    local value="$2"
    local suffix="${COMPONENT//-/_}_${arm}_stride${STRIDE}"
    local -a args=(--scene "${SCENE}" --stride "${STRIDE}" --data-root "${DATA_ROOT}" --dataset "${DATASET_NAME}" --dataset-config "${DATASET_CONFIG}" --input-mode "${INPUT_MODE}")
    args+=("${RUN_ARGS[@]}")
    local encoder_key="siglip2"
    case "${value}" in
        laion/*) encoder_key="clip" ;;
        qihoo360/fg-clip2-*) encoder_key="fg_clip2" ;;
        google/siglip2-*) encoder_key="siglip2" ;;
    esac
    case "${COMPONENT}" in
        detector) args+=(--yoloe-checkpoint "${value}") ;;
        mask-encoder) args+=(--mask-encoder "${encoder_key}" --mask-model-id "${value}") ;;
        proposal-encoder) args+=(--box-encoder "${encoder_key}" --box-model-id "${value}") ;;
        vlm) args+=(--vlm-model "${value}") ;;
    esac
    log "${COMPONENT} ${arm}: ${value}"
    DETECTIONS_EXP_SUFFIX="s_modernized_benchmark_${suffix}" \
    MAP_EXP_SUFFIX="r_modernized_benchmark_${suffix}" \
        "${SCRIPT_DIR}/run_modernized_pipeline_local.sh" "${args[@]}"
}

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
run_arm baseline "${BASELINE}"
run_arm candidate "${CANDIDATE}"
log "Both isolated arms completed. Record paired metrics using scripts/MODERNIZED_COMPONENT_BENCHMARKS.md"
