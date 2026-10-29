#!/usr/bin/env bash
# Run the existing ConceptGraphs pipeline against one edge-compatible server.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/edge-compatible/profile.sh"
EDGE_STRIDE="${STRIDE:-50}"
for ((arg_index = 1; arg_index <= $#; arg_index++)); do
    if [[ "${!arg_index}" == "--stride" ]] && ((arg_index + 1 < $#)); then
        next_index=$((arg_index + 1))
        EDGE_STRIDE="${!next_index}"
        break
    fi
done
export DETECTIONS_EXP_SUFFIX="${DETECTIONS_EXP_SUFFIX:-s_detections_stride${EDGE_STRIDE}_edge-compatible}"
export MAP_EXP_SUFFIX="${MAP_EXP_SUFFIX:-r_mapping_stride${EDGE_STRIDE}_edge-compatible}"
exec "${SCRIPT_DIR}/run_pipeline_local.sh" "$@"
