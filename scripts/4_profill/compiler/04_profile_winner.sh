#!/usr/bin/env bash

# Run the expensive operator-level Quick/Official profile only for the matrix
# baseline and final winner.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_ID=""
MODE="quick"
WINNER_ONLY=0
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id) RUN_ID="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --winner-only) WINNER_ONLY=1; shift ;;
        --force) FORCE=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${RUN_ID}" ]]; then
    echo "--run-id is required" >&2
    exit 2
fi
if [[ "${MODE}" != "quick" && "${MODE}" != "official" ]]; then
    echo "--mode must be quick or official" >&2
    exit 2
fi

COMPARE="${ROOT}/scripts/4_profill/compiler/03_compare_matrix.py"
PROFILE="${ROOT}/scripts/4_profill/02_profile_e7.py"
WINNER_BUILD="$(python3 "${COMPARE}" --run-id "${RUN_ID}" --print-field winner-build-dir)"
WINNER_VARIANT="$(python3 "${COMPARE}" --run-id "${RUN_ID}" --print-field winner-variant)"
BASELINE_BUILD="$(python3 "${COMPARE}" --run-id "${RUN_ID}" --print-field baseline-build-dir)"
BASELINE_VARIANT="$(python3 "${COMPARE}" --run-id "${RUN_ID}" --print-field baseline-variant)"

profile_variant() {
    local label="$1"
    local variant="$2"
    local build_dir="$3"
    local force_args=()
    if [[ "${FORCE}" == "1" ]]; then force_args=(--force); fi
    python3 "${PROFILE}" \
        --baseline-binary "${build_dir}/campp_runtime_benchmark_final" \
        --profiler-binary "${build_dir}/campp_e7_profiler_final" \
        --expected-suite final \
        --mode "${MODE}" \
        --raw-dir "${ROOT}/runs/profiling/e7_98/compiler_matrix/${RUN_ID}/profiles/${label}/raw" \
        --output-dir "${ROOT}/results/profiling/e7_98/compiler_matrix/${RUN_ID}/finalists/${label}" \
        "${force_args[@]}"
    echo "profiled ${label}: ${variant}"
}

if [[ "${WINNER_ONLY}" != "1" ]]; then
    profile_variant baseline "${BASELINE_VARIANT}" "${BASELINE_BUILD}"
fi
if [[ "${WINNER_BUILD}" != "${BASELINE_BUILD}" || "${WINNER_ONLY}" == "1" ]]; then
    profile_variant winner "${WINNER_VARIANT}" "${WINNER_BUILD}"
fi
