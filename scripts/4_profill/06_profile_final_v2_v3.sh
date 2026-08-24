#!/usr/bin/env bash

# final V2와 layer-hybrid V3를 같은 입력/프로토콜로 연속 측정한다.
# E2E보다 먼저 retained tensor 전체 bitwise gate를 통과시킨다.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
V2_BUILD_DIR="${V2_BUILD_DIR:-${ROOT}/build/profill/final_v2_aggressive}"
V3_BUILD_DIR="${V3_BUILD_DIR:-${ROOT}/build/profill/final_v3_hybrid}"
V2_SUITE="qconv_v4+fused_combined_hybrid+bn_v2_spatial2+dequant_neon_combined+fused_dqrq_neon+remaining_optimized"
V3_SUITE="qconv_layer_hybrid_v3+fused_layer_hybrid_v3+bn_v2_spatial2+dequant_neon_combined+fused_dqrq_neon+remaining_optimized"

PRE_FLIGHT_ONLY=0
FORCE=0
PROFILE_ARGS=()
for argument in "$@"; do
    case "${argument}" in
        --preflight-only)
            PRE_FLIGHT_ONLY=1
            PROFILE_ARGS+=("${argument}")
            ;;
        --force)
            FORCE=1
            PROFILE_ARGS+=("${argument}")
            ;;
        *)
            PROFILE_ARGS+=("${argument}")
            ;;
    esac
done

VALIDATION_ARGS=(
    --v2-binary "${V2_BUILD_DIR}/campp_reference_dump_final"
    --v3-binary "${V3_BUILD_DIR}/campp_reference_dump_final"
)
if [[ "${PRE_FLIGHT_ONLY}" -eq 1 ]]; then
    VALIDATION_ARGS+=(--preflight-only)
elif [[ "${FORCE}" -eq 1 ]]; then
    VALIDATION_ARGS+=(--force)
fi

echo "[1/4] retained tensor V2/V3 bitwise validation"
python3 "${ROOT}/scripts/4_profill/optimization/15_validate_final_v2_v3.py" \
    "${VALIDATION_ARGS[@]}"

echo "[2/4] final V2 E2E profile"
BUILD_DIR="${V2_BUILD_DIR}" \
RESULT_TAG="final_v2_aggressive" \
EXPECTED_SUITE_CONFIG="${V2_SUITE}" \
    bash "${ROOT}/scripts/4_profill/03_profile_final_e7.sh" \
    "${PROFILE_ARGS[@]}"

echo "[3/4] final V3 E2E profile"
BUILD_DIR="${V3_BUILD_DIR}" \
RESULT_TAG="final_v3_hybrid" \
EXPECTED_SUITE_CONFIG="${V3_SUITE}" \
    bash "${ROOT}/scripts/4_profill/03_profile_final_e7.sh" \
    "${PROFILE_ARGS[@]}"

if [[ "${PRE_FLIGHT_ONLY}" -eq 1 ]]; then
    echo "[4/4] comparison skipped during preflight"
    exit 0
fi

COMPARE_ARGS=()
if [[ "${FORCE}" -eq 1 ]]; then
    COMPARE_ARGS+=(--force)
fi
echo "[4/4] final V3 versus V2 report"
python3 "${ROOT}/scripts/4_profill/optimization/14_compare_final_v2_v3.py" \
    "${COMPARE_ARGS[@]}"
