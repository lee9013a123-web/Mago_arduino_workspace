#!/usr/bin/env bash
# Phase 3의 네 번째 실행 스크립트다.
#
# 역할:
# - weights.bin과 plan_98/298/498/998.bin을 차례로 선택한다.
# - 02번 스크립트가 저장한 것과 같은 feature 입력으로 C Reference Runtime을
#   실행한다.
# - 모든 중간 Tensor와 최종 embedding을 덤프한다.
#
# 원시 출력은 runs/runtime/c_reference에 저장한다.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build}"
BUNDLE_DIR="${BUNDLE_DIR:-${ROOT}/models/compiled/reference}"
ORT_DIR="${ORT_DIR:-${ROOT}/runs/runtime/ort_reference}"
OUT_DIR="${OUT_DIR:-${ROOT}/runs/runtime/c_reference}"
BUCKETS="${BUCKETS:-98 298 498 998}"

DUMP_TOOL="${BUILD_DIR}/campp_reference_dump"
if [[ ! -x "${DUMP_TOOL}" ]]; then
    echo "덤프 도구가 없다: ${DUMP_TOOL}" >&2
    echo "먼저 03_build_reference_runtime.sh를 실행한다." >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

for frames in ${BUCKETS}; do
    plan="${BUNDLE_DIR}/execution_plans/plan_${frames}.bin"
    feature="${ORT_DIR}/feature_${frames}.f32"
    if [[ ! -f "${feature}" ]]; then
        echo "feature 입력이 없다: ${feature}" >&2
        echo "먼저 02_dump_ort_references.py를 실행한다." >&2
        exit 1
    fi
    echo "[${frames} frames] C Reference Runtime 실행"
    "${DUMP_TOOL}" \
        "${plan}" \
        "${BUNDLE_DIR}/weights.bin" \
        "${feature}" \
        "${OUT_DIR}/c_${frames}"
done

echo "저장 완료: ${OUT_DIR}"
