#!/usr/bin/env bash
# C Reference Runtime으로 고정 feature를 실행해 Tensor dump를 만든다.
# Phase 16의 기본 대상은 3초(298 frame)이며, BUCKETS를 지정하면 다른 plan도
# 같은 방식으로 실행할 수 있다.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build}"
BUNDLE_DIR="${BUNDLE_DIR:-${ROOT}/models/compiled/reference}"
ORT_DIR="${ORT_DIR:-${ROOT}/runs/runtime/ort_reference}"
OUT_DIR="${OUT_DIR:-${ROOT}/runs/runtime/c_reference}"
BUCKETS="${BUCKETS:-298}"
DUMP_TOOL="${DUMP_TOOL:-${BUILD_DIR}/campp_reference_dump}"

if [[ ! -x "${DUMP_TOOL}" && -x "${DUMP_TOOL}.exe" ]]; then
    DUMP_TOOL="${DUMP_TOOL}.exe"
fi
if [[ ! -x "${DUMP_TOOL}" ]]; then
    echo "Tensor dump 실행 파일이 없습니다: ${DUMP_TOOL}" >&2
    echo "먼저 scripts/3_runtime/03_build_reference_runtime.sh을 실행하세요." >&2
    exit 1
fi

weights="${BUNDLE_DIR}/weights.bin"
if [[ ! -f "${weights}" ]]; then
    echo "weights 파일이 없습니다: ${weights}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

for frames in ${BUCKETS}; do
    plan="${BUNDLE_DIR}/execution_plans/plan_${frames}.bin"
    feature="${ORT_DIR}/feature_${frames}.f32"
    if [[ ! -f "${plan}" ]]; then
        echo "execution plan이 없습니다: ${plan}" >&2
        exit 1
    fi
    if [[ ! -f "${feature}" ]]; then
        echo "ORT와 공유할 feature 입력이 없습니다: ${feature}" >&2
        echo "먼저 scripts/3_runtime/02_dump_ort_references.py를 실행하세요." >&2
        exit 1
    fi

    echo "[${frames} frames] C Reference Runtime 실행"
    "${DUMP_TOOL}" \
        "${plan}" \
        "${weights}" \
        "${feature}" \
        "${OUT_DIR}/c_${frames}"
done

echo "Tensor dump 저장 완료: ${OUT_DIR}"
