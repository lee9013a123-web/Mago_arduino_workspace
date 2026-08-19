#!/usr/bin/env bash

# 최종 최적화 suite를 실제 E7 graph에 적용해 전체 latency와 Operator profile을
# 다시 측정한다. 계측 overhead 비교도 같은 suite의 비계측 binary를 사용한다.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build/profill}"
RESULT_TAG="${RESULT_TAG:-final_combined}"
RAW_DIR="${RAW_DIR:-${ROOT}/runs/profiling/e7_98/${RESULT_TAG}/raw}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/results/profiling/e7_98/${RESULT_TAG}}"

exec python3 "${ROOT}/scripts/4_profill/02_profile_e7.py" \
    --baseline-binary "${BUILD_DIR}/campp_runtime_benchmark_final" \
    --profiler-binary "${BUILD_DIR}/campp_e7_profiler_final" \
    --expected-suite final \
    --raw-dir "${RAW_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
