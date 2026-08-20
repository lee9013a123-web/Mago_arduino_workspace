#!/usr/bin/env bash

# 최종 최적화 suite를 실제 E7 graph에 적용해 전체 latency와 Operator profile을
# 다시 측정한다. 계측 overhead 비교도 같은 suite의 비계측 binary를 사용한다.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build/profill/final_v2_aggressive}"
RESULT_TAG="${RESULT_TAG:-final_v2_aggressive}"
RAW_DIR="${RAW_DIR:-${ROOT}/runs/profiling/e7_98/${RESULT_TAG}/raw}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/results/profiling/e7_98/${RESULT_TAG}}"
EXPECTED_SUITE_CONFIG="${EXPECTED_SUITE_CONFIG:-qconv_v5+fused_combined_v5+bn_v2_spatial2+dequant_neon_combined+fused_dqrq_neon+remaining_optimized}"

exec python3 "${ROOT}/scripts/4_profill/02_profile_e7.py" \
    --baseline-binary "${BUILD_DIR}/campp_runtime_benchmark_final" \
    --profiler-binary "${BUILD_DIR}/campp_e7_profiler_final" \
    --expected-suite final \
    --expected-suite-config "${EXPECTED_SUITE_CONFIG}" \
    --raw-dir "${RAW_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
