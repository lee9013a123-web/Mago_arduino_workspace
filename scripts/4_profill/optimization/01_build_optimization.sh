#!/usr/bin/env bash

# E7 Operator 내부 진단용 microbenchmark를 Release 최적화 + debug symbol로 빌드한다.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build/profill/optimization}"
CC="${CC:-gcc}"
CFLAGS="${CFLAGS:--std=c11 -O3 -g -DNDEBUG -Wall -Wextra}"

mkdir -p "${BUILD_DIR}"

mapfile -t RUNTIME_SOURCES < <(
    find "${ROOT}/src/c/runtime" -name '*.c' \
        -not -path '*command_line*' | sort
)

INCLUDES=(
    -I "${ROOT}/src/c/runtime"
    -I "${ROOT}/src/c/runtime/include"
    -I "${ROOT}/src/c/profill/include"
    -I "${ROOT}/src/c/profill/optimization/include"
)

# shellcheck disable=SC2086
"${CC}" ${CFLAGS} -DCAMPP_ENABLE_OPTIMIZATION_DIAGNOSTICS=1 \
    "${INCLUDES[@]}" \
    "${RUNTIME_SOURCES[@]}" \
    "${ROOT}/src/c/profill/operator_profiler.c" \
    "${ROOT}/src/c/profill/runtime_fixture.c" \
    "${ROOT}/src/c/profill/optimization/diagnostics/stage_probe.c" \
    "${ROOT}/src/c/profill/optimization/diagnostics/linux_pmu.c" \
    "${ROOT}/src/c/profill/optimization/command_line/campp_operator_microbench.c" \
    -lm -o "${BUILD_DIR}/campp_operator_microbench"

# shellcheck disable=SC2086
"${CC}" ${CFLAGS} \
    "${INCLUDES[@]}" \
    "${ROOT}/src/c/profill/operator_profiler.c" \
    "${ROOT}/src/c/profill/optimization/diagnostics/stage_probe.c" \
    "${ROOT}/tests/runtime/profill/test_optimization_probe.c" \
    -lm -o "${BUILD_DIR}/test_optimization_probe"

{
    printf 'cc=%s\n' "${CC}"
    printf 'cflags=%s\n' "${CFLAGS}"
    printf 'diagnostic_macro=CAMPP_ENABLE_OPTIMIZATION_DIAGNOSTICS=1\n'
} > "${BUILD_DIR}/build_metadata.txt"

echo "Optimization diagnostics build complete"
echo "  microbench: ${BUILD_DIR}/campp_operator_microbench"
echo "  C test:     ${BUILD_DIR}/test_optimization_probe"
