#!/usr/bin/env bash

# E7 baseline과 Operator profiler를 동일한 Release option으로 빌드한다.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build/profill}"
CC="${CC:-gcc}"
CFLAGS="${CFLAGS:--std=c11 -O3 -DNDEBUG -Wall -Wextra}"

mkdir -p "${BUILD_DIR}"

mapfile -t RUNTIME_SOURCES < <(
    find "${ROOT}/src/c/runtime" -name '*.c' \
        -not -path '*command_line*' | sort
)

COMMON_INCLUDES=(
    -I "${ROOT}/src/c/runtime"
    -I "${ROOT}/src/c/runtime/include"
)
PROFILL_INCLUDES=(
    -I "${ROOT}/src/c/profill/include"
)

echo "E7 profiling tools build (${CC})"
echo "  build dir: ${BUILD_DIR}"

# shellcheck disable=SC2086
"${CC}" ${CFLAGS} \
    "${COMMON_INCLUDES[@]}" \
    "${RUNTIME_SOURCES[@]}" \
    "${ROOT}/src/c/runtime/command_line/campp_runtime_benchmark.c" \
    -lm -o "${BUILD_DIR}/campp_runtime_benchmark"

# shellcheck disable=SC2086
"${CC}" ${CFLAGS} -DCAMPP_ENABLE_OPERATOR_PROFILING=1 \
    "${COMMON_INCLUDES[@]}" "${PROFILL_INCLUDES[@]}" \
    "${RUNTIME_SOURCES[@]}" \
    "${ROOT}/src/c/profill/operator_profiler.c" \
    "${ROOT}/src/c/profill/command_line/campp_e7_profiler.c" \
    -lm -o "${BUILD_DIR}/campp_e7_profiler"

# shellcheck disable=SC2086
"${CC}" ${CFLAGS} \
    "${COMMON_INCLUDES[@]}" "${PROFILL_INCLUDES[@]}" \
    "${ROOT}/src/c/profill/operator_profiler.c" \
    "${ROOT}/tests/runtime/profill/test_operator_profiler.c" \
    -lm -o "${BUILD_DIR}/test_operator_profiler"

# graph_executor의 compile-time hook이 실제 sample을 남기는지 검증한다.
# shellcheck disable=SC2086
"${CC}" ${CFLAGS} -DCAMPP_ENABLE_OPERATOR_PROFILING=1 \
    "${COMMON_INCLUDES[@]}" "${PROFILL_INCLUDES[@]}" \
    "${RUNTIME_SOURCES[@]}" \
    "${ROOT}/src/c/profill/operator_profiler.c" \
    "${ROOT}/tests/runtime/operator_replay_tests/test_graph_executor.c" \
    -lm -o "${BUILD_DIR}/test_profiled_graph_executor"

{
    printf 'cc=%s\n' "${CC}"
    printf 'cflags=%s\n' "${CFLAGS}"
    printf 'profiling_macro=CAMPP_ENABLE_OPERATOR_PROFILING=1\n'
} > "${BUILD_DIR}/build_metadata.txt"

echo "Build complete"
echo "  baseline: ${BUILD_DIR}/campp_runtime_benchmark"
echo "  profiler: ${BUILD_DIR}/campp_e7_profiler"
echo "  C test:   ${BUILD_DIR}/test_operator_profiler"
echo "  hook test:${BUILD_DIR}/test_profiled_graph_executor"
