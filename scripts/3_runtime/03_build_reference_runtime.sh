#!/usr/bin/env bash
# Phase 3의 세 번째 실행 스크립트다.
#
# 역할:
# - cpu_reference backend와 replay test/dump 도구를 보드에서 빌드한다.
# - 산출물은 build/ 아래에 둔다.
#
# src/c/runtime/CMakeLists.txt는 아직 target을 선언하지 않은 상태라 여기서는
# 컴파일러를 직접 호출한다. CMakeLists.txt가 채워지면 이 스크립트를 cmake
# 호출로 교체한다.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build}"
CC="${CC:-gcc}"
CFLAGS="${CFLAGS:--std=c11 -O2 -Wall -Wextra}"

mkdir -p "${BUILD_DIR}"

mapfile -t RUNTIME_SOURCES < <(
    find "${ROOT}/src/c/runtime" -name '*.c' \
        -not -path '*cpu_aarch64*' \
        -not -path '*command_line*' | sort
)

echo "Reference Runtime 빌드 (${CC})"
echo "  runtime sources: ${#RUNTIME_SOURCES[@]}개"

build_target() {
    local name="$1"
    local entry="$2"
    echo "  ${name}"
    # shellcheck disable=SC2086
    "${CC}" ${CFLAGS} \
        -I "${ROOT}/src/c/runtime" \
        -I "${ROOT}/src/c/runtime/include" \
        "${RUNTIME_SOURCES[@]}" "${entry}" \
        -lm -o "${BUILD_DIR}/${name}"
}

build_target campp_reference_dump \
    "${ROOT}/tests/runtime/operator_replay_tests/campp_reference_dump.c"
build_target campp_operator_replay \
    "${ROOT}/tests/runtime/operator_replay_tests/campp_operator_replay.c"
build_target test_graph_executor \
    "${ROOT}/tests/runtime/operator_replay_tests/test_graph_executor.c"
build_target test_memory_bounds \
    "${ROOT}/tests/runtime/memory_safety_tests/test_memory_bounds.c"
build_target test_tensor_arena \
    "${ROOT}/tests/runtime/memory_safety_tests/test_tensor_arena.c"
build_target test_tensor_arena_equivalence \
    "${ROOT}/tests/runtime/memory_safety_tests/test_tensor_arena_equivalence.c"

echo "빌드 완료: ${BUILD_DIR}"
