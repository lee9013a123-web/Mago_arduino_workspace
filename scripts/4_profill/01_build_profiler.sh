#!/usr/bin/env bash

# E7 baseline과 Operator profiler를 동일한 Release option으로 빌드한다.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build/profill}"
CC="${CC:-gcc}"
CPPFLAGS="${CPPFLAGS:-}"
CFLAGS="${CFLAGS:--std=c11 -O3 -DNDEBUG -Wall -Wextra}"
LDFLAGS="${LDFLAGS:-}"
BUILD_VARIANT="${BUILD_VARIANT:-default}"
BUILD_SCOPE="${BUILD_SCOPE:-all}"
STRIP="${STRIP:-strip}"
STRIP_FINAL="${STRIP_FINAL:-0}"

if [[ "${BUILD_SCOPE}" != "all" && \
      "${BUILD_SCOPE}" != "compiler_matrix" && \
      "${BUILD_SCOPE}" != "compiler_quick" && \
      "${BUILD_SCOPE}" != "profiler_finalist" ]]; then
    echo "invalid BUILD_SCOPE: ${BUILD_SCOPE}" >&2
    exit 2
fi

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

QCONV_CANDIDATE_DIR="${ROOT}/src/c/profill/optimization/candidates/qlinear_conv"
QCONV_MICROKERNEL_DIR="${QCONV_CANDIDATE_DIR}/microkernels"
FUSED_QCONV_CANDIDATE_DIR="${ROOT}/src/c/profill/optimization/candidates/fused_quant_qconv"
BN_CANDIDATE_DIR="${ROOT}/src/c/profill/optimization/candidates/bn_relu_quant"
DEQUANT_CANDIDATE_DIR="${ROOT}/src/c/profill/optimization/candidates/dequantize_linear"
COMMON_CANDIDATE_DIR="${ROOT}/src/c/profill/optimization/candidates/common"
REMAINING_CANDIDATE_DIR="${ROOT}/src/c/profill/optimization/candidates/remaining_ops"
FINAL_SUITE_SOURCE="${ROOT}/src/c/profill/optimization/integration/final_candidate_suite.c"

FINAL_INCLUDES=(
    -I "${ROOT}/src/c/profill/optimization/include"
    -I "${QCONV_CANDIDATE_DIR}"
    -I "${QCONV_MICROKERNEL_DIR}"
    -I "${FUSED_QCONV_CANDIDATE_DIR}"
    -I "${BN_CANDIDATE_DIR}"
    -I "${DEQUANT_CANDIDATE_DIR}"
    -I "${COMMON_CANDIDATE_DIR}"
    -I "${REMAINING_CANDIDATE_DIR}"
)

FINAL_CANDIDATE_SOURCES=(
    "${QCONV_CANDIDATE_DIR}/qconv_address_fastpath.c"
    "${QCONV_CANDIDATE_DIR}/qconv_mac_neon.c"
    "${QCONV_MICROKERNEL_DIR}/qconv_mac_4x8.c"
    "${QCONV_MICROKERNEL_DIR}/qconv_mac_4x8_intrinsics.c"
    "${QCONV_MICROKERNEL_DIR}/qconv_mac_4x8_aarch64.S"
    "${QCONV_CANDIDATE_DIR}/qconv_candidate.c"
    "${FUSED_QCONV_CANDIDATE_DIR}/fused_input_quant_neon.c"
    "${FUSED_QCONV_CANDIDATE_DIR}/fused_quant_qconv_candidate.c"
    "${BN_CANDIDATE_DIR}/bn_iteration_fastpath.c"
    "${BN_CANDIDATE_DIR}/bn_affine_fastpath.c"
    "${BN_CANDIDATE_DIR}/bn_quant_neon.c"
    "${BN_CANDIDATE_DIR}/bn_candidate.c"
    "${DEQUANT_CANDIDATE_DIR}/dequant_layout_plan.c"
    "${DEQUANT_CANDIDATE_DIR}/dequant_scalar_fastpath.c"
    "${DEQUANT_CANDIDATE_DIR}/dequant_neon.c"
    "${DEQUANT_CANDIDATE_DIR}/dequant_candidate.c"
    "${COMMON_CANDIDATE_DIR}/packed_iteration.c"
    "${COMMON_CANDIDATE_DIR}/elementwise_neon.c"
    "${COMMON_CANDIDATE_DIR}/reduction_neon.c"
    "${COMMON_CANDIDATE_DIR}/block_copy_fastpath.c"
    "${COMMON_CANDIDATE_DIR}/sigmoid_lut.c"
    "${REMAINING_CANDIDATE_DIR}/remaining_candidate.c"
)

echo "E7 profiling tools build (${CC})"
echo "  build dir: ${BUILD_DIR}"

if [[ "${BUILD_SCOPE}" == "all" ]]; then
    # shellcheck disable=SC2086
    "${CC}" ${CPPFLAGS} ${CFLAGS} \
        "${COMMON_INCLUDES[@]}" \
        "${RUNTIME_SOURCES[@]}" \
        "${ROOT}/src/c/runtime/command_line/campp_runtime_benchmark.c" \
        ${LDFLAGS} -lm -o "${BUILD_DIR}/campp_runtime_benchmark"

    # shellcheck disable=SC2086
    "${CC}" ${CPPFLAGS} ${CFLAGS} -DCAMPP_ENABLE_OPERATOR_PROFILING=1 \
        "${COMMON_INCLUDES[@]}" "${PROFILL_INCLUDES[@]}" \
        "${RUNTIME_SOURCES[@]}" \
        "${ROOT}/src/c/profill/operator_profiler.c" \
        "${ROOT}/src/c/profill/runtime_fixture.c" \
        "${ROOT}/src/c/profill/command_line/campp_e7_profiler.c" \
        ${LDFLAGS} -lm -o "${BUILD_DIR}/campp_e7_profiler"
fi

# 최종 후보를 실제 RuntimeContext registry에 적용한 비계측/계측 쌍이다.
# 두 실행 파일은 같은 runtime/candidate source와 Release option을 사용한다.
# shellcheck disable=SC2086
"${CC}" ${CPPFLAGS} ${CFLAGS} -DCAMPP_ENABLE_FINAL_CANDIDATE_SUITE=1 \
    "${COMMON_INCLUDES[@]}" "${PROFILL_INCLUDES[@]}" \
    "${FINAL_INCLUDES[@]}" \
    "${RUNTIME_SOURCES[@]}" \
    "${FINAL_CANDIDATE_SOURCES[@]}" \
    "${FINAL_SUITE_SOURCE}" \
    "${ROOT}/src/c/runtime/command_line/campp_runtime_benchmark.c" \
    ${LDFLAGS} -lm -o "${BUILD_DIR}/campp_runtime_benchmark_final"

if [[ "${BUILD_SCOPE}" != "compiler_quick" ]]; then
    # shellcheck disable=SC2086
    "${CC}" ${CPPFLAGS} ${CFLAGS} \
        -DCAMPP_ENABLE_OPERATOR_PROFILING=1 \
        -DCAMPP_ENABLE_FINAL_CANDIDATE_SUITE=1 \
        "${COMMON_INCLUDES[@]}" "${PROFILL_INCLUDES[@]}" \
        "${FINAL_INCLUDES[@]}" \
        "${RUNTIME_SOURCES[@]}" \
        "${ROOT}/src/c/profill/operator_profiler.c" \
        "${ROOT}/src/c/profill/runtime_fixture.c" \
        "${FINAL_CANDIDATE_SOURCES[@]}" \
        "${FINAL_SUITE_SOURCE}" \
        "${ROOT}/src/c/profill/command_line/campp_e7_profiler.c" \
        ${LDFLAGS} -lm -o "${BUILD_DIR}/campp_e7_profiler_final"
fi

if [[ "${BUILD_SCOPE}" == "all" ]]; then
    # shellcheck disable=SC2086
    "${CC}" ${CPPFLAGS} ${CFLAGS} \
        "${COMMON_INCLUDES[@]}" "${PROFILL_INCLUDES[@]}" \
        "${ROOT}/src/c/profill/operator_profiler.c" \
        "${ROOT}/tests/runtime/profill/test_operator_profiler.c" \
        ${LDFLAGS} -lm -o "${BUILD_DIR}/test_operator_profiler"

    # graph_executor의 compile-time hook이 실제 sample을 남기는지 검증한다.
    # shellcheck disable=SC2086
    "${CC}" ${CPPFLAGS} ${CFLAGS} -DCAMPP_ENABLE_OPERATOR_PROFILING=1 \
        "${COMMON_INCLUDES[@]}" "${PROFILL_INCLUDES[@]}" \
        "${RUNTIME_SOURCES[@]}" \
        "${ROOT}/src/c/profill/operator_profiler.c" \
        "${ROOT}/tests/runtime/operator_replay_tests/test_graph_executor.c" \
        ${LDFLAGS} -lm -o "${BUILD_DIR}/test_profiled_graph_executor"
fi

if [[ "${BUILD_SCOPE}" != "compiler_quick" ]]; then
    # 최종 registry가 정확한 14개 entry만 교체하는지 검증한다.
    # shellcheck disable=SC2086
    "${CC}" ${CPPFLAGS} ${CFLAGS} \
        "${COMMON_INCLUDES[@]}" "${PROFILL_INCLUDES[@]}" \
        "${FINAL_INCLUDES[@]}" \
        "${RUNTIME_SOURCES[@]}" \
        "${FINAL_CANDIDATE_SOURCES[@]}" \
        "${FINAL_SUITE_SOURCE}" \
        "${ROOT}/tests/runtime/profill/test_final_candidate_suite.c" \
        ${LDFLAGS} -lm -o "${BUILD_DIR}/test_final_candidate_suite"
fi

if [[ "${BUILD_SCOPE}" == "compiler_matrix" || \
      "${BUILD_SCOPE}" == "compiler_quick" ]]; then
    # Retained Tensor 전체를 compiler variant 사이에서 bitwise 비교한다.
    # shellcheck disable=SC2086
    "${CC}" ${CPPFLAGS} ${CFLAGS} \
        -DCAMPP_ENABLE_FINAL_CANDIDATE_SUITE=1 \
        "${COMMON_INCLUDES[@]}" "${PROFILL_INCLUDES[@]}" \
        "${FINAL_INCLUDES[@]}" \
        "${RUNTIME_SOURCES[@]}" \
        "${FINAL_CANDIDATE_SOURCES[@]}" \
        "${FINAL_SUITE_SOURCE}" \
        "${ROOT}/tests/runtime/operator_replay_tests/campp_reference_dump.c" \
        ${LDFLAGS} -lm -o "${BUILD_DIR}/campp_reference_dump_final"
fi

if [[ "${STRIP_FINAL}" == "1" ]]; then
    cp "${BUILD_DIR}/campp_runtime_benchmark_final" \
        "${BUILD_DIR}/campp_runtime_benchmark_final.unstripped"
    "${STRIP}" --strip-unneeded \
        "${BUILD_DIR}/campp_runtime_benchmark_final"
fi

{
    printf 'cc=%s\n' "${CC}"
    printf 'cppflags=%s\n' "${CPPFLAGS}"
    printf 'cflags=%s\n' "${CFLAGS}"
    printf 'ldflags=%s\n' "${LDFLAGS}"
    printf 'build_variant=%s\n' "${BUILD_VARIANT}"
    printf 'build_scope=%s\n' "${BUILD_SCOPE}"
    printf 'strip_final=%s\n' "${STRIP_FINAL}"
    printf 'profiling_macro=CAMPP_ENABLE_OPERATOR_PROFILING=1\n'
    printf 'final_suite_macro=CAMPP_ENABLE_FINAL_CANDIDATE_SUITE=1\n'
    printf 'final_suite=%s\n' 'qconv_mac_fixed+fused_combined_fixed+bn_combined+dequant_neon_combined+remaining_optimized'
} > "${BUILD_DIR}/build_metadata.txt"

metadata_binaries=(
    campp_runtime_benchmark_final
)
if [[ "${BUILD_SCOPE}" != "compiler_quick" ]]; then
    metadata_binaries+=(campp_e7_profiler_final)
fi
if [[ "${BUILD_SCOPE}" == "all" ]]; then
    metadata_binaries+=(campp_runtime_benchmark campp_e7_profiler)
fi
for binary in "${metadata_binaries[@]}"; do
    cp "${BUILD_DIR}/build_metadata.txt" \
        "${BUILD_DIR}/${binary}.build.txt"
done

echo "Build complete"
if [[ "${BUILD_SCOPE}" == "all" ]]; then
    echo "  baseline: ${BUILD_DIR}/campp_runtime_benchmark"
    echo "  profiler: ${BUILD_DIR}/campp_e7_profiler"
fi
echo "  final baseline: ${BUILD_DIR}/campp_runtime_benchmark_final"
if [[ "${BUILD_SCOPE}" != "compiler_quick" ]]; then
    echo "  final profiler: ${BUILD_DIR}/campp_e7_profiler_final"
fi
if [[ "${BUILD_SCOPE}" == "all" ]]; then
    echo "  C test:   ${BUILD_DIR}/test_operator_profiler"
    echo "  hook test:${BUILD_DIR}/test_profiled_graph_executor"
fi
if [[ "${BUILD_SCOPE}" != "compiler_quick" ]]; then
    echo "  suite test:${BUILD_DIR}/test_final_candidate_suite"
fi
if [[ "${BUILD_SCOPE}" == "compiler_matrix" || \
      "${BUILD_SCOPE}" == "compiler_quick" ]]; then
    echo "  tensor dump:${BUILD_DIR}/campp_reference_dump_final"
fi
