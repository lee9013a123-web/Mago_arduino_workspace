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

CANDIDATE_DIR="${ROOT}/src/c/profill/optimization/candidates/qlinear_conv"
QCONV_MICROKERNEL_DIR="${CANDIDATE_DIR}/microkernels"
FUSED_QCONV_CANDIDATE_DIR="${ROOT}/src/c/profill/optimization/candidates/fused_quant_qconv"
BN_CANDIDATE_DIR="${ROOT}/src/c/profill/optimization/candidates/bn_relu_quant"
DEQUANT_CANDIDATE_DIR="${ROOT}/src/c/profill/optimization/candidates/dequantize_linear"
COMMON_CANDIDATE_DIR="${ROOT}/src/c/profill/optimization/candidates/common"
REMAINING_CANDIDATE_DIR="${ROOT}/src/c/profill/optimization/candidates/remaining_ops"
CANDIDATE_SOURCES=(
    "${CANDIDATE_DIR}/qconv_address_fastpath.c"
    "${CANDIDATE_DIR}/qconv_mac_neon.c"
    "${QCONV_MICROKERNEL_DIR}/qconv_mac_4x8.c"
    "${QCONV_MICROKERNEL_DIR}/qconv_mac_4x8_intrinsics.c"
    "${QCONV_MICROKERNEL_DIR}/qconv_mac_4x8_aarch64.S"
    "${CANDIDATE_DIR}/qconv_candidate.c"
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

INCLUDES=(
    -I "${ROOT}/src/c/runtime"
    -I "${ROOT}/src/c/runtime/include"
    -I "${ROOT}/src/c/profill/include"
    -I "${ROOT}/src/c/profill/optimization/include"
    -I "${CANDIDATE_DIR}"
    -I "${QCONV_MICROKERNEL_DIR}"
    -I "${FUSED_QCONV_CANDIDATE_DIR}"
    -I "${BN_CANDIDATE_DIR}"
    -I "${DEQUANT_CANDIDATE_DIR}"
    -I "${COMMON_CANDIDATE_DIR}"
    -I "${REMAINING_CANDIDATE_DIR}"
)

# shellcheck disable=SC2086
"${CC}" ${CFLAGS} -DCAMPP_ENABLE_OPTIMIZATION_DIAGNOSTICS=1 \
    "${INCLUDES[@]}" \
    "${RUNTIME_SOURCES[@]}" \
    "${ROOT}/src/c/profill/operator_profiler.c" \
    "${ROOT}/src/c/profill/runtime_fixture.c" \
    "${ROOT}/src/c/profill/optimization/diagnostics/stage_probe.c" \
    "${ROOT}/src/c/profill/optimization/diagnostics/linux_pmu.c" \
    "${ROOT}/src/c/profill/optimization/diagnostics/perf_sample_window.c" \
    "${CANDIDATE_SOURCES[@]}" \
    "${ROOT}/src/c/profill/optimization/command_line/campp_operator_microbench.c" \
    -lm -o "${BUILD_DIR}/campp_operator_microbench"

# perf annotate용 binary. Runtime kernel에는 stage clock 호출을 컴파일하지 않는다.
# shellcheck disable=SC2086
"${CC}" ${CFLAGS} \
    "${INCLUDES[@]}" \
    "${RUNTIME_SOURCES[@]}" \
    "${ROOT}/src/c/profill/operator_profiler.c" \
    "${ROOT}/src/c/profill/runtime_fixture.c" \
    "${ROOT}/src/c/profill/optimization/diagnostics/stage_probe.c" \
    "${ROOT}/src/c/profill/optimization/diagnostics/linux_pmu.c" \
    "${ROOT}/src/c/profill/optimization/diagnostics/perf_sample_window.c" \
    "${CANDIDATE_SOURCES[@]}" \
    "${ROOT}/src/c/profill/optimization/command_line/campp_operator_microbench.c" \
    -lm -o "${BUILD_DIR}/campp_operator_hotspot"

# shellcheck disable=SC2086
"${CC}" ${CFLAGS} \
    "${INCLUDES[@]}" \
    "${ROOT}/src/c/profill/operator_profiler.c" \
    "${ROOT}/src/c/profill/optimization/diagnostics/stage_probe.c" \
    "${ROOT}/src/c/profill/optimization/diagnostics/perf_sample_window.c" \
    "${ROOT}/tests/runtime/profill/test_optimization_probe.c" \
    -lm -o "${BUILD_DIR}/test_optimization_probe"

# shellcheck disable=SC2086
"${CC}" ${CFLAGS} \
    "${INCLUDES[@]}" \
    "${RUNTIME_SOURCES[@]}" \
    "${CANDIDATE_SOURCES[@]}" \
    "${ROOT}/tests/runtime/profill/test_qconv_candidate.c" \
    -lm -o "${BUILD_DIR}/test_qconv_candidate"

# shellcheck disable=SC2086
"${CC}" ${CFLAGS} \
    "${INCLUDES[@]}" \
    "${RUNTIME_SOURCES[@]}" \
    "${CANDIDATE_SOURCES[@]}" \
    "${ROOT}/tests/runtime/profill/test_qconv_microkernel_4x8.c" \
    -lm -o "${BUILD_DIR}/test_qconv_microkernel_4x8"

# shellcheck disable=SC2086
"${CC}" ${CFLAGS} \
    "${INCLUDES[@]}" \
    "${RUNTIME_SOURCES[@]}" \
    "${CANDIDATE_SOURCES[@]}" \
    "${ROOT}/tests/runtime/profill/test_bn_candidate.c" \
    -lm -o "${BUILD_DIR}/test_bn_candidate"

# shellcheck disable=SC2086
"${CC}" ${CFLAGS} \
    "${INCLUDES[@]}" \
    "${RUNTIME_SOURCES[@]}" \
    "${CANDIDATE_SOURCES[@]}" \
    "${ROOT}/tests/runtime/profill/test_dequant_candidate.c" \
    -lm -o "${BUILD_DIR}/test_dequant_candidate"

# shellcheck disable=SC2086
"${CC}" ${CFLAGS} \
    "${INCLUDES[@]}" \
    "${RUNTIME_SOURCES[@]}" \
    "${CANDIDATE_SOURCES[@]}" \
    "${ROOT}/tests/runtime/profill/test_remaining_candidates.c" \
    -lm -o "${BUILD_DIR}/test_remaining_candidates"

{
    printf 'cc=%s\n' "${CC}"
    printf 'cflags=%s\n' "${CFLAGS}"
    printf 'diagnostic_macro=CAMPP_ENABLE_OPTIMIZATION_DIAGNOSTICS=1\n'
    printf 'hotspot_stage_probe=disabled\n'
    printf 'qconv_candidate_modes=baseline,address,mac,combined,mac_fixed,mac_asm\n'
    printf 'fused_qconv_candidate_modes=baseline,mac,combined\n'
    printf 'bn_candidate_modes=baseline,address,affine,quant,combined\n'
    printf 'dequant_candidate_modes=baseline,address,parameter,scalar_combined,neon_combined\n'
    printf 'remaining_candidate_modes=baseline,optimized\n'
} > "${BUILD_DIR}/build_metadata.txt"

echo "Optimization diagnostics build complete"
echo "  microbench: ${BUILD_DIR}/campp_operator_microbench"
echo "  hotspot:    ${BUILD_DIR}/campp_operator_hotspot"
echo "  C test:     ${BUILD_DIR}/test_optimization_probe"
echo "  QConv test: ${BUILD_DIR}/test_qconv_candidate"
echo "  QConv 4x8:  ${BUILD_DIR}/test_qconv_microkernel_4x8"
echo "  BN test:    ${BUILD_DIR}/test_bn_candidate"
echo "  Dequant:    ${BUILD_DIR}/test_dequant_candidate"
echo "  Remaining:  ${BUILD_DIR}/test_remaining_candidates"
