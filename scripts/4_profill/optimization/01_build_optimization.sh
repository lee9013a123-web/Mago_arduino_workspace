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
QCONV_V4_DIR="${ROOT}/src/c/profill/optimization/candidates/qlinear_conv_v4"
QCONV_V4_DISPATCH_DIR="${QCONV_V4_DIR}/dispatch"
QCONV_V4_PLANNING_DIR="${QCONV_V4_DIR}/planning"
QCONV_V4_PARAMETERS_DIR="${QCONV_V4_DIR}/parameters"
QCONV_V4_MICROKERNEL_DIR="${QCONV_V4_DIR}/microkernels"
QCONV_V4_REQUANT_DIR="${QCONV_V4_DIR}/requant"
QCONV_V4_STORE_DIR="${QCONV_V4_DIR}/store"
QCONV_V5_DIR="${ROOT}/src/c/profill/optimization/candidates/qlinear_conv_v5"
QCONV_V5_DISPATCH_DIR="${QCONV_V5_DIR}/dispatch"
QCONV_V5_PLANNING_DIR="${QCONV_V5_DIR}/planning"
QCONV_V5_PARAMETERS_DIR="${QCONV_V5_DIR}/parameters"
QCONV_V5_MICROKERNEL_DIR="${QCONV_V5_DIR}/microkernels"
QCONV_V5_PACKING_DIR="${QCONV_V5_DIR}/packing"
QCONV_V5_INSTRUMENTATION_DIR="${QCONV_V5_DIR}/instrumentation"
QCONV_ADDRESS_V2_DIR="${ROOT}/src/c/profill/optimization/candidates/qlinear_conv_address_v2"
QCONV_ADDRESS_V2_SOURCES=(
    "${QCONV_ADDRESS_V2_DIR}/planning/qconv_address_plan.c"
    "${QCONV_ADDRESS_V2_DIR}/kernels/one_by_one/qconv_address_1x1.c"
    "${QCONV_ADDRESS_V2_DIR}/kernels/three_by_three/qconv_address_3x3.c"
    "${QCONV_ADDRESS_V2_DIR}/kernels/generic/qconv_address_generic.c"
)
FUSED_QCONV_CANDIDATE_DIR="${ROOT}/src/c/profill/optimization/candidates/fused_quant_qconv"
BN_CANDIDATE_DIR="${ROOT}/src/c/profill/optimization/candidates/bn_relu_quant"
BN_V2_DIR="${ROOT}/src/c/profill/optimization/candidates/bn_relu_quant_v2"
BN_V2_PLANNING_DIR="${BN_V2_DIR}/planning"
BN_V2_PARAMETERS_DIR="${BN_V2_DIR}/parameters"
BN_V2_DISPATCH_DIR="${BN_V2_DIR}/dispatch"
BN_V2_MICROKERNEL_DIR="${BN_V2_DIR}/microkernels"
DEQUANT_CANDIDATE_DIR="${ROOT}/src/c/profill/optimization/candidates/dequantize_linear"
FUSED_DQRQ_CANDIDATE_DIR="${ROOT}/src/c/profill/optimization/candidates/fused_dequant_relu_quant"
COMMON_CANDIDATE_DIR="${ROOT}/src/c/profill/optimization/candidates/common"
REMAINING_CANDIDATE_DIR="${ROOT}/src/c/profill/optimization/candidates/remaining_ops"
CANDIDATE_SOURCES=(
    "${CANDIDATE_DIR}/qconv_address_fastpath.c"
    "${CANDIDATE_DIR}/qconv_mac_neon.c"
    "${QCONV_MICROKERNEL_DIR}/qconv_mac_4x8.c"
    "${QCONV_MICROKERNEL_DIR}/qconv_mac_4x8_intrinsics.c"
    "${QCONV_MICROKERNEL_DIR}/qconv_mac_4x8_aarch64.S"
    "${QCONV_V4_PLANNING_DIR}/qconv_v4_execution_plan.c"
    "${QCONV_V4_PLANNING_DIR}/qconv_v4_tile_plan.c"
    "${QCONV_V4_PARAMETERS_DIR}/qconv_v4_parameters.c"
    "${QCONV_V4_MICROKERNEL_DIR}/qconv_mac_1x1_8x8_intrinsics.c"
    "${QCONV_V4_MICROKERNEL_DIR}/qconv_mac_3x3_interior_8x8_intrinsics.c"
    "${QCONV_V4_MICROKERNEL_DIR}/qconv_mac_tail_intrinsics.c"
    "${QCONV_V4_REQUANT_DIR}/qconv_requant_neon8.c"
    "${QCONV_V4_STORE_DIR}/qconv_store_channel_packed.c"
    "${QCONV_V4_DISPATCH_DIR}/qconv_v4_dispatch.c"
    "${QCONV_V4_DISPATCH_DIR}/qconv_hybrid_dispatch.c"
    "${QCONV_V5_PLANNING_DIR}/qconv_v5_validated_raw_plan.c"
    "${QCONV_V5_PARAMETERS_DIR}/qconv_v5_zero_point_fastpath.c"
    "${QCONV_V5_MICROKERNEL_DIR}/qconv_mac_1x1_8x8_real_intrinsics.c"
    "${QCONV_V5_MICROKERNEL_DIR}/qconv_mac_3x3_cin32_unroll_intrinsics.c"
    "${QCONV_V5_MICROKERNEL_DIR}/qconv_mac_3x3_sliding_8x8_intrinsics.c"
    "${QCONV_V5_MICROKERNEL_DIR}/qconv_mac_tail_v5_intrinsics.c"
    "${QCONV_V5_PACKING_DIR}/qconv_v5_weight_pack.c"
    "${QCONV_V5_INSTRUMENTATION_DIR}/qconv_v5_stage_tags.c"
    "${QCONV_V5_DISPATCH_DIR}/qconv_v5_dispatch.c"
    "${CANDIDATE_DIR}/qconv_candidate.c"
    "${FUSED_QCONV_CANDIDATE_DIR}/fused_input_quant_neon.c"
    "${FUSED_QCONV_CANDIDATE_DIR}/fused_quant_qconv_candidate.c"
    "${BN_CANDIDATE_DIR}/bn_iteration_fastpath.c"
    "${BN_CANDIDATE_DIR}/bn_affine_fastpath.c"
    "${BN_CANDIDATE_DIR}/bn_quant_neon.c"
    "${BN_CANDIDATE_DIR}/bn_candidate.c"
    "${BN_V2_PLANNING_DIR}/bn_v2_execution_plan.c"
    "${BN_V2_PARAMETERS_DIR}/bn_v2_parameter_block.c"
    "${BN_V2_MICROKERNEL_DIR}/bn_neon16_exact.c"
    "${BN_V2_MICROKERNEL_DIR}/bn_neon16_spatial2.c"
    "${BN_V2_MICROKERNEL_DIR}/bn_neon16_prescaled.c"
    "${BN_V2_DISPATCH_DIR}/bn_v2_dispatch.c"
    "${BN_V2_DIR}/bn_v2_candidate.c"
    "${DEQUANT_CANDIDATE_DIR}/dequant_layout_plan.c"
    "${DEQUANT_CANDIDATE_DIR}/dequant_scalar_fastpath.c"
    "${DEQUANT_CANDIDATE_DIR}/dequant_neon.c"
    "${DEQUANT_CANDIDATE_DIR}/dequant_candidate.c"
    "${FUSED_DQRQ_CANDIDATE_DIR}/fused_dequant_relu_quant_candidate.c"
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
    -I "${QCONV_V4_DIR}"
    -I "${QCONV_V4_DISPATCH_DIR}"
    -I "${QCONV_V4_PLANNING_DIR}"
    -I "${QCONV_V4_PARAMETERS_DIR}"
    -I "${QCONV_V4_MICROKERNEL_DIR}"
    -I "${QCONV_V4_REQUANT_DIR}"
    -I "${QCONV_V4_STORE_DIR}"
    -I "${QCONV_V5_DIR}"
    -I "${QCONV_V5_DISPATCH_DIR}"
    -I "${QCONV_V5_PLANNING_DIR}"
    -I "${QCONV_V5_PARAMETERS_DIR}"
    -I "${QCONV_V5_MICROKERNEL_DIR}"
    -I "${QCONV_V5_PACKING_DIR}"
    -I "${QCONV_V5_INSTRUMENTATION_DIR}"
    -I "${QCONV_ADDRESS_V2_DIR}"
    -I "${FUSED_QCONV_CANDIDATE_DIR}"
    -I "${BN_CANDIDATE_DIR}"
    -I "${BN_V2_DIR}"
    -I "${BN_V2_PLANNING_DIR}"
    -I "${BN_V2_PARAMETERS_DIR}"
    -I "${BN_V2_DISPATCH_DIR}"
    -I "${BN_V2_MICROKERNEL_DIR}"
    -I "${DEQUANT_CANDIDATE_DIR}"
    -I "${FUSED_DQRQ_CANDIDATE_DIR}"
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

# fused family 전체를 입력당 단일 graph traversal로 측정한다. Stage probe를
# 제외해 baseline과 candidate 모두 production에 가까운 kernel body만 잰다.
# shellcheck disable=SC2086
"${CC}" ${CFLAGS} \
    "${INCLUDES[@]}" \
    "${RUNTIME_SOURCES[@]}" \
    "${ROOT}/src/c/profill/operator_profiler.c" \
    "${ROOT}/src/c/profill/runtime_fixture.c" \
    "${CANDIDATE_SOURCES[@]}" \
    "${ROOT}/src/c/profill/optimization/command_line/campp_fused_qconv_family_bench.c" \
    -lm -o "${BUILD_DIR}/campp_fused_qconv_family_bench"

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
    "${ROOT}/tests/runtime/profill/qconv_v4/test_qconv_v4_primitives.c" \
    -lm -o "${BUILD_DIR}/test_qconv_v4_primitives"

# shellcheck disable=SC2086
"${CC}" ${CFLAGS} \
    "${INCLUDES[@]}" \
    "${RUNTIME_SOURCES[@]}" \
    "${CANDIDATE_SOURCES[@]}" \
    "${ROOT}/tests/runtime/profill/qconv_v5/test_qconv_v5_primitives.c" \
    -lm -o "${BUILD_DIR}/test_qconv_v5_primitives"

# Address-v2 stays outside CANDIDATE_SOURCES until its standalone bitwise and
# latency gates pass. This target validates the independent provider contract.
# shellcheck disable=SC2086
"${CC}" ${CFLAGS} \
    "${INCLUDES[@]}" \
    "${RUNTIME_SOURCES[@]}" \
    "${QCONV_ADDRESS_V2_SOURCES[@]}" \
    "${ROOT}/tests/runtime/profill/qconv_address_v2/test_qconv_address_v2.c" \
    -lm -o "${BUILD_DIR}/test_qconv_address_v2"

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
    "${ROOT}/tests/runtime/profill/test_bn_v2_candidate.c" \
    -lm -o "${BUILD_DIR}/test_bn_v2_candidate"

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
    "${ROOT}/tests/runtime/profill/test_fused_dequant_relu_quant_candidate.c" \
    -lm -o "${BUILD_DIR}/test_fused_dequant_relu_quant_candidate"

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
    printf 'qconv_candidate_modes=baseline,address,mac,combined,mac_fixed,mac_asm,v4,hybrid,v5\n'
    printf 'fused_qconv_candidate_modes=baseline,mac,combined,mac_fixed,quant_neon,combined_fixed,combined_v4,combined_hybrid\n'
    printf 'bn_candidate_modes=baseline,address,affine,quant,combined,v2_exact16,v2_spatial2,v2_prescaled\n'
    printf 'dequant_candidate_modes=baseline,address,parameter,scalar_combined,neon_combined\n'
    printf 'fused_dqrq_candidate_modes=baseline,scalar,neon\n'
    printf 'remaining_candidate_modes=baseline,optimized\n'
    printf 'fused_family_batch_graph_traversal=enabled\n'
} > "${BUILD_DIR}/build_metadata.txt"

echo "Optimization diagnostics build complete"
echo "  microbench: ${BUILD_DIR}/campp_operator_microbench"
echo "  fused batch:${BUILD_DIR}/campp_fused_qconv_family_bench"
echo "  hotspot:    ${BUILD_DIR}/campp_operator_hotspot"
echo "  C test:     ${BUILD_DIR}/test_optimization_probe"
echo "  QConv test: ${BUILD_DIR}/test_qconv_candidate"
echo "  QConv 4x8:  ${BUILD_DIR}/test_qconv_microkernel_4x8"
echo "  QConv addr: ${BUILD_DIR}/test_qconv_address_v2"
echo "  QConv v4:   ${BUILD_DIR}/test_qconv_v4_primitives"
echo "  QConv v5:   ${BUILD_DIR}/test_qconv_v5_primitives"
echo "  BN test:    ${BUILD_DIR}/test_bn_candidate"
echo "  BN v2 test: ${BUILD_DIR}/test_bn_v2_candidate"
echo "  Dequant:    ${BUILD_DIR}/test_dequant_candidate"
echo "  Fused DQRQ: ${BUILD_DIR}/test_fused_dequant_relu_quant_candidate"
echo "  Remaining:  ${BUILD_DIR}/test_remaining_candidates"
