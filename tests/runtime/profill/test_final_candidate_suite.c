#include <stdio.h>
#include <string.h>

#include "backends/cpu_aarch64/aarch64_kernels.h"
#include "bn_candidate.h"
#include "campp_profill/optimization/final_candidate_suite.h"
#include "dequant_candidate.h"
#include "fused_quant_qconv_candidate.h"
#include "qconv_candidate.h"
#include "remaining_candidate.h"

#define CHECK_TRUE(condition)                                           \
    do {                                                                \
        if (!(condition)) {                                             \
            fprintf(                                                    \
                stderr, "CHECK failed at line %d: %s\n",             \
                __LINE__, #condition);                                  \
            return 1;                                                   \
        }                                                               \
    } while (0)

#define CHECK_STATUS(expression)                                       \
    do {                                                                \
        const CamppStatus actual = (expression);                        \
        if (actual != CAMPP_STATUS_OK) {                                \
            fprintf(                                                    \
                stderr, "STATUS failed at line %d: %s\n",            \
                __LINE__, campp_status_name(actual));                   \
            return 1;                                                   \
        }                                                               \
    } while (0)

typedef struct ExpectedRemainingEntry {
    uint16_t opcode;
    uint16_t kernel_id;
    const char *name;
} ExpectedRemainingEntry;

static int check_entry(
    const CamppKernelRegistry *registry, uint16_t opcode, uint16_t kernel_id,
    const CamppKernelEntry *expected, const char *expected_name)
{
    const CamppKernelEntry *actual = NULL;
    CHECK_STATUS(campp_kernel_registry_lookup(
        registry, opcode, kernel_id, &actual));
    CHECK_TRUE(actual != NULL);
    CHECK_TRUE(expected != NULL);
    CHECK_TRUE(actual->run == expected->run);
    CHECK_TRUE(actual->scratch_bytes == expected->scratch_bytes);
    CHECK_TRUE(strcmp(actual->name, expected_name) == 0);
    return 0;
}

int main(void)
{
    static const ExpectedRemainingEntry remaining[] = {
        {CAMPP_OP_ADD, CAMPP_AARCH64_PACKED_KERNEL_ID,
         "add_stride_optimized"},
        {CAMPP_OP_RELU, CAMPP_AARCH64_PACKED_KERNEL_ID,
         "relu_stride_optimized"},
        {CAMPP_OP_EXPAND, CAMPP_AARCH64_PACKED_KERNEL_ID,
         "expand_stride_optimized"},
        {CAMPP_OP_SLICE, CAMPP_AARCH64_PACKED_KERNEL_ID,
         "slice_stride_optimized"},
        {CAMPP_OP_QUANTIZE_LINEAR, CAMPP_AARCH64_PACKED_KERNEL_ID,
         "quantize_linear_stride_optimized"},
        {CAMPP_OP_REDUCE_MEAN, CAMPP_AARCH64_PACKED_KERNEL_ID,
         "reduce_mean_stride_optimized"},
        {CAMPP_OP_MUL, CAMPP_FUSION_EPILOGUE_KERNEL_ID,
         "fused_dequant_sigmoid_mul_optimized"},
        {CAMPP_OP_AVERAGE_POOL, CAMPP_AARCH64_PACKED_KERNEL_ID,
         "average_pool_stride_optimized"},
        {CAMPP_OP_RESHAPE, CAMPP_AARCH64_PACKED_KERNEL_ID,
         "reshape_stride_optimized"},
        {CAMPP_OP_CONCAT, CAMPP_FUSION_STATS_POOLING_KERNEL_ID,
         "fused_statistics_pooling_optimized"}
    };
    const CamppKernelRegistry *base = campp_cpu_aarch64_registry();
    const CamppKernelEntry *base_sigmoid = NULL;
    const CamppKernelEntry *final_sigmoid = NULL;
    CamppFinalCandidateSuite suite;
    size_t index;

    CHECK_STATUS(campp_final_candidate_suite_create(base, &suite));
    CHECK_STATUS(campp_kernel_registry_validate(&suite.registry));
    CHECK_TRUE(suite.registry.backend_id == CAMPP_BACKEND_CPU_AARCH64);
    CHECK_TRUE(strcmp(suite.registry.name, "cpu_aarch64_o4i4_final") == 0);
    CHECK_TRUE(suite.registry.entry_count == base->entry_count);
    CHECK_TRUE(suite.stats.qconv_entries == 1u);
    CHECK_TRUE(suite.stats.fused_qconv_entries == 1u);
    CHECK_TRUE(suite.stats.bn_entries == 1u);
    CHECK_TRUE(suite.stats.dequant_entries == 1u);
    CHECK_TRUE(suite.stats.remaining_entries == 10u);
    CHECK_TRUE(suite.stats.total_entries == 14u);

    CHECK_TRUE(check_entry(
        &suite.registry, CAMPP_OP_QLINEAR_CONV,
        CAMPP_AARCH64_PACKED_KERNEL_ID,
        campp_qconv_candidate_entry(CAMPP_QCONV_CANDIDATE_MAC_FIXED),
        "qlinear_conv_o4i4_mac_fixed") == 0);
    CHECK_TRUE(check_entry(
        &suite.registry, CAMPP_OP_QLINEAR_CONV,
        CAMPP_FUSION_QUANT_QCONV_KERNEL_ID,
        campp_fused_qconv_candidate_entry(
            CAMPP_FUSED_QCONV_CANDIDATE_COMBINED_FIXED),
        "fused_quant_qlinear_conv_o4i4_combined_fixed") == 0);
    CHECK_TRUE(check_entry(
        &suite.registry, CAMPP_OP_BATCH_NORMALIZATION,
        CAMPP_FUSION_BN_RELU_QUANT_KERNEL_ID,
        campp_bn_candidate_entry(CAMPP_BN_CANDIDATE_COMBINED),
        "fused_bn_relu_quant_combined") == 0);
    CHECK_TRUE(check_entry(
        &suite.registry, CAMPP_OP_DEQUANTIZE_LINEAR,
        CAMPP_AARCH64_PACKED_KERNEL_ID,
        campp_dequant_candidate_entry(
            CAMPP_DEQUANT_CANDIDATE_NEON_COMBINED),
        "dequantize_linear_neon_combined") == 0);

    for (index = 0u; index < sizeof(remaining) / sizeof(remaining[0]); ++index) {
        CHECK_TRUE(check_entry(
            &suite.registry, remaining[index].opcode,
            remaining[index].kernel_id,
            campp_remaining_candidate_entry(
                CAMPP_REMAINING_CANDIDATE_OPTIMIZED,
                remaining[index].opcode, remaining[index].kernel_id),
            remaining[index].name) == 0);
    }

    CHECK_STATUS(campp_kernel_registry_lookup(
        base, CAMPP_OP_SIGMOID, CAMPP_AARCH64_PACKED_KERNEL_ID,
        &base_sigmoid));
    CHECK_STATUS(campp_kernel_registry_lookup(
        &suite.registry, CAMPP_OP_SIGMOID,
        CAMPP_AARCH64_PACKED_KERNEL_ID, &final_sigmoid));
    CHECK_TRUE(final_sigmoid->run == base_sigmoid->run);
    CHECK_TRUE(final_sigmoid->scratch_bytes == base_sigmoid->scratch_bytes);
    CHECK_TRUE(strcmp(final_sigmoid->name, base_sigmoid->name) == 0);
    CHECK_TRUE(strcmp(
        campp_final_candidate_suite_name(),
        "qconv_mac_fixed+fused_combined_fixed+bn_combined+"
        "dequant_neon_combined+remaining_optimized") == 0);

    puts("final candidate suite tests passed");
    return 0;
}
