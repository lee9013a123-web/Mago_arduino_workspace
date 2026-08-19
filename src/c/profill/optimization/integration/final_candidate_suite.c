#include "campp_profill/optimization/final_candidate_suite.h"

#include <stddef.h>
#include <string.h>

#include "backends/cpu_aarch64/aarch64_kernels.h"
#include "bn_candidate.h"
#include "dequant_candidate.h"
#include "fused_quant_qconv_candidate.h"
#include "qconv_candidate.h"
#include "remaining_candidate.h"

typedef enum CamppFinalCandidateFamily {
    CAMPP_FINAL_FAMILY_NONE = 0,
    CAMPP_FINAL_FAMILY_QCONV,
    CAMPP_FINAL_FAMILY_FUSED_QCONV,
    CAMPP_FINAL_FAMILY_BN,
    CAMPP_FINAL_FAMILY_DEQUANT,
    CAMPP_FINAL_FAMILY_REMAINING
} CamppFinalCandidateFamily;

static const char *campp_final_remaining_name(
    uint16_t opcode, uint16_t kernel_id)
{
    if (kernel_id == CAMPP_AARCH64_PACKED_KERNEL_ID) {
        switch (opcode) {
        case CAMPP_OP_ADD: return "add_stride_optimized";
        case CAMPP_OP_RELU: return "relu_stride_optimized";
        case CAMPP_OP_EXPAND: return "expand_stride_optimized";
        case CAMPP_OP_SLICE: return "slice_stride_optimized";
        case CAMPP_OP_QUANTIZE_LINEAR:
            return "quantize_linear_stride_optimized";
        case CAMPP_OP_REDUCE_MEAN: return "reduce_mean_stride_optimized";
        case CAMPP_OP_AVERAGE_POOL:
            return "average_pool_stride_optimized";
        case CAMPP_OP_RESHAPE: return "reshape_stride_optimized";
        default: break;
        }
    }
    if (opcode == CAMPP_OP_MUL &&
        kernel_id == CAMPP_FUSION_EPILOGUE_KERNEL_ID) {
        return "fused_dequant_sigmoid_mul_optimized";
    }
    if (opcode == CAMPP_OP_CONCAT &&
        kernel_id == CAMPP_FUSION_STATS_POOLING_KERNEL_ID) {
        return "fused_statistics_pooling_optimized";
    }
    return NULL;
}

static const CamppKernelEntry *campp_final_selected_entry(
    uint16_t opcode, uint16_t kernel_id,
    CamppFinalCandidateFamily *out_family, const char **out_name)
{
    const CamppKernelEntry *entry;

    *out_family = CAMPP_FINAL_FAMILY_NONE;
    *out_name = NULL;
    if (opcode == CAMPP_OP_QLINEAR_CONV &&
        kernel_id == CAMPP_AARCH64_PACKED_KERNEL_ID) {
        *out_family = CAMPP_FINAL_FAMILY_QCONV;
        *out_name = "qlinear_conv_o4i4_mac_fixed";
        return campp_qconv_candidate_entry(CAMPP_QCONV_CANDIDATE_MAC_FIXED);
    }
    if (opcode == CAMPP_OP_QLINEAR_CONV &&
        kernel_id == CAMPP_FUSION_QUANT_QCONV_KERNEL_ID) {
        *out_family = CAMPP_FINAL_FAMILY_FUSED_QCONV;
        *out_name = "fused_quant_qlinear_conv_o4i4_combined_fixed";
        return campp_fused_qconv_candidate_entry(
            CAMPP_FUSED_QCONV_CANDIDATE_COMBINED_FIXED);
    }
    if (opcode == CAMPP_OP_BATCH_NORMALIZATION &&
        kernel_id == CAMPP_FUSION_BN_RELU_QUANT_KERNEL_ID) {
        *out_family = CAMPP_FINAL_FAMILY_BN;
        *out_name = "fused_bn_relu_quant_combined";
        return campp_bn_candidate_entry(CAMPP_BN_CANDIDATE_COMBINED);
    }
    if (opcode == CAMPP_OP_DEQUANTIZE_LINEAR &&
        kernel_id == CAMPP_AARCH64_PACKED_KERNEL_ID) {
        *out_family = CAMPP_FINAL_FAMILY_DEQUANT;
        *out_name = "dequantize_linear_neon_combined";
        return campp_dequant_candidate_entry(
            CAMPP_DEQUANT_CANDIDATE_NEON_COMBINED);
    }

    entry = campp_remaining_candidate_entry(
        CAMPP_REMAINING_CANDIDATE_OPTIMIZED, opcode, kernel_id);
    if (entry != NULL) {
        *out_family = CAMPP_FINAL_FAMILY_REMAINING;
        *out_name = campp_final_remaining_name(opcode, kernel_id);
    }
    return entry;
}

static void campp_final_count_entry(
    CamppFinalCandidateSuiteStats *stats, CamppFinalCandidateFamily family)
{
    switch (family) {
    case CAMPP_FINAL_FAMILY_QCONV: stats->qconv_entries += 1u; break;
    case CAMPP_FINAL_FAMILY_FUSED_QCONV:
        stats->fused_qconv_entries += 1u;
        break;
    case CAMPP_FINAL_FAMILY_BN: stats->bn_entries += 1u; break;
    case CAMPP_FINAL_FAMILY_DEQUANT: stats->dequant_entries += 1u; break;
    case CAMPP_FINAL_FAMILY_REMAINING:
        stats->remaining_entries += 1u;
        break;
    default: return;
    }
    stats->total_entries += 1u;
}

CamppStatus campp_final_candidate_suite_create(
    const CamppKernelRegistry *base_registry,
    CamppFinalCandidateSuite *out_suite)
{
    size_t index;

    if (base_registry == NULL || out_suite == NULL ||
        base_registry->backend_id != CAMPP_BACKEND_CPU_AARCH64 ||
        base_registry->entries == NULL ||
        base_registry->entry_count > CAMPP_FINAL_CANDIDATE_MAX_KERNELS) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }

    memset(out_suite, 0, sizeof(*out_suite));
    for (index = 0u; index < base_registry->entry_count; ++index) {
        const CamppKernelEntry *base = &base_registry->entries[index];
        const CamppKernelEntry *selected;
        CamppFinalCandidateFamily family;
        const char *name;

        out_suite->entries[index] = *base;
        selected = campp_final_selected_entry(
            base->opcode, base->kernel_id, &family, &name);
        if (selected == NULL) continue;
        if (selected->opcode != base->opcode ||
            selected->kernel_id != base->kernel_id ||
            selected->run == NULL || name == NULL) {
            memset(out_suite, 0, sizeof(*out_suite));
            return CAMPP_STATUS_INVALID_ARGUMENT;
        }
        out_suite->entries[index] = *selected;
        out_suite->entries[index].name = name;
        campp_final_count_entry(&out_suite->stats, family);
    }

    /* Registry 변경으로 후보가 조용히 누락된 채 측정되는 것을 허용하지 않는다. */
    if (out_suite->stats.qconv_entries != 1u ||
        out_suite->stats.fused_qconv_entries != 1u ||
        out_suite->stats.bn_entries != 1u ||
        out_suite->stats.dequant_entries != 1u ||
        out_suite->stats.remaining_entries != 10u ||
        out_suite->stats.total_entries != 14u) {
        memset(out_suite, 0, sizeof(*out_suite));
        return CAMPP_STATUS_MISSING_KERNEL;
    }

    out_suite->registry.backend_id = base_registry->backend_id;
    out_suite->registry.name = "cpu_aarch64_o4i4_final";
    out_suite->registry.entries = out_suite->entries;
    out_suite->registry.entry_count = base_registry->entry_count;
    return campp_kernel_registry_validate(&out_suite->registry);
}

const char *campp_final_candidate_suite_name(void)
{
    return "qconv_mac_fixed+fused_combined_fixed+bn_combined+"
        "dequant_neon_combined+remaining_optimized";
}
