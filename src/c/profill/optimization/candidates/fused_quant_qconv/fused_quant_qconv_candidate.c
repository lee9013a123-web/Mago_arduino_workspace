#include "fused_quant_qconv_candidate.h"

#include <string.h>

#include "backends/cpu_aarch64/aarch64_kernels.h"
#include "backends/cpu_aarch64/fused_kernels/fused_quant_qconv_internal.h"
#include "fused_input_quant_neon.h"
#include "internal/runtime_model.h"
#include "qconv_candidate.h"

static CamppStatus campp_fused_qconv_candidate_run(
    CamppQconvCandidateMode qconv_mode, int use_quant_neon,
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count,
    void *scratch, size_t scratch_size)
{
    CamppKernelRun qconv_run = campp_aarch64_qlinear_conv_o4i4;
    CamppFusedInputQuantizeRun quantize_run =
        campp_fused_quantize_input_scalar;
    if (qconv_mode != CAMPP_QCONV_CANDIDATE_BASELINE) {
        const CamppKernelEntry *qconv =
            campp_qconv_candidate_entry(qconv_mode);
        if (qconv == NULL || qconv->run == NULL) {
            return CAMPP_STATUS_INVALID_ARGUMENT;
        }
        qconv_run = qconv->run;
    }
    if (use_quant_neon) quantize_run = campp_fused_input_quantize_neon;
    return campp_fused_quant_qlinear_conv_with_components(
        model, op, inputs, input_count, outputs, output_count,
        scratch, scratch_size, quantize_run, qconv_run);
}

#define CAMPP_DEFINE_FUSED_QCONV_CANDIDATE(                            \
    function_name, qconv_mode, use_quant_neon)                         \
    static CamppStatus function_name(                                  \
        const CamppRuntimeModel *model,                                \
        const CamppOperatorDescriptor *op,                             \
        const CamppTensorView *inputs, uint8_t input_count,            \
        CamppTensorView *outputs, uint8_t output_count,                 \
        void *scratch, size_t scratch_size)                            \
    {                                                                  \
        return campp_fused_qconv_candidate_run(                        \
            (qconv_mode), (use_quant_neon), model, op, inputs,         \
            input_count,                                               \
            outputs, output_count, scratch, scratch_size);             \
    }

CAMPP_DEFINE_FUSED_QCONV_CANDIDATE(
    campp_fused_qconv_candidate_mac, CAMPP_QCONV_CANDIDATE_MAC, 0)
CAMPP_DEFINE_FUSED_QCONV_CANDIDATE(
    campp_fused_qconv_candidate_combined,
    CAMPP_QCONV_CANDIDATE_COMBINED, 0)
CAMPP_DEFINE_FUSED_QCONV_CANDIDATE(
    campp_fused_qconv_candidate_mac_fixed,
    CAMPP_QCONV_CANDIDATE_MAC_FIXED, 0)
CAMPP_DEFINE_FUSED_QCONV_CANDIDATE(
    campp_fused_qconv_candidate_quant_neon,
    CAMPP_QCONV_CANDIDATE_BASELINE, 1)
CAMPP_DEFINE_FUSED_QCONV_CANDIDATE(
    campp_fused_qconv_candidate_combined_fixed,
    CAMPP_QCONV_CANDIDATE_MAC_FIXED, 1)

static const CamppKernelEntry CAMPP_FUSED_QCONV_CANDIDATE_ENTRIES[] = {
    {
        CAMPP_OP_QLINEAR_CONV,
        CAMPP_FUSION_QUANT_QCONV_KERNEL_ID,
        campp_fused_qconv_candidate_mac,
        campp_fused_quant_qlinear_conv_scratch_bytes,
        "fused_quant_qlinear_conv_o4i4"
    },
    {
        CAMPP_OP_QLINEAR_CONV,
        CAMPP_FUSION_QUANT_QCONV_KERNEL_ID,
        campp_fused_qconv_candidate_combined,
        campp_fused_quant_qlinear_conv_scratch_bytes,
        "fused_quant_qlinear_conv_o4i4"
    },
    {
        CAMPP_OP_QLINEAR_CONV,
        CAMPP_FUSION_QUANT_QCONV_KERNEL_ID,
        campp_fused_qconv_candidate_mac_fixed,
        campp_fused_quant_qlinear_conv_scratch_bytes,
        "fused_quant_qlinear_conv_o4i4"
    },
    {
        CAMPP_OP_QLINEAR_CONV,
        CAMPP_FUSION_QUANT_QCONV_KERNEL_ID,
        campp_fused_qconv_candidate_quant_neon,
        campp_fused_quant_qlinear_conv_scratch_bytes,
        "fused_quant_qlinear_conv_o4i4"
    },
    {
        CAMPP_OP_QLINEAR_CONV,
        CAMPP_FUSION_QUANT_QCONV_KERNEL_ID,
        campp_fused_qconv_candidate_combined_fixed,
        campp_fused_quant_qlinear_conv_scratch_bytes,
        "fused_quant_qlinear_conv_o4i4"
    }
};

const char *campp_fused_qconv_candidate_mode_name(
    CamppFusedQconvCandidateMode mode)
{
    static const char *const names[] = {
        "baseline", "mac", "combined", "mac_fixed", "quant_neon",
        "combined_fixed"
    };
    return mode >= CAMPP_FUSED_QCONV_CANDIDATE_BASELINE &&
        mode <= CAMPP_FUSED_QCONV_CANDIDATE_COMBINED_FIXED
        ? names[mode] : "invalid";
}

int campp_fused_qconv_candidate_mode_parse(
    const char *text, CamppFusedQconvCandidateMode *out_mode)
{
    CamppFusedQconvCandidateMode mode;
    if (text == NULL || out_mode == NULL) return 1;
    for (mode = CAMPP_FUSED_QCONV_CANDIDATE_BASELINE;
         mode <= CAMPP_FUSED_QCONV_CANDIDATE_COMBINED_FIXED;
         mode = (CamppFusedQconvCandidateMode)(mode + 1)) {
        if (strcmp(
                text, campp_fused_qconv_candidate_mode_name(mode)) == 0) {
            *out_mode = mode;
            return 0;
        }
    }
    return 1;
}

const CamppKernelEntry *campp_fused_qconv_candidate_entry(
    CamppFusedQconvCandidateMode mode)
{
    if (mode == CAMPP_FUSED_QCONV_CANDIDATE_BASELINE) return NULL;
    if (mode < CAMPP_FUSED_QCONV_CANDIDATE_MAC ||
        mode > CAMPP_FUSED_QCONV_CANDIDATE_COMBINED_FIXED) {
        return NULL;
    }
    return &CAMPP_FUSED_QCONV_CANDIDATE_ENTRIES[mode - 1];
}

#undef CAMPP_DEFINE_FUSED_QCONV_CANDIDATE
