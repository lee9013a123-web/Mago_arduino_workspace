#include "fused_quant_qconv_candidate.h"

#include <string.h>

#include "backends/cpu_aarch64/aarch64_kernels.h"
#include "backends/cpu_aarch64/fused_kernels/fused_quant_qconv_internal.h"
#include "internal/runtime_model.h"
#include "qconv_candidate.h"

static CamppStatus campp_fused_qconv_candidate_run(
    CamppQconvCandidateMode qconv_mode,
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count,
    void *scratch, size_t scratch_size)
{
    const CamppKernelEntry *qconv =
        campp_qconv_candidate_entry(qconv_mode);
    if (qconv == NULL || qconv->run == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    return campp_fused_quant_qlinear_conv_with_runner(
        model, op, inputs, input_count, outputs, output_count,
        scratch, scratch_size, qconv->run);
}

#define CAMPP_DEFINE_FUSED_QCONV_CANDIDATE(function_name, qconv_mode)  \
    static CamppStatus function_name(                                  \
        const CamppRuntimeModel *model,                                \
        const CamppOperatorDescriptor *op,                             \
        const CamppTensorView *inputs, uint8_t input_count,            \
        CamppTensorView *outputs, uint8_t output_count,                 \
        void *scratch, size_t scratch_size)                            \
    {                                                                  \
        return campp_fused_qconv_candidate_run(                        \
            (qconv_mode), model, op, inputs, input_count,              \
            outputs, output_count, scratch, scratch_size);             \
    }

CAMPP_DEFINE_FUSED_QCONV_CANDIDATE(
    campp_fused_qconv_candidate_mac, CAMPP_QCONV_CANDIDATE_MAC)
CAMPP_DEFINE_FUSED_QCONV_CANDIDATE(
    campp_fused_qconv_candidate_combined,
    CAMPP_QCONV_CANDIDATE_COMBINED)

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
    }
};

const char *campp_fused_qconv_candidate_mode_name(
    CamppFusedQconvCandidateMode mode)
{
    static const char *const names[] = {
        "baseline", "mac", "combined"
    };
    return mode >= CAMPP_FUSED_QCONV_CANDIDATE_BASELINE &&
        mode <= CAMPP_FUSED_QCONV_CANDIDATE_COMBINED
        ? names[mode] : "invalid";
}

int campp_fused_qconv_candidate_mode_parse(
    const char *text, CamppFusedQconvCandidateMode *out_mode)
{
    CamppFusedQconvCandidateMode mode;
    if (text == NULL || out_mode == NULL) return 1;
    for (mode = CAMPP_FUSED_QCONV_CANDIDATE_BASELINE;
         mode <= CAMPP_FUSED_QCONV_CANDIDATE_COMBINED;
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
        mode > CAMPP_FUSED_QCONV_CANDIDATE_COMBINED) {
        return NULL;
    }
    return &CAMPP_FUSED_QCONV_CANDIDATE_ENTRIES[mode - 1];
}

#undef CAMPP_DEFINE_FUSED_QCONV_CANDIDATE
