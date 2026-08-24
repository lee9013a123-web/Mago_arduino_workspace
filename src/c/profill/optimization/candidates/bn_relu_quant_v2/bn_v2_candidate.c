#include "bn_v2_candidate.h"

#include <stdbool.h>

#include "backends/cpu_aarch64/aarch64_kernels.h"

#if defined(CAMPP_ENABLE_OPTIMIZATION_DIAGNOSTICS)
#include "campp_profill/optimization/stage_probe.h"
#else
#define CAMPP_OPTIMIZATION_STAGE_BEGIN(variable) ((void)0)
#define CAMPP_OPTIMIZATION_STAGE_END(stage, variable) ((void)0)
#endif

static CamppStatus campp_bn_v2_run(
    CamppBnV2Mode mode, const CamppRuntimeModel *model,
    const CamppOperatorDescriptor *op, const CamppTensorView *inputs,
    uint8_t input_count, CamppTensorView *outputs, uint8_t output_count,
    void *scratch, size_t scratch_size)
{
    CamppBnV2ExecutionPlan plan;
    CamppBnV2Parameters parameters;
    CamppStatus status;

    CAMPP_OPTIMIZATION_STAGE_BEGIN(setup_started_ns);
    status = campp_bn_v2_execution_plan_create(
        model, op, inputs, input_count, outputs, output_count, &plan);
    if (status == CAMPP_STATUS_OK) {
        status = campp_bn_v2_parameters_prepare(
            &plan, mode == CAMPP_BN_V2_PRESCALED, &parameters);
    }
    CAMPP_OPTIMIZATION_STAGE_END(
        CAMPP_OPT_STAGE_BN_SETUP, setup_started_ns);
    if (status == CAMPP_STATUS_NOT_IMPLEMENTED) {
        return campp_fused_bn_relu_quant(
            model, op, inputs, input_count, outputs, output_count,
            scratch, scratch_size);
    }
    if (status != CAMPP_STATUS_OK) return status;
    CAMPP_OPTIMIZATION_STAGE_BEGIN(elementwise_started_ns);
    status = campp_bn_v2_dispatch(mode, &plan, &parameters);
    CAMPP_OPTIMIZATION_STAGE_END(
        CAMPP_OPT_STAGE_BN_ELEMENTWISE, elementwise_started_ns);
    return status;
}

#define CAMPP_DEFINE_BN_V2_CANDIDATE(function_name, candidate_mode)    \
    static CamppStatus function_name(                                  \
        const CamppRuntimeModel *model, const CamppOperatorDescriptor *op, \
        const CamppTensorView *inputs, uint8_t input_count,            \
        CamppTensorView *outputs, uint8_t output_count,                \
        void *scratch, size_t scratch_size)                            \
    {                                                                  \
        return campp_bn_v2_run(                                        \
            candidate_mode, model, op, inputs, input_count, outputs,  \
            output_count, scratch, scratch_size);                     \
    }

CAMPP_DEFINE_BN_V2_CANDIDATE(
    campp_bn_v2_exact16, CAMPP_BN_V2_EXACT16)
CAMPP_DEFINE_BN_V2_CANDIDATE(
    campp_bn_v2_spatial2, CAMPP_BN_V2_SPATIAL2)
CAMPP_DEFINE_BN_V2_CANDIDATE(
    campp_bn_v2_prescaled, CAMPP_BN_V2_PRESCALED)

static const CamppKernelEntry CAMPP_BN_V2_ENTRIES[] = {
    {CAMPP_OP_BATCH_NORMALIZATION, CAMPP_FUSION_BN_RELU_QUANT_KERNEL_ID,
     campp_bn_v2_exact16, NULL, "fused_bn_relu_quant_v2_exact16"},
    {CAMPP_OP_BATCH_NORMALIZATION, CAMPP_FUSION_BN_RELU_QUANT_KERNEL_ID,
     campp_bn_v2_spatial2, NULL, "fused_bn_relu_quant_v2_spatial2"},
    {CAMPP_OP_BATCH_NORMALIZATION, CAMPP_FUSION_BN_RELU_QUANT_KERNEL_ID,
     campp_bn_v2_prescaled, NULL, "fused_bn_relu_quant_v2_prescaled"}
};

const CamppKernelEntry *campp_bn_v2_candidate_entry(CamppBnV2Mode mode)
{
    if (mode < CAMPP_BN_V2_EXACT16 || mode > CAMPP_BN_V2_PRESCALED) {
        return NULL;
    }
    return &CAMPP_BN_V2_ENTRIES[mode];
}

#undef CAMPP_DEFINE_BN_V2_CANDIDATE
