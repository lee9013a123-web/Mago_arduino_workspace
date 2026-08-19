#include "qconv_hybrid_dispatch.h"

#include "qconv_candidate.h"
#include "qconv_v4_dispatch.h"

CamppQconvHybridPath campp_qconv_hybrid_select_path(
    const CamppTensorView *inputs, uint8_t input_count)
{
    const CamppTensorView *weight;
    if (inputs == NULL || input_count < 8u) {
        return CAMPP_QCONV_HYBRID_PATH_MAC_FIXED;
    }
    weight = &inputs[3];

    /* Bucket-98 measured V4 winners. Unmeasured shapes stay on MAC Fixed. */
    if (weight->rank == 4u && weight->dimensions[0] == 32u &&
        weight->dimensions[1] == 32u && weight->dimensions[2] == 3u &&
        weight->dimensions[3] == 3u) {
        return CAMPP_QCONV_HYBRID_PATH_V4;
    }
    if (weight->rank == 3u && weight->dimensions[2] == 1u &&
        ((weight->dimensions[0] == 64u &&
          weight->dimensions[1] == 128u) ||
         (weight->dimensions[0] == 192u &&
          weight->dimensions[1] == 1024u))) {
        return CAMPP_QCONV_HYBRID_PATH_V4;
    }
    return CAMPP_QCONV_HYBRID_PATH_MAC_FIXED;
}

CamppStatus campp_qconv_hybrid_run(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count,
    void *scratch, size_t scratch_size)
{
    if (campp_qconv_hybrid_select_path(inputs, input_count) ==
        CAMPP_QCONV_HYBRID_PATH_V4) {
        return campp_qconv_v4_run(
            model, op, inputs, input_count, outputs, output_count,
            scratch, scratch_size);
    }
    {
        const CamppKernelEntry *fixed = campp_qconv_candidate_entry(
            CAMPP_QCONV_CANDIDATE_MAC_FIXED);
        if (fixed == NULL || fixed->run == NULL) {
            return CAMPP_STATUS_MISSING_KERNEL;
        }
        return fixed->run(
            model, op, inputs, input_count, outputs, output_count,
            scratch, scratch_size);
    }
}
