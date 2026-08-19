#include "qconv_v5_dispatch.h"

#include "qconv_v4_dispatch.h"

CamppStatus campp_qconv_v5_run(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count,
    void *scratch, size_t scratch_size)
{
    /*
     * v5 starts as an independently selectable, bitwise-equivalent staging
     * candidate.  MAC-side kernels are added below this dispatch boundary and
     * can replace the delegated v4 path one gate at a time.
     */
    return campp_qconv_v4_run(
        model, op, inputs, input_count, outputs, output_count,
        scratch, scratch_size);
}
