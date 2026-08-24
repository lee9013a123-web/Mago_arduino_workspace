#ifndef CAMPP_PROFILL_FUSED_INPUT_QUANT_NEON_H
#define CAMPP_PROFILL_FUSED_INPUT_QUANT_NEON_H

#include "campp_runtime/status_code.h"
#include "internal/tensor_view.h"

CamppStatus campp_fused_input_quantize_neon(
    const CamppTensorView *input, CamppTensorView *output,
    float scale, int32_t zero_point);

#endif /* CAMPP_PROFILL_FUSED_INPUT_QUANT_NEON_H */
