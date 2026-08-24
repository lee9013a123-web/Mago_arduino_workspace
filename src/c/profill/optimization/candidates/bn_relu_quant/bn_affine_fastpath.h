#ifndef CAMPP_PROFILL_BN_AFFINE_FASTPATH_H
#define CAMPP_PROFILL_BN_AFFINE_FASTPATH_H

#include <stdint.h>

#include "campp_runtime/status_code.h"
#include "internal/tensor_view.h"

typedef struct CamppBnAffine {
    float multiplier;
    float additive;
} CamppBnAffine;

CamppStatus campp_bn_affine_channel(
    const CamppTensorView *scale, const CamppTensorView *bias,
    const CamppTensorView *mean, const CamppTensorView *variance,
    uint32_t channel, float epsilon, CamppBnAffine *out_affine);

#endif /* CAMPP_PROFILL_BN_AFFINE_FASTPATH_H */
