#ifndef CAMPP_PROFILL_QCONV_ADDRESS_FASTPATH_H
#define CAMPP_PROFILL_QCONV_ADDRESS_FASTPATH_H

#include <stdbool.h>
#include <stdint.h>

#include "internal/tensor_view.h"

#define CAMPP_QCONV_CANDIDATE_TILE 8u

typedef struct CamppQconvAddressPlan {
    const CamppTensorView *input;
    const CamppTensorView *output;
    uint8_t spatial_rank;
    int64_t strides[2];
    int64_t dilations[2];
    int64_t pads[2];
} CamppQconvAddressPlan;

bool campp_qconv_address_plan_create(
    const CamppTensorView *input, const CamppTensorView *output,
    uint8_t spatial_rank, const int64_t strides[2],
    const int64_t dilations[2], const int64_t pads[4],
    CamppQconvAddressPlan *out_plan);

void campp_qconv_address_output_tile(
    const CamppQconvAddressPlan *plan, uint32_t tile_start,
    uint32_t tile_count,
    uint32_t coordinates[CAMPP_QCONV_CANDIDATE_TILE][2]);

const uint8_t *campp_qconv_address_input_base(
    const CamppQconvAddressPlan *plan, uint32_t batch,
    uint32_t group_channel, const uint32_t output_coordinates[2],
    const uint32_t kernel_coordinates[2], bool direct_offset);

#endif /* CAMPP_PROFILL_QCONV_ADDRESS_FASTPATH_H */
