#ifndef CAMPP_PROFILL_QCONV_V4_STORE_CHANNEL_PACKED_H
#define CAMPP_PROFILL_QCONV_V4_STORE_CHANNEL_PACKED_H

#include <stdbool.h>
#include <stdint.h>

#include "internal/kernel_registry.h"
#include "qconv_mac_neon.h"

typedef struct CamppQconvV4StorePlan {
    uint8_t *data;
    uint64_t storage_span_bytes;
    uint32_t batch_stride;
    uint32_t row_stride;
    uint32_t spatial_stride;
    uint32_t channel_capacity;
    uint32_t batches;
    uint32_t output_channels;
    uint32_t spatial_count;
    uint32_t width;
    uint8_t rank;
    bool padded_tail_store;
} CamppQconvV4StorePlan;

CamppStatus campp_qconv_v4_store_plan_create(
    CamppTensorView *output, CamppQconvV4StorePlan *out_plan);

CamppStatus campp_qconv_v4_store_channel_packed(
    const CamppQconvV4StorePlan *plan,
    uint32_t batch, uint32_t output_channel,
    uint32_t spatial_index,
    const uint8_t bytes[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    uint32_t valid_outputs);

#endif /* CAMPP_PROFILL_QCONV_V4_STORE_CHANNEL_PACKED_H */
