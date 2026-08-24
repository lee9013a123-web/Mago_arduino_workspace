#ifndef CAMPP_PROFILL_PACKED_ITERATION_H
#define CAMPP_PROFILL_PACKED_ITERATION_H

#include <stdbool.h>
#include <stdint.h>

#include "internal/tensor_view.h"

typedef struct CamppPackedIteration {
    const CamppTensorView *view;
    uint32_t batches;
    uint32_t channels;
    uint32_t height;
    uint32_t width;
    uint32_t strides[4];
    uint32_t element_size;
} CamppPackedIteration;

bool campp_packed_iteration_create(
    const CamppTensorView *view, CamppPackedIteration *out_plan);

bool campp_packed_iteration_same_shape(
    const CamppPackedIteration *left, const CamppPackedIteration *right);

uint64_t campp_packed_iteration_offset(
    const CamppPackedIteration *plan, uint32_t batch, uint32_t channel,
    uint32_t height, uint32_t width);

const uint8_t *campp_packed_iteration_const_pointer(
    const CamppPackedIteration *plan, uint32_t batch, uint32_t channel,
    uint32_t height, uint32_t width);

uint8_t *campp_packed_iteration_pointer(
    const CamppPackedIteration *plan, uint32_t batch, uint32_t channel,
    uint32_t height, uint32_t width);

#endif /* CAMPP_PROFILL_PACKED_ITERATION_H */
