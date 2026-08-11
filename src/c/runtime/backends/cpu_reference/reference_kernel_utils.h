#ifndef CAMPP_RUNTIME_CPU_REFERENCE_KERNEL_UTILS_H
#define CAMPP_RUNTIME_CPU_REFERENCE_KERNEL_UTILS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "campp_runtime/status_code.h"
#include "internal/tensor_view.h"

CamppStatus campp_reference_validate_invocation(
    const CamppTensorView *inputs, uint8_t input_count,
    uint8_t minimum_inputs, uint8_t maximum_inputs,
    CamppTensorView *outputs, uint8_t output_count);

CamppStatus campp_reference_normalize_axis(
    int64_t axis, uint8_t rank, uint8_t *out_axis);

void campp_reference_unravel_index(
    uint64_t linear_index, uint8_t rank,
    const uint32_t dimensions[CAMPP_TENSOR_MAX_RANK],
    uint32_t coordinates[CAMPP_TENSOR_MAX_RANK]);

uint64_t campp_reference_offset_for_linear(
    const CamppTensorView *view, uint64_t linear_index);

CamppStatus campp_reference_copy_tensor(
    const CamppTensorView *input, CamppTensorView *output);

bool campp_reference_shapes_equal(
    const CamppTensorView *first, const CamppTensorView *second);

CamppStatus campp_reference_read_f32(
    const CamppTensorView *view, uint64_t linear_index, float *out_value);

CamppStatus campp_reference_read_i64(
    const CamppTensorView *view, uint64_t linear_index, int64_t *out_value);

CamppStatus campp_reference_read_quantized(
    const CamppTensorView *view, uint64_t linear_index, int32_t *out_value);

CamppStatus campp_reference_write_quantized(
    CamppTensorView *view, uint64_t linear_index, int64_t value);

#endif /* CAMPP_RUNTIME_CPU_REFERENCE_KERNEL_UTILS_H */
