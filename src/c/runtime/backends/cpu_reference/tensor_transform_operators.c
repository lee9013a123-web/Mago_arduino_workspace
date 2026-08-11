#include "reference_kernels.h"

#include <limits.h>
#include <stdint.h>
#include <string.h>

#include "internal/runtime_model.h"
#include "reference_kernel_utils.h"

static CamppStatus campp_read_axes(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    uint16_t key, int64_t axes[CAMPP_TENSOR_MAX_RANK], uint8_t *out_count)
{
    return campp_runtime_model_attribute_ints(
        model, op, key, axes, CAMPP_TENSOR_MAX_RANK, out_count);
}

static CamppStatus campp_read_shape_tensor(
    const CamppTensorView *shape, int64_t values[CAMPP_TENSOR_MAX_RANK],
    uint8_t *out_count)
{
    const uint64_t count = campp_tensor_view_element_count(shape);
    uint64_t index;
    CamppStatus status;

    if (shape == NULL || values == NULL || out_count == NULL ||
        shape->dtype != CAMPP_DTYPE_INT64 || count > CAMPP_TENSOR_MAX_RANK) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    for (index = 0u; index < count; ++index) {
        status = campp_reference_read_i64(shape, index, &values[index]);
        if (status != CAMPP_STATUS_OK) {
            return status;
        }
    }
    *out_count = (uint8_t)count;
    return CAMPP_STATUS_OK;
}

static CamppStatus campp_validate_requested_shape(
    const CamppTensorView *input, const CamppTensorView *output,
    const int64_t *requested, uint8_t requested_rank)
{
    uint64_t known_product = 1u;
    uint64_t input_count = campp_tensor_view_element_count(input);
    int8_t inferred_axis = -1;
    uint8_t axis;

    if (requested_rank != output->rank) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    for (axis = 0u; axis < requested_rank; ++axis) {
        uint64_t dimension;

        if (requested[axis] == -1) {
            if (inferred_axis >= 0) {
                return CAMPP_STATUS_SHAPE_MISMATCH;
            }
            inferred_axis = (int8_t)axis;
            continue;
        }
        if (requested[axis] == 0) {
            if (axis >= input->rank) {
                return CAMPP_STATUS_SHAPE_MISMATCH;
            }
            dimension = input->dimensions[axis];
        } else if (requested[axis] > 0) {
            dimension = (uint64_t)requested[axis];
        } else {
            return CAMPP_STATUS_SHAPE_MISMATCH;
        }
        if (dimension != output->dimensions[axis] ||
            known_product > UINT64_MAX / dimension) {
            return CAMPP_STATUS_SHAPE_MISMATCH;
        }
        known_product *= dimension;
    }
    if (inferred_axis >= 0) {
        uint64_t inferred;
        if (known_product == 0u || input_count % known_product != 0u) {
            return CAMPP_STATUS_SHAPE_MISMATCH;
        }
        inferred = input_count / known_product;
        if (inferred != output->dimensions[(uint8_t)inferred_axis]) {
            return CAMPP_STATUS_SHAPE_MISMATCH;
        }
    } else if (known_product != input_count) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    return campp_tensor_view_element_count(output) == input_count
               ? CAMPP_STATUS_OK
               : CAMPP_STATUS_SHAPE_MISMATCH;
}

CamppStatus campp_reference_reshape(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count, void *scratch,
    size_t scratch_size)
{
    int64_t requested[CAMPP_TENSOR_MAX_RANK];
    uint8_t requested_rank;
    CamppStatus status;

    (void)model;
    (void)op;
    (void)scratch;
    (void)scratch_size;
    status = campp_reference_validate_invocation(
        inputs, input_count, 2u, 2u, outputs, output_count);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_read_shape_tensor(&inputs[1], requested, &requested_rank);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_validate_requested_shape(
        &inputs[0], &outputs[0], requested, requested_rank);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    return campp_reference_copy_tensor(&inputs[0], &outputs[0]);
}

CamppStatus campp_reference_squeeze(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count, void *scratch,
    size_t scratch_size)
{
    int64_t axes[CAMPP_TENSOR_MAX_RANK];
    bool removed[CAMPP_TENSOR_MAX_RANK] = {false, false, false, false};
    uint8_t axis_count = 0u;
    uint8_t index;
    uint8_t output_axis = 0u;
    CamppStatus status;

    (void)scratch;
    (void)scratch_size;
    status = campp_reference_validate_invocation(
        inputs, input_count, 1u, 1u, outputs, output_count);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_read_axes(model, op, CAMPP_ATTR_AXES, axes, &axis_count);
    if (status == CAMPP_STATUS_MISSING_ATTRIBUTE) {
        for (index = 0u; index < inputs[0].rank; ++index) {
            removed[index] = inputs[0].dimensions[index] == 1u;
        }
    } else if (status != CAMPP_STATUS_OK) {
        return status;
    } else {
        for (index = 0u; index < axis_count; ++index) {
            uint8_t normalized;
            status = campp_reference_normalize_axis(
                axes[index], inputs[0].rank, &normalized);
            if (status != CAMPP_STATUS_OK || removed[normalized] ||
                inputs[0].dimensions[normalized] != 1u) {
                return CAMPP_STATUS_SHAPE_MISMATCH;
            }
            removed[normalized] = true;
        }
    }
    for (index = 0u; index < inputs[0].rank; ++index) {
        if (!removed[index]) {
            if (output_axis >= outputs[0].rank ||
                outputs[0].dimensions[output_axis] !=
                    inputs[0].dimensions[index]) {
                return CAMPP_STATUS_SHAPE_MISMATCH;
            }
            ++output_axis;
        }
    }
    if (output_axis != outputs[0].rank) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    return campp_reference_copy_tensor(&inputs[0], &outputs[0]);
}

CamppStatus campp_reference_unsqueeze(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count, void *scratch,
    size_t scratch_size)
{
    int64_t axes[CAMPP_TENSOR_MAX_RANK];
    bool inserted[CAMPP_TENSOR_MAX_RANK] = {false, false, false, false};
    uint8_t axis_count;
    uint8_t index;
    uint8_t input_axis = 0u;
    CamppStatus status;

    (void)scratch;
    (void)scratch_size;
    status = campp_reference_validate_invocation(
        inputs, input_count, 1u, 1u, outputs, output_count);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_read_axes(model, op, CAMPP_ATTR_AXES, axes, &axis_count);
    if (status != CAMPP_STATUS_OK ||
        (uint8_t)(inputs[0].rank + axis_count) != outputs[0].rank) {
        return status == CAMPP_STATUS_OK ? CAMPP_STATUS_SHAPE_MISMATCH : status;
    }
    for (index = 0u; index < axis_count; ++index) {
        uint8_t normalized;
        status = campp_reference_normalize_axis(
            axes[index], outputs[0].rank, &normalized);
        if (status != CAMPP_STATUS_OK || inserted[normalized]) {
            return CAMPP_STATUS_SHAPE_MISMATCH;
        }
        inserted[normalized] = true;
    }
    for (index = 0u; index < outputs[0].rank; ++index) {
        if (inserted[index]) {
            if (outputs[0].dimensions[index] != 1u) {
                return CAMPP_STATUS_SHAPE_MISMATCH;
            }
        } else {
            if (input_axis >= inputs[0].rank ||
                outputs[0].dimensions[index] !=
                    inputs[0].dimensions[input_axis]) {
                return CAMPP_STATUS_SHAPE_MISMATCH;
            }
            ++input_axis;
        }
    }
    return campp_reference_copy_tensor(&inputs[0], &outputs[0]);
}

CamppStatus campp_reference_transpose(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count, void *scratch,
    size_t scratch_size)
{
    int64_t permutation[CAMPP_TENSOR_MAX_RANK];
    bool used[CAMPP_TENSOR_MAX_RANK] = {false, false, false, false};
    uint8_t permutation_count;
    uint8_t axis;
    uint64_t count;
    uint64_t index;
    uint32_t element_size;
    CamppStatus status;

    (void)scratch;
    (void)scratch_size;
    status = campp_reference_validate_invocation(
        inputs, input_count, 1u, 1u, outputs, output_count);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    if (inputs[0].dtype != outputs[0].dtype ||
        inputs[0].rank != outputs[0].rank) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    status = campp_read_axes(
        model, op, CAMPP_ATTR_PERM, permutation, &permutation_count);
    if (status != CAMPP_STATUS_OK || permutation_count != inputs[0].rank) {
        return status == CAMPP_STATUS_OK ? CAMPP_STATUS_SHAPE_MISMATCH : status;
    }
    for (axis = 0u; axis < permutation_count; ++axis) {
        if (permutation[axis] < 0 ||
            permutation[axis] >= (int64_t)inputs[0].rank ||
            used[(uint8_t)permutation[axis]] ||
            outputs[0].dimensions[axis] !=
                inputs[0].dimensions[(uint8_t)permutation[axis]]) {
            return CAMPP_STATUS_SHAPE_MISMATCH;
        }
        used[(uint8_t)permutation[axis]] = true;
    }
    element_size = campp_dtype_byte_size(inputs[0].dtype);
    if (element_size == 0u) {
        return CAMPP_STATUS_UNSUPPORTED_DTYPE;
    }
    count = campp_tensor_view_element_count(&outputs[0]);
    for (index = 0u; index < count; ++index) {
        uint32_t output_coordinates[CAMPP_TENSOR_MAX_RANK];
        uint32_t input_coordinates[CAMPP_TENSOR_MAX_RANK] = {0u, 0u, 0u, 0u};
        uint64_t input_offset;
        uint64_t output_offset;

        campp_reference_unravel_index(
            index, outputs[0].rank, outputs[0].dimensions,
            output_coordinates);
        for (axis = 0u; axis < permutation_count; ++axis) {
            input_coordinates[(uint8_t)permutation[axis]] =
                output_coordinates[axis];
        }
        input_offset = campp_tensor_view_byte_offset(
            &inputs[0], input_coordinates);
        output_offset = campp_tensor_view_byte_offset(
            &outputs[0], output_coordinates);
        memcpy(
            (uint8_t *)outputs[0].data + output_offset,
            (const uint8_t *)inputs[0].data + input_offset, element_size);
    }
    return CAMPP_STATUS_OK;
}

CamppStatus campp_reference_concat(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count, void *scratch,
    size_t scratch_size)
{
    int64_t axis_value[1];
    uint8_t attribute_count;
    uint8_t axis;
    uint8_t input_index;
    uint64_t concatenated = 0u;
    uint64_t count;
    uint64_t index;
    uint32_t element_size;
    CamppStatus status;

    (void)scratch;
    (void)scratch_size;
    status = campp_reference_validate_invocation(
        inputs, input_count, 1u, CAMPP_OPERATOR_INPUT_CAPACITY,
        outputs, output_count);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_runtime_model_attribute_ints(
        model, op, CAMPP_ATTR_AXIS, axis_value, 1u, &attribute_count);
    if (status != CAMPP_STATUS_OK || attribute_count != 1u) {
        return status == CAMPP_STATUS_OK ? CAMPP_STATUS_CORRUPT_PLAN : status;
    }
    status = campp_reference_normalize_axis(
        axis_value[0], outputs[0].rank, &axis);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    for (input_index = 0u; input_index < input_count; ++input_index) {
        uint8_t dimension_axis;
        if (inputs[input_index].dtype != outputs[0].dtype ||
            inputs[input_index].rank != outputs[0].rank) {
            return CAMPP_STATUS_SHAPE_MISMATCH;
        }
        for (dimension_axis = 0u; dimension_axis < outputs[0].rank;
             ++dimension_axis) {
            if (dimension_axis != axis &&
                inputs[input_index].dimensions[dimension_axis] !=
                    outputs[0].dimensions[dimension_axis]) {
                return CAMPP_STATUS_SHAPE_MISMATCH;
            }
        }
        concatenated += inputs[input_index].dimensions[axis];
    }
    if (concatenated != outputs[0].dimensions[axis]) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    element_size = campp_dtype_byte_size(outputs[0].dtype);
    if (element_size == 0u) {
        return CAMPP_STATUS_UNSUPPORTED_DTYPE;
    }
    count = campp_tensor_view_element_count(&outputs[0]);
    for (index = 0u; index < count; ++index) {
        uint32_t coordinates[CAMPP_TENSOR_MAX_RANK];
        uint32_t source_coordinates[CAMPP_TENSOR_MAX_RANK];
        uint32_t position;
        uint32_t consumed = 0u;
        const CamppTensorView *source = NULL;
        uint64_t source_offset;
        uint64_t output_offset;

        campp_reference_unravel_index(
            index, outputs[0].rank, outputs[0].dimensions, coordinates);
        position = coordinates[axis];
        for (input_index = 0u; input_index < input_count; ++input_index) {
            const uint32_t width = inputs[input_index].dimensions[axis];
            if (position < consumed + width) {
                source = &inputs[input_index];
                break;
            }
            consumed += width;
        }
        if (source == NULL) {
            return CAMPP_STATUS_SHAPE_MISMATCH;
        }
        memcpy(source_coordinates, coordinates, sizeof(source_coordinates));
        source_coordinates[axis] = position - consumed;
        source_offset = campp_tensor_view_byte_offset(
            source, source_coordinates);
        output_offset = campp_tensor_view_byte_offset(
            &outputs[0], coordinates);
        memcpy(
            (uint8_t *)outputs[0].data + output_offset,
            (const uint8_t *)source->data + source_offset, element_size);
    }
    return CAMPP_STATUS_OK;
}

CamppStatus campp_reference_expand(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count, void *scratch,
    size_t scratch_size)
{
    int64_t requested[CAMPP_TENSOR_MAX_RANK];
    uint8_t requested_rank;
    uint8_t input_axis;
    const CamppTensorView *input;
    CamppTensorView *output;
    uint32_t element_size;
    uint64_t count;
    uint64_t index;
    CamppStatus status;

    (void)model;
    (void)op;
    (void)scratch;
    (void)scratch_size;
    status = campp_reference_validate_invocation(
        inputs, input_count, 2u, 2u, outputs, output_count);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    input = &inputs[0];
    output = &outputs[0];
    status = campp_read_shape_tensor(&inputs[1], requested, &requested_rank);
    if (status != CAMPP_STATUS_OK || requested_rank != output->rank ||
        input->rank > output->rank || input->dtype != output->dtype) {
        return status == CAMPP_STATUS_OK ? CAMPP_STATUS_SHAPE_MISMATCH : status;
    }
    for (input_axis = 0u; input_axis < output->rank; ++input_axis) {
        if (requested[input_axis] <= 0 ||
            (uint64_t)requested[input_axis] != output->dimensions[input_axis]) {
            return CAMPP_STATUS_SHAPE_MISMATCH;
        }
    }
    for (input_axis = 0u; input_axis < input->rank; ++input_axis) {
        const uint8_t output_axis =
            (uint8_t)(output->rank - input->rank + input_axis);
        if (input->dimensions[input_axis] != 1u &&
            input->dimensions[input_axis] != output->dimensions[output_axis]) {
            return CAMPP_STATUS_SHAPE_MISMATCH;
        }
    }
    element_size = campp_dtype_byte_size(input->dtype);
    if (element_size == 0u) {
        return CAMPP_STATUS_UNSUPPORTED_DTYPE;
    }
    count = campp_tensor_view_element_count(output);
    for (index = 0u; index < count; ++index) {
        uint32_t output_coordinates[CAMPP_TENSOR_MAX_RANK];
        uint32_t input_coordinates[CAMPP_TENSOR_MAX_RANK] = {0u, 0u, 0u, 0u};
        uint64_t input_offset;
        uint64_t output_offset;

        campp_reference_unravel_index(
            index, output->rank, output->dimensions, output_coordinates);
        for (input_axis = 0u; input_axis < input->rank; ++input_axis) {
            const uint8_t output_axis =
                (uint8_t)(output->rank - input->rank + input_axis);
            input_coordinates[input_axis] =
                input->dimensions[input_axis] == 1u
                    ? 0u
                    : output_coordinates[output_axis];
        }
        input_offset = campp_tensor_view_byte_offset(input, input_coordinates);
        output_offset = campp_tensor_view_byte_offset(output, output_coordinates);
        memcpy(
            (uint8_t *)output->data + output_offset,
            (const uint8_t *)input->data + input_offset, element_size);
    }
    return CAMPP_STATUS_OK;
}

static int64_t campp_slice_normalize_index(
    int64_t value, int64_t dimension, int64_t step, bool is_start)
{
    if (step > 0) {
        if (value == INT64_MIN) {
            return 0;
        }
        if (value == INT64_MAX) {
            return dimension;
        }
        if (value < 0) {
            value += dimension;
        }
        if (value < 0) {
            return 0;
        }
        return value > dimension ? dimension : value;
    }
    if (value == INT64_MIN) {
        return -1;
    }
    if (value == INT64_MAX) {
        return dimension - 1;
    }
    if (value < 0) {
        value += dimension;
    }
    if (value < -1) {
        return -1;
    }
    if (value >= dimension) {
        return dimension - 1;
    }
    (void)is_start;
    return value;
}

CamppStatus campp_reference_slice(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count, void *scratch,
    size_t scratch_size)
{
    int64_t starts[CAMPP_TENSOR_MAX_RANK] = {0, 0, 0, 0};
    int64_t ends[CAMPP_TENSOR_MAX_RANK] = {0, 0, 0, 0};
    int64_t axes[CAMPP_TENSOR_MAX_RANK] = {0, 1, 2, 3};
    int64_t steps[CAMPP_TENSOR_MAX_RANK] = {1, 1, 1, 1};
    int64_t normalized_starts[CAMPP_TENSOR_MAX_RANK] = {0, 0, 0, 0};
    int64_t normalized_steps[CAMPP_TENSOR_MAX_RANK] = {1, 1, 1, 1};
    bool assigned[CAMPP_TENSOR_MAX_RANK] = {false, false, false, false};
    uint64_t parameter_count;
    uint64_t index;
    uint8_t axis;
    uint32_t element_size;
    CamppStatus status;

    (void)model;
    (void)op;
    (void)scratch;
    (void)scratch_size;
    status = campp_reference_validate_invocation(
        inputs, input_count, 3u, 5u, outputs, output_count);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    if (inputs[0].dtype != outputs[0].dtype ||
        inputs[0].rank != outputs[0].rank) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    parameter_count = campp_tensor_view_element_count(&inputs[1]);
    if (parameter_count == 0u || parameter_count > CAMPP_TENSOR_MAX_RANK ||
        campp_tensor_view_element_count(&inputs[2]) != parameter_count ||
        (input_count >= 4u &&
         campp_tensor_view_element_count(&inputs[3]) != parameter_count) ||
        (input_count >= 5u &&
         campp_tensor_view_element_count(&inputs[4]) != parameter_count)) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    for (index = 0u; index < parameter_count; ++index) {
        status = campp_reference_read_i64(&inputs[1], index, &starts[index]);
        if (status != CAMPP_STATUS_OK) return status;
        status = campp_reference_read_i64(&inputs[2], index, &ends[index]);
        if (status != CAMPP_STATUS_OK) return status;
        if (input_count >= 4u) {
            status = campp_reference_read_i64(&inputs[3], index, &axes[index]);
            if (status != CAMPP_STATUS_OK) return status;
        }
        if (input_count >= 5u) {
            status = campp_reference_read_i64(&inputs[4], index, &steps[index]);
            if (status != CAMPP_STATUS_OK) return status;
        }
    }
    for (axis = 0u; axis < inputs[0].rank; ++axis) {
        normalized_starts[axis] = 0;
        normalized_steps[axis] = 1;
    }
    for (index = 0u; index < parameter_count; ++index) {
        uint8_t normalized_axis;
        int64_t start;
        int64_t end;
        uint64_t expected_length;
        int64_t actual_dimension;

        status = campp_reference_normalize_axis(
            axes[index], inputs[0].rank, &normalized_axis);
        if (status != CAMPP_STATUS_OK || assigned[normalized_axis] ||
            steps[index] == 0 || steps[index] == INT64_MIN) {
            return CAMPP_STATUS_SHAPE_MISMATCH;
        }
        actual_dimension = inputs[0].dimensions[normalized_axis];
        start = campp_slice_normalize_index(
            starts[index], actual_dimension, steps[index], true);
        end = campp_slice_normalize_index(
            ends[index], actual_dimension, steps[index], false);
        if (steps[index] > 0 && start < end) {
            const uint64_t distance = (uint64_t)(end - start);
            const uint64_t step = (uint64_t)steps[index];
            expected_length = (distance + step - 1u) / step;
        } else if (steps[index] < 0 && start > end) {
            const uint64_t distance = (uint64_t)(start - end);
            const uint64_t magnitude = (uint64_t)(-steps[index]);
            expected_length = (distance + magnitude - 1u) / magnitude;
        } else {
            expected_length = 0u;
        }
        if (expected_length != outputs[0].dimensions[normalized_axis]) {
            return CAMPP_STATUS_SHAPE_MISMATCH;
        }
        normalized_starts[normalized_axis] = start;
        normalized_steps[normalized_axis] = steps[index];
        assigned[normalized_axis] = true;
    }
    for (axis = 0u; axis < inputs[0].rank; ++axis) {
        if (!assigned[axis] &&
            outputs[0].dimensions[axis] != inputs[0].dimensions[axis]) {
            return CAMPP_STATUS_SHAPE_MISMATCH;
        }
    }
    element_size = campp_dtype_byte_size(inputs[0].dtype);
    if (element_size == 0u) {
        return CAMPP_STATUS_UNSUPPORTED_DTYPE;
    }
    for (index = 0u; index < campp_tensor_view_element_count(&outputs[0]);
         ++index) {
        uint32_t output_coordinates[CAMPP_TENSOR_MAX_RANK];
        uint32_t input_coordinates[CAMPP_TENSOR_MAX_RANK];
        uint64_t input_offset;
        uint64_t output_offset;

        campp_reference_unravel_index(
            index, outputs[0].rank, outputs[0].dimensions,
            output_coordinates);
        for (axis = 0u; axis < inputs[0].rank; ++axis) {
            const int64_t coordinate =
                normalized_starts[axis] +
                (int64_t)output_coordinates[axis] * normalized_steps[axis];
            if (coordinate < 0 ||
                coordinate >= (int64_t)inputs[0].dimensions[axis]) {
                return CAMPP_STATUS_SHAPE_MISMATCH;
            }
            input_coordinates[axis] = (uint32_t)coordinate;
        }
        input_offset = campp_tensor_view_byte_offset(
            &inputs[0], input_coordinates);
        output_offset = campp_tensor_view_byte_offset(
            &outputs[0], output_coordinates);
        memcpy(
            (uint8_t *)outputs[0].data + output_offset,
            (const uint8_t *)inputs[0].data + input_offset, element_size);
    }
    return CAMPP_STATUS_OK;
}
