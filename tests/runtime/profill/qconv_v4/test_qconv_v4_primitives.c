#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "campp_runtime/tensor_descriptor.h"
#include "planning/qconv_v4_tile_plan.h"
#include "qconv_address_fastpath.h"
#include "requant/qconv_requant_neon8.h"
#include "store/qconv_store_channel_packed.h"

#define CHECK_TRUE(expression)                                        \
    do {                                                              \
        if (!(expression)) {                                          \
            fprintf(stderr, "CHECK failed at line %d: %s\n",       \
                    __LINE__, #expression);                           \
            return 1;                                                 \
        }                                                             \
    } while (0)

static void init_channel_packed_view(
    CamppTensorView *view, void *data, uint8_t dtype, uint8_t rank,
    const uint32_t *dimensions, uint32_t channel_capacity)
{
    uint32_t spatial_stride = channel_capacity;
    uint8_t axis;
    memset(view, 0, sizeof(*view));
    view->data = data;
    view->dtype = dtype;
    view->rank = rank;
    for (axis = 0u; axis < rank; ++axis) {
        view->dimensions[axis] = dimensions[axis];
    }
    view->byte_strides[1] = 1u;
    for (axis = rank; axis > 2u; --axis) {
        view->byte_strides[axis - 1u] = spatial_stride;
        spatial_stride *= dimensions[axis - 1u];
    }
    view->byte_strides[0] = spatial_stride;
    view->logical_byte_size =
        (uint64_t)dimensions[0] * dimensions[1];
    for (axis = 2u; axis < rank; ++axis) {
        view->logical_byte_size *= dimensions[axis];
    }
    view->storage_span_bytes =
        (uint64_t)dimensions[0] * spatial_stride;
}

static int reference_requantize(
    int32_t accumulator, float multiplier, uint8_t dtype,
    int32_t output_zero)
{
    const int64_t minimum = dtype == CAMPP_DTYPE_UINT8 ? 0 : -128;
    const int64_t maximum = dtype == CAMPP_DTYPE_UINT8 ? 255 : 127;
    const float scaled = (float)accumulator * multiplier;
    int64_t rounded;
    if (scaled <= -2147483648.0f) {
        rounded = INT32_MIN;
    } else if (scaled >= 2147483520.0f) {
        rounded = INT32_MAX;
    } else {
        rounded = (int64_t)nearbyintf(scaled);
    }
    rounded += output_zero;
    if (rounded < minimum) rounded = minimum;
    if (rounded > maximum) rounded = maximum;
    return (int)rounded;
}

static int test_requantize8(uint8_t dtype, int32_t output_zero)
{
    const int32_t accumulators[8] = {
        INT32_MIN, -255, -5, -1, 1, 5, 255, INT32_MAX
    };
    const float multipliers[8] = {
        0.0001f, 0.5f, 0.5f, 1.5f, 1.5f, 0.5f, 0.5f, 0.0001f
    };
    uint8_t bytes[8];
    uint32_t lane;
    CHECK_TRUE(campp_qconv_v4_requantize8(
        accumulators, multipliers, dtype, output_zero, bytes) == 0);
    for (lane = 0u; lane < 8u; ++lane) {
        const int expected = reference_requantize(
            accumulators[lane], multipliers[lane], dtype, output_zero);
        const int actual = dtype == CAMPP_DTYPE_UINT8
            ? bytes[lane] : (int)(int8_t)bytes[lane];
        CHECK_TRUE(actual == expected);
    }
    return 0;
}

static int test_1x1_tile_plan(void)
{
    uint8_t input_data[128] = {0};
    uint8_t output_data[128] = {0};
    const uint32_t dimensions[3] = {1u, 8u, 16u};
    CamppTensorView input;
    CamppTensorView output;
    CamppQconvV4ExecutionPlan plan;
    CamppQconvV4TilePlan tile;
    int64_t strides[2] = {1, 1};
    int64_t dilations[2] = {1, 1};
    int64_t pads[4] = {0, 0, 0, 0};

    init_channel_packed_view(
        &input, input_data, CAMPP_DTYPE_UINT8, 3u, dimensions, 8u);
    init_channel_packed_view(
        &output, output_data, CAMPP_DTYPE_UINT8, 3u, dimensions, 8u);
    memset(&plan, 0, sizeof(plan));
    plan.input = &input;
    plan.output = &output;
    plan.spatial_rank = 1u;
    plan.inputs_per_group = 8u;
    plan.input_channels = 8u;
    plan.output_spatial = 16u;
    plan.kernel_elements = 1u;
    plan.kernel_shape[0] = 1;
    plan.strides[0] = 1;
    plan.dilations[0] = 1;
    plan.preferred_path = CAMPP_QCONV_V4_PATH_1X1;
    CHECK_TRUE(campp_qconv_address_plan_create(
        &input, &output, 1u, strides, dilations, pads,
        &plan.address));
    CHECK_TRUE(campp_qconv_v4_tile_plan_create(
        &plan, 0u, 0u, 0u, 8u, &tile) == CAMPP_STATUS_OK);
    CHECK_TRUE(tile.path == CAMPP_QCONV_V4_PATH_1X1);
    CHECK_TRUE(tile.input_points[0][0] == input_data);
    CHECK_TRUE(tile.input_points[0][7] == input_data + 56u);
    CHECK_TRUE(campp_qconv_v4_tile_plan_create(
        &plan, 0u, 0u, 8u, 4u, &tile) == CAMPP_STATUS_OK);
    CHECK_TRUE(tile.path == CAMPP_QCONV_V4_PATH_GENERIC);
    return 0;
}

static int test_3x3_tile_plan(void)
{
    uint8_t input_data[1u * 8u * 5u * 12u] = {0};
    uint8_t output_data[1u * 8u * 5u * 12u] = {0};
    const uint32_t dimensions[4] = {1u, 8u, 5u, 12u};
    CamppTensorView input;
    CamppTensorView output;
    CamppQconvV4ExecutionPlan plan;
    CamppQconvV4TilePlan tile;
    int64_t strides[2] = {1, 1};
    int64_t dilations[2] = {1, 1};
    int64_t pads[4] = {1, 1, 1, 1};

    init_channel_packed_view(
        &input, input_data, CAMPP_DTYPE_UINT8, 4u, dimensions, 8u);
    init_channel_packed_view(
        &output, output_data, CAMPP_DTYPE_UINT8, 4u, dimensions, 8u);
    memset(&plan, 0, sizeof(plan));
    plan.input = &input;
    plan.output = &output;
    plan.spatial_rank = 2u;
    plan.inputs_per_group = 8u;
    plan.input_channels = 8u;
    plan.output_spatial = 60u;
    plan.kernel_elements = 9u;
    plan.kernel_shape[0] = 3;
    plan.kernel_shape[1] = 3;
    plan.strides[0] = 1;
    plan.strides[1] = 1;
    plan.dilations[0] = 1;
    plan.dilations[1] = 1;
    plan.pads[0] = 1;
    plan.pads[1] = 1;
    plan.preferred_path = CAMPP_QCONV_V4_PATH_3X3_INTERIOR;
    CHECK_TRUE(campp_qconv_address_plan_create(
        &input, &output, 2u, strides, dilations, pads,
        &plan.address));
    CHECK_TRUE(campp_qconv_v4_tile_plan_create(
        &plan, 0u, 0u, 13u, 8u, &tile) == CAMPP_STATUS_OK);
    CHECK_TRUE(tile.path == CAMPP_QCONV_V4_PATH_3X3_INTERIOR);
    CHECK_TRUE(tile.input_points[0][0] == input_data);
    CHECK_TRUE(tile.input_points[8][7] == input_data + (2u * 12u + 9u) * 8u);
    CHECK_TRUE(campp_qconv_v4_tile_plan_create(
        &plan, 0u, 0u, 0u, 8u, &tile) == CAMPP_STATUS_OK);
    CHECK_TRUE(tile.path == CAMPP_QCONV_V4_PATH_GENERIC);
    return 0;
}

static int test_channel_store(void)
{
    uint8_t output_data[32];
    const uint32_t dimensions[3] = {1u, 10u, 2u};
    const uint8_t bytes[8] = {11u, 12u, 13u, 14u, 15u, 16u, 17u, 18u};
    CamppTensorView output;
    CamppQconvV4StorePlan store_plan;

    memset(output_data, 0xa5, sizeof(output_data));
    init_channel_packed_view(
        &output, output_data, CAMPP_DTYPE_UINT8, 3u, dimensions, 16u);
    CHECK_TRUE(campp_qconv_v4_store_plan_create(
        &output, &store_plan) == CAMPP_STATUS_OK);
    CHECK_TRUE(campp_qconv_v4_store_channel_packed(
        &store_plan, 0u, 0u, 0u, bytes, 8u) == CAMPP_STATUS_OK);
    CHECK_TRUE(memcmp(output_data, bytes, 8u) == 0);
    CHECK_TRUE(campp_qconv_v4_store_channel_packed(
        &store_plan, 0u, 8u, 0u, bytes, 2u) == CAMPP_STATUS_OK);
    CHECK_TRUE(memcmp(output_data + 8u, bytes, 8u) == 0);

    memset(output_data, 0xa5, sizeof(output_data));
    output.flags |= CAMPP_TENSOR_FLAG_ALIASED;
    CHECK_TRUE(campp_qconv_v4_store_plan_create(
        &output, &store_plan) == CAMPP_STATUS_OK);
    CHECK_TRUE(campp_qconv_v4_store_channel_packed(
        &store_plan, 0u, 8u, 0u, bytes, 2u) == CAMPP_STATUS_OK);
    CHECK_TRUE(memcmp(output_data + 8u, bytes, 2u) == 0);
    CHECK_TRUE(output_data[10] == 0xa5);
    return 0;
}

int main(void)
{
    CHECK_TRUE(test_requantize8(CAMPP_DTYPE_UINT8, 127) == 0);
    CHECK_TRUE(test_requantize8(CAMPP_DTYPE_INT8, -3) == 0);
    CHECK_TRUE(test_1x1_tile_plan() == 0);
    CHECK_TRUE(test_3x3_tile_plan() == 0);
    CHECK_TRUE(test_channel_store() == 0);
    puts("QConv v4 primitives: PASS");
    return 0;
}
