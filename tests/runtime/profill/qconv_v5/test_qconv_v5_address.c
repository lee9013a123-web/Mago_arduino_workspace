#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "address/qconv_v5_address.h"
#include "planning/qconv_v5_execution_plan.h"

#define CHECK_TRUE(expression)                                      \
    do {                                                            \
        if (!(expression)) {                                        \
            fprintf(stderr, "CHECK failed at line %d: %s\n",     \
                    __LINE__, #expression);                         \
            return 1;                                               \
        }                                                           \
    } while (0)

static CamppQconvV5AddressGeometry geometry_1d(
    const uint8_t *input, uint64_t span, uint32_t input_length,
    uint32_t output_length, int64_t pad)
{
    CamppQconvV5AddressGeometry geometry = {
        input, span, (uint64_t)input_length * 8u, 1u, {8u, 0u},
        {input_length, 1u}, {output_length, 1u}, {1u, 1u},
        {1, 1}, {1, 1}, {pad, 0}, 1u, 8u, 8u, 1u
    };
    return geometry;
}

static CamppQconvV5AddressGeometry geometry_2d(
    const uint8_t *input, uint64_t span, uint32_t height,
    uint32_t width, int64_t pad)
{
    CamppQconvV5AddressGeometry geometry = {
        input, span, (uint64_t)height * width * 8u, 1u,
        {(uint64_t)width * 8u, 8u}, {height, width}, {height, width},
        {3u, 3u}, {1, 1}, {1, 1}, {pad, pad}, 1u, 8u, 8u, 2u
    };
    return geometry;
}

static void init_channel_packed_view(
    CamppTensorView *view, void *data, uint8_t rank,
    const uint32_t *dimensions, uint32_t channel_capacity)
{
    uint32_t spatial_stride = channel_capacity;
    uint8_t axis;

    memset(view, 0, sizeof(*view));
    view->data = data;
    view->dtype = CAMPP_DTYPE_UINT8;
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
    view->storage_span_bytes =
        (uint64_t)dimensions[0] * spatial_stride;
}

static int test_one_by_one_pointer_walk(void)
{
    uint8_t input[128] = {0};
    CamppQconvV5AddressGeometry geometry =
        geometry_1d(input, sizeof(input), 16u, 16u, 0);
    CamppQconvV5AddressPlan plan;
    CamppQconvV5AddressTile tile;
    uint32_t lane;

    tile.input_points[1][0] = input + sizeof(input) - 1u;
    CHECK_TRUE(campp_qconv_v5_address_plan_create(
        &geometry, &plan) == CAMPP_QCONV_V5_ADDRESS_OK);
    CHECK_TRUE(campp_qconv_v5_address_tile_1x1(
        &plan, 0u, 0u, 0u, 8u, &tile) == CAMPP_QCONV_V5_ADDRESS_OK);
    for (lane = 0u; lane < 8u; ++lane) {
        CHECK_TRUE(tile.output_linear[lane] == lane);
        CHECK_TRUE(tile.input_points[0][lane] == input + lane * 8u);
    }
    CHECK_TRUE(tile.input_points[1][0] == input + sizeof(input) - 1u);
    return 0;
}

static int test_three_by_three_and_padding(void)
{
    uint8_t input[1u * 8u * 5u * 12u] = {0};
    CamppQconvV5AddressGeometry geometry =
        geometry_2d(input, sizeof(input), 5u, 12u, 1);
    CamppQconvV5AddressPlan plan;
    CamppQconvV5AddressTile tile;
    uint32_t kernel_row;
    uint32_t kernel_column;
    uint32_t lane;

    CHECK_TRUE(campp_qconv_v5_address_plan_create(
        &geometry, &plan) == CAMPP_QCONV_V5_ADDRESS_OK);
    CHECK_TRUE(plan.interior_begin[0] == 1u &&
        plan.interior_end[0] == 4u);
    CHECK_TRUE(plan.interior_begin[1] == 1u &&
        plan.interior_end[1] == 11u);
    CHECK_TRUE(campp_qconv_v5_address_tile_3x3(
        &plan, 0u, 0u, 1u, 1u, 8u, &tile) ==
        CAMPP_QCONV_V5_ADDRESS_OK);
    for (kernel_row = 0u; kernel_row < 3u; ++kernel_row) {
        for (kernel_column = 0u; kernel_column < 3u; ++kernel_column) {
            const uint32_t kernel = kernel_row * 3u + kernel_column;
            for (lane = 0u; lane < 8u; ++lane) {
                const uint64_t expected =
                    ((uint64_t)kernel_row * 12u
                     + kernel_column + lane) * 8u;
                CHECK_TRUE(tile.input_points[kernel][lane] ==
                    input + expected);
            }
        }
    }

    for (kernel_row = 0u;
         kernel_row < CAMPP_QCONV_V5_ADDRESS_KERNEL_MAX; ++kernel_row) {
        for (lane = 0u; lane < CAMPP_QCONV_V5_ADDRESS_TILE_MAX; ++lane) {
            tile.input_points[kernel_row][lane] =
                input + sizeof(input) - 1u;
        }
    }
    CHECK_TRUE(campp_qconv_v5_address_tile_generic_2d(
        &plan, 0u, 0u, 0u, 0u, 8u, &tile) ==
        CAMPP_QCONV_V5_ADDRESS_OK);
    CHECK_TRUE(tile.input_points[0][0] == NULL);
    CHECK_TRUE(tile.input_points[1][0] == NULL);
    CHECK_TRUE(tile.input_points[3][0] == NULL);
    CHECK_TRUE(tile.input_points[4][0] == input);
    CHECK_TRUE(tile.input_points[8][7] ==
        input + (1u * 12u + 8u) * 8u);
    return 0;
}

static int test_row_schedule(void)
{
    uint8_t input[1u * 8u * 5u * 12u] = {0};
    CamppQconvV5AddressGeometry geometry =
        geometry_2d(input, sizeof(input), 5u, 12u, 1);
    CamppQconvV5AddressPlan plan;
    CamppQconvV5AddressCursor cursor;
    CamppQconvV5AddressWorkItem work;
    CamppQconvV5AddressStatus status;
    uint32_t covered = 0u;
    uint32_t interior_tiles = 0u;

    CHECK_TRUE(campp_qconv_v5_address_plan_create(
        &geometry, &plan) == CAMPP_QCONV_V5_ADDRESS_OK);
    campp_qconv_v5_address_schedule_begin(&cursor);
    while ((status = campp_qconv_v5_address_schedule_next(
                &plan, &cursor, &work)) == CAMPP_QCONV_V5_ADDRESS_OK) {
        CHECK_TRUE(work.output_column + work.tile_count <= 12u);
        covered += work.tile_count;
        if (work.path ==
            CAMPP_QCONV_V5_ADDRESS_PATH_THREE_BY_THREE_INTERIOR) {
            CHECK_TRUE(work.output_row >= 1u && work.output_row < 4u);
            CHECK_TRUE(work.output_column == 1u);
            CHECK_TRUE(work.tile_count == 8u);
            interior_tiles += 1u;
        }
    }
    CHECK_TRUE(status == CAMPP_QCONV_V5_ADDRESS_DONE);
    CHECK_TRUE(covered == 5u * 12u);
    CHECK_TRUE(interior_tiles == 3u);
    return 0;
}

static int test_v4_plan_adapter(void)
{
    uint8_t input_data[128] = {0};
    uint8_t output_data[128] = {0};
    const uint32_t dimensions[3] = {1u, 8u, 16u};
    CamppTensorView input;
    CamppTensorView output;
    CamppQconvV4ExecutionPlan v4;
    CamppQconvV5AddressPlan address;

    init_channel_packed_view(&input, input_data, 3u, dimensions, 8u);
    init_channel_packed_view(&output, output_data, 3u, dimensions, 8u);
    memset(&v4, 0, sizeof(v4));
    v4.input = &input;
    v4.output = &output;
    v4.spatial_rank = 1u;
    v4.input_channels = 8u;
    v4.inputs_per_group = 8u;
    v4.kernel_shape[0] = 1;
    v4.strides[0] = 1;
    v4.dilations[0] = 1;
    CHECK_TRUE(campp_qconv_v5_address_plan_from_v4(
        &v4, &address) == CAMPP_STATUS_OK);
    CHECK_TRUE(address.preferred_path ==
        CAMPP_QCONV_V5_ADDRESS_PATH_ONE_BY_ONE);
    CHECK_TRUE(address.output_step_bytes[0] == 8u);
    return 0;
}

int main(void)
{
    CHECK_TRUE(test_one_by_one_pointer_walk() == 0);
    CHECK_TRUE(test_three_by_three_and_padding() == 0);
    CHECK_TRUE(test_row_schedule() == 0);
    CHECK_TRUE(test_v4_plan_adapter() == 0);
    puts("QConv v5 address: PASS");
    return 0;
}
