#include <stdint.h>
#include <stdio.h>

#include "api/qconv_address_provider.h"

#define CHECK_TRUE(expression)                                      \
    do {                                                            \
        if (!(expression)) {                                        \
            fprintf(stderr, "CHECK failed at line %d: %s\n",     \
                    __LINE__, #expression);                         \
            return 1;                                               \
        }                                                           \
    } while (0)

static CamppQconvAddressGeometry geometry_1d(
    const uint8_t *input, uint64_t span, uint32_t input_length,
    uint32_t output_length, uint32_t kernel, int64_t pad)
{
    CamppQconvAddressGeometry geometry = {
        input,
        span,
        (uint64_t)input_length * 8u,
        1u,
        {8u, 0u},
        {input_length, 1u},
        {output_length, 1u},
        {kernel, 1u},
        {1, 1},
        {1, 1},
        {pad, 0},
        1u,
        8u,
        8u,
        1u
    };
    return geometry;
}

static CamppQconvAddressGeometry geometry_2d(
    const uint8_t *input, uint64_t span, uint32_t height,
    uint32_t width, int64_t pad)
{
    CamppQconvAddressGeometry geometry = {
        input,
        span,
        (uint64_t)height * width * 8u,
        1u,
        {(uint64_t)width * 8u, 8u},
        {height, width},
        {height, width},
        {3u, 3u},
        {1, 1},
        {1, 1},
        {pad, pad},
        1u,
        8u,
        8u,
        2u
    };
    return geometry;
}

static int test_one_by_one_pointer_walk(void)
{
    uint8_t input[128] = {0};
    CamppQconvAddressGeometry geometry =
        geometry_1d(input, sizeof(input), 16u, 16u, 1u, 0);
    CamppQconvAddressPlan plan;
    CamppQconvAddressTile tile;
    uint32_t index;

    tile.input_points[1][0] = input + sizeof(input) - 1u;
    CHECK_TRUE(campp_qconv_address_v2_plan_create(
        &geometry, &plan) == CAMPP_QCONV_ADDRESS_OK);
    CHECK_TRUE(plan.preferred_path == CAMPP_QCONV_ADDRESS_PATH_ONE_BY_ONE);
    CHECK_TRUE(plan.output_step_bytes[0] == 8u);
    CHECK_TRUE(campp_qconv_address_v2_tile_one_by_one(
        &plan, 0u, 0u, 0u, 8u, &tile) == CAMPP_QCONV_ADDRESS_OK);
    CHECK_TRUE(tile.tile_count == 8u);
    CHECK_TRUE(tile.kernel_elements == 1u);
    for (index = 0u; index < 8u; ++index) {
        CHECK_TRUE(tile.output_linear[index] == index);
        CHECK_TRUE(tile.input_points[0][index] == input + index * 8u);
    }
    CHECK_TRUE(tile.input_points[1][0] == input + sizeof(input) - 1u);
    return 0;
}

static int test_one_by_one_padding_falls_back(void)
{
    uint8_t input[128] = {0};
    CamppQconvAddressGeometry geometry =
        geometry_1d(input, sizeof(input), 16u, 18u, 1u, 1);
    CamppQconvAddressPlan plan;
    CamppQconvAddressTile tile;

    CHECK_TRUE(campp_qconv_address_v2_plan_create(
        &geometry, &plan) == CAMPP_QCONV_ADDRESS_OK);
    CHECK_TRUE(campp_qconv_address_v2_tile_one_by_one(
        &plan, 0u, 0u, 0u, 8u, &tile) ==
        CAMPP_QCONV_ADDRESS_UNSUPPORTED);
    CHECK_TRUE(campp_qconv_address_v2_tile_generic_1d(
        &plan, 0u, 0u, 0u, 8u, &tile) == CAMPP_QCONV_ADDRESS_OK);
    CHECK_TRUE(tile.input_points[0][0] == NULL);
    CHECK_TRUE(tile.input_points[0][1] == input);
    CHECK_TRUE(tile.input_points[0][7] == input + 48u);
    return 0;
}

static int test_three_by_three_pointer_walk(void)
{
    uint8_t input[1u * 8u * 5u * 12u] = {0};
    CamppQconvAddressGeometry geometry =
        geometry_2d(input, sizeof(input), 5u, 12u, 1);
    CamppQconvAddressPlan plan;
    CamppQconvAddressTile tile;
    uint32_t kernel_row;
    uint32_t kernel_column;
    uint32_t lane;

    CHECK_TRUE(campp_qconv_address_v2_plan_create(
        &geometry, &plan) == CAMPP_QCONV_ADDRESS_OK);
    CHECK_TRUE(plan.preferred_path ==
        CAMPP_QCONV_ADDRESS_PATH_THREE_BY_THREE_INTERIOR);
    CHECK_TRUE(plan.interior_begin[0] == 1u);
    CHECK_TRUE(plan.interior_end[0] == 4u);
    CHECK_TRUE(plan.interior_begin[1] == 1u);
    CHECK_TRUE(plan.interior_end[1] == 11u);
    CHECK_TRUE(campp_qconv_address_v2_tile_three_by_three(
        &plan, 0u, 0u, 1u, 1u, 8u, &tile) ==
        CAMPP_QCONV_ADDRESS_OK);
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
    CHECK_TRUE(tile.input_points[8][7] ==
        input + (2u * 12u + 9u) * 8u);
    return 0;
}

static int test_generic_explicitly_writes_padding(void)
{
    uint8_t input[1u * 8u * 5u * 12u] = {0};
    CamppQconvAddressGeometry geometry =
        geometry_2d(input, sizeof(input), 5u, 12u, 1);
    CamppQconvAddressPlan plan;
    CamppQconvAddressTile tile;
    uint32_t kernel;
    uint32_t lane;

    CHECK_TRUE(campp_qconv_address_v2_plan_create(
        &geometry, &plan) == CAMPP_QCONV_ADDRESS_OK);
    for (kernel = 0u; kernel < CAMPP_QCONV_ADDRESS_KERNEL_MAX; ++kernel) {
        for (lane = 0u; lane < CAMPP_QCONV_ADDRESS_TILE_MAX; ++lane) {
            tile.input_points[kernel][lane] = input + sizeof(input) - 1u;
        }
    }
    CHECK_TRUE(campp_qconv_address_v2_tile_generic_2d(
        &plan, 0u, 0u, 0u, 0u, 8u, &tile) == CAMPP_QCONV_ADDRESS_OK);
    CHECK_TRUE(tile.path == CAMPP_QCONV_ADDRESS_PATH_GENERIC);
    CHECK_TRUE(tile.input_points[0][0] == NULL);
    CHECK_TRUE(tile.input_points[1][0] == NULL);
    CHECK_TRUE(tile.input_points[3][0] == NULL);
    CHECK_TRUE(tile.input_points[4][0] == input);
    CHECK_TRUE(tile.input_points[8][7] ==
        input + (1u * 12u + 8u) * 8u);
    for (kernel = 0u; kernel < tile.kernel_elements; ++kernel) {
        for (lane = 0u; lane < tile.tile_count; ++lane) {
            CHECK_TRUE(tile.input_points[kernel][lane] == NULL ||
                (tile.input_points[kernel][lane] >= input &&
                 tile.input_points[kernel][lane] < input + sizeof(input)));
        }
    }
    return 0;
}

static int test_storage_span_guard(void)
{
    uint8_t input[1u * 8u * 5u * 12u] = {0};
    CamppQconvAddressGeometry geometry =
        geometry_2d(input, sizeof(input) - 17u, 5u, 12u, 1);
    CamppQconvAddressPlan plan;
    CamppQconvAddressTile tile;

    CHECK_TRUE(campp_qconv_address_v2_plan_create(
        &geometry, &plan) == CAMPP_QCONV_ADDRESS_OK);
    CHECK_TRUE(campp_qconv_address_v2_tile_three_by_three(
        &plan, 0u, 0u, 3u, 1u, 8u, &tile) ==
        CAMPP_QCONV_ADDRESS_OUT_OF_STORAGE);
    return 0;
}

static int test_row_schedule(void)
{
    uint8_t input[1u * 8u * 5u * 12u] = {0};
    CamppQconvAddressGeometry geometry =
        geometry_2d(input, sizeof(input), 5u, 12u, 1);
    CamppQconvAddressPlan plan;
    CamppQconvAddressCursor cursor;
    CamppQconvAddressWorkItem work;
    uint32_t covered = 0u;
    uint32_t interior_tiles = 0u;
    CamppQconvAddressStatus status;

    CHECK_TRUE(campp_qconv_address_v2_plan_create(
        &geometry, &plan) == CAMPP_QCONV_ADDRESS_OK);
    campp_qconv_address_v2_schedule_begin(&cursor);
    while ((status = campp_qconv_address_v2_schedule_next(
                &plan, &cursor, &work)) == CAMPP_QCONV_ADDRESS_OK) {
        CHECK_TRUE(work.tile_count > 0u);
        CHECK_TRUE(work.output_column + work.tile_count <= 12u);
        covered += work.tile_count;
        if (work.path ==
            CAMPP_QCONV_ADDRESS_PATH_THREE_BY_THREE_INTERIOR) {
            CHECK_TRUE(work.output_row >= 1u && work.output_row < 4u);
            CHECK_TRUE(work.output_column == 1u);
            CHECK_TRUE(work.tile_count == 8u);
            interior_tiles += 1u;
        }
    }
    CHECK_TRUE(status == CAMPP_QCONV_ADDRESS_DONE);
    CHECK_TRUE(covered == 5u * 12u);
    CHECK_TRUE(interior_tiles == 3u);
    return 0;
}

int main(void)
{
    CHECK_TRUE(test_one_by_one_pointer_walk() == 0);
    CHECK_TRUE(test_one_by_one_padding_falls_back() == 0);
    CHECK_TRUE(test_three_by_three_pointer_walk() == 0);
    CHECK_TRUE(test_generic_explicitly_writes_padding() == 0);
    CHECK_TRUE(test_storage_span_guard() == 0);
    CHECK_TRUE(test_row_schedule() == 0);
    puts("QConv address v2: PASS");
    return 0;
}
