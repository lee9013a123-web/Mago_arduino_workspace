#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "campp_runtime/tensor_descriptor.h"
#include "qconv_mac_4x8.h"
#include "qconv_mac_neon.h"

#define CHECK_TRUE(expression)                                         \
    do {                                                               \
        if (!(expression)) {                                           \
            fprintf(stderr, "CHECK failed at line %d: %s\n",        \
                    __LINE__, #expression);                            \
            return 1;                                                  \
        }                                                              \
    } while (0)

static void initialize_case(
    uint32_t kernel_elements,
    uint8_t input_data[CAMPP_QCONV_CANDIDATE_MAX_KERNEL_ELEMENTS]
        [CAMPP_QCONV_CANDIDATE_TILE][8],
    const uint8_t *input_points
        [CAMPP_QCONV_CANDIDATE_MAX_KERNEL_ELEMENTS]
        [CAMPP_QCONV_CANDIDATE_TILE],
    uint8_t packed_weight0[CAMPP_QCONV_CANDIDATE_MAX_KERNEL_ELEMENTS * 32],
    uint8_t packed_weight1[CAMPP_QCONV_CANDIDATE_MAX_KERNEL_ELEMENTS * 32])
{
    uint32_t kernel;
    for (kernel = 0u; kernel < kernel_elements; ++kernel) {
        uint32_t tile;
        for (tile = 0u; tile < CAMPP_QCONV_CANDIDATE_TILE; ++tile) {
            uint32_t channel;
            input_points[kernel][tile] = input_data[kernel][tile];
            for (channel = 0u; channel < 8u; ++channel) {
                input_data[kernel][tile][channel] = (uint8_t)(
                    111u + (kernel * 13u + tile * 7u + channel * 3u) % 31u);
            }
        }
        {
            uint32_t input_block;
            for (input_block = 0u; input_block < 2u; ++input_block) {
                uint32_t output_lane;
                for (output_lane = 0u; output_lane < 4u; ++output_lane) {
                    uint32_t input_lane;
                    for (input_lane = 0u; input_lane < 4u; ++input_lane) {
                        const size_t offset =
                            ((size_t)kernel * 2u + input_block) * 16u
                            + output_lane * 4u + input_lane;
                        const int8_t value0 = (int8_t)(
                            (int32_t)((kernel + input_block * 5u
                                      + output_lane * 3u + input_lane)
                                     % 15u) - 7);
                        const int8_t value1 = (int8_t)(
                            (int32_t)((kernel * 3u + input_block * 7u
                                      + output_lane * 5u + input_lane * 2u)
                                     % 17u) - 8);
                        memcpy(packed_weight0 + offset, &value0, 1u);
                        memcpy(packed_weight1 + offset, &value1, 1u);
                    }
                }
            }
        }
    }
}

static int run_case(
    uint32_t kernel_elements, CamppQconvMac4x8Implementation implementation)
{
    uint8_t input_data[CAMPP_QCONV_CANDIDATE_MAX_KERNEL_ELEMENTS]
        [CAMPP_QCONV_CANDIDATE_TILE][8] = {{{0}}};
    const uint8_t *input_points
        [CAMPP_QCONV_CANDIDATE_MAX_KERNEL_ELEMENTS]
        [CAMPP_QCONV_CANDIDATE_TILE] = {{NULL}};
    uint8_t packed_weight0
        [CAMPP_QCONV_CANDIDATE_MAX_KERNEL_ELEMENTS * 32] = {0};
    uint8_t packed_weight1
        [CAMPP_QCONV_CANDIDATE_MAX_KERNEL_ELEMENTS * 32] = {0};
    const uint8_t *packed_weights[2] = {
        packed_weight0, packed_weight1
    };
    const int32_t weight_zero[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE] = {
        -3, 1, 2, -1, 4, -2, 0, 3
    };
    const int32_t bias[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE] = {
        17, -23, 31, -47, 53, -61, 71, -83
    };
    int32_t reference[CAMPP_QCONV_CANDIDATE_TILE]
        [CAMPP_QCONV_CANDIDATE_OUTPUT_TILE] = {{0}};
    int32_t candidate[CAMPP_QCONV_CANDIDATE_TILE]
        [CAMPP_QCONV_CANDIDATE_OUTPUT_TILE] = {{0}};
    int32_t validated[CAMPP_QCONV_CANDIDATE_TILE]
        [CAMPP_QCONV_CANDIDATE_OUTPUT_TILE] = {{0}};
    CamppQconvMac4x8Result result;

    initialize_case(
        kernel_elements, input_data, input_points,
        packed_weight0, packed_weight1);
    CHECK_TRUE(campp_qconv_mac_neon_tile_v2(
        input_points, kernel_elements, CAMPP_QCONV_CANDIDATE_TILE,
        packed_weights, 8u, CAMPP_DTYPE_UINT8, CAMPP_DTYPE_INT8,
        127, weight_zero, bias, CAMPP_QCONV_CANDIDATE_OUTPUT_TILE,
        reference) == 0);
    result = campp_qconv_mac_4x8_try_tile(
        input_points, kernel_elements, CAMPP_QCONV_CANDIDATE_TILE,
        packed_weights, 8u, CAMPP_DTYPE_UINT8, CAMPP_DTYPE_INT8,
        127, weight_zero, bias, CAMPP_QCONV_CANDIDATE_OUTPUT_TILE,
        implementation, candidate);
#if defined(__aarch64__) && defined(__ARM_NEON)
    CHECK_TRUE(result == CAMPP_QCONV_MAC_4X8_OK);
    CHECK_TRUE(memcmp(reference, candidate, sizeof(reference)) == 0);
#else
    CHECK_TRUE(result == CAMPP_QCONV_MAC_4X8_UNSUPPORTED);
#endif
    if (implementation == CAMPP_QCONV_MAC_4X8_INTRINSICS) {
        result = campp_qconv_mac_4x8_intrinsics_validated(
            input_points, kernel_elements, packed_weights, 8u, 127,
            weight_zero, bias, validated);
#if defined(__aarch64__) && defined(__ARM_NEON)
        CHECK_TRUE(result == CAMPP_QCONV_MAC_4X8_OK);
        CHECK_TRUE(memcmp(reference, validated, sizeof(reference)) == 0);
#else
        CHECK_TRUE(result == CAMPP_QCONV_MAC_4X8_UNSUPPORTED);
#endif
    }
    return 0;
}

static int check_fallback_gate(void)
{
    uint8_t input_data[CAMPP_QCONV_CANDIDATE_MAX_KERNEL_ELEMENTS]
        [CAMPP_QCONV_CANDIDATE_TILE][8] = {{{0}}};
    const uint8_t *input_points
        [CAMPP_QCONV_CANDIDATE_MAX_KERNEL_ELEMENTS]
        [CAMPP_QCONV_CANDIDATE_TILE] = {{NULL}};
    uint8_t packed_weight0
        [CAMPP_QCONV_CANDIDATE_MAX_KERNEL_ELEMENTS * 32] = {0};
    uint8_t packed_weight1
        [CAMPP_QCONV_CANDIDATE_MAX_KERNEL_ELEMENTS * 32] = {0};
    const uint8_t *packed_weights[2] = {
        packed_weight0, packed_weight1
    };
    int32_t weight_zero[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE] = {0};
    int32_t bias[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE] = {0};
    int32_t output[CAMPP_QCONV_CANDIDATE_TILE]
        [CAMPP_QCONV_CANDIDATE_OUTPUT_TILE] = {{0}};

    initialize_case(
        1u, input_data, input_points, packed_weight0, packed_weight1);
    CHECK_TRUE(campp_qconv_mac_4x8_try_tile(
        input_points, 1u, 7u, packed_weights, 8u,
        CAMPP_DTYPE_UINT8, CAMPP_DTYPE_INT8, 127,
        weight_zero, bias, CAMPP_QCONV_CANDIDATE_OUTPUT_TILE,
        CAMPP_QCONV_MAC_4X8_INTRINSICS, output) ==
        CAMPP_QCONV_MAC_4X8_UNSUPPORTED);
    input_points[0][0] = NULL;
    CHECK_TRUE(campp_qconv_mac_4x8_try_tile(
        input_points, 1u, CAMPP_QCONV_CANDIDATE_TILE,
        packed_weights, 8u, CAMPP_DTYPE_UINT8, CAMPP_DTYPE_INT8, 127,
        weight_zero, bias, CAMPP_QCONV_CANDIDATE_OUTPUT_TILE,
        CAMPP_QCONV_MAC_4X8_INTRINSICS, output) ==
        CAMPP_QCONV_MAC_4X8_UNSUPPORTED);
    return 0;
}

int main(void)
{
    CHECK_TRUE(run_case(1u, CAMPP_QCONV_MAC_4X8_INTRINSICS) == 0);
    CHECK_TRUE(run_case(9u, CAMPP_QCONV_MAC_4X8_INTRINSICS) == 0);
    CHECK_TRUE(run_case(1u, CAMPP_QCONV_MAC_4X8_ASSEMBLY) == 0);
    CHECK_TRUE(run_case(9u, CAMPP_QCONV_MAC_4X8_ASSEMBLY) == 0);
    CHECK_TRUE(check_fallback_gate() == 0);
    puts("QConv fixed 4x8 microkernels: PASS");
    return 0;
}
