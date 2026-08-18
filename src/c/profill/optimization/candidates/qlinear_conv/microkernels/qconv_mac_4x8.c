#include "qconv_mac_4x8.h"

#include <limits.h>
#include <stddef.h>

#include "campp_runtime/tensor_descriptor.h"

#if defined(__aarch64__)
_Static_assert(offsetof(CamppQconvMac4x8Params, input_points) == 0u,
               "qconv 4x8 assembly ABI: input_points");
_Static_assert(offsetof(CamppQconvMac4x8Params, packed_weight0) == 8u,
               "qconv 4x8 assembly ABI: packed_weight0");
_Static_assert(offsetof(CamppQconvMac4x8Params, packed_weight1) == 16u,
               "qconv 4x8 assembly ABI: packed_weight1");
_Static_assert(offsetof(CamppQconvMac4x8Params, weight_zero) == 24u,
               "qconv 4x8 assembly ABI: weight_zero");
_Static_assert(offsetof(CamppQconvMac4x8Params, output) == 32u,
               "qconv 4x8 assembly ABI: output");
_Static_assert(offsetof(CamppQconvMac4x8Params, kernel_elements) == 40u,
               "qconv 4x8 assembly ABI: kernel_elements");
_Static_assert(offsetof(CamppQconvMac4x8Params, input_channels) == 44u,
               "qconv 4x8 assembly ABI: input_channels");
_Static_assert(offsetof(CamppQconvMac4x8Params, input_zero) == 48u,
               "qconv 4x8 assembly ABI: input_zero");
#endif

static int campp_qconv_mac_4x8_is_supported(
    const uint8_t *const input_points
        [CAMPP_QCONV_CANDIDATE_MAX_KERNEL_ELEMENTS]
        [CAMPP_QCONV_CANDIDATE_TILE],
    uint32_t kernel_elements, uint32_t tile_count,
    const uint8_t *const packed_weights[2], uint32_t input_channels,
    uint8_t input_dtype, uint8_t weight_dtype, int32_t input_zero,
    const int32_t weight_zero[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    const int32_t bias[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    uint32_t valid_outputs,
    int32_t accumulators[CAMPP_QCONV_CANDIDATE_TILE]
        [CAMPP_QCONV_CANDIDATE_OUTPUT_TILE])
{
    uint32_t kernel;
    uint32_t tile;
    uint32_t lane;

    if (input_points == NULL || packed_weights == NULL ||
        packed_weights[0] == NULL || packed_weights[1] == NULL ||
        weight_zero == NULL || bias == NULL || accumulators == NULL ||
        kernel_elements == 0u ||
        kernel_elements > CAMPP_QCONV_CANDIDATE_MAX_KERNEL_ELEMENTS ||
        tile_count != CAMPP_QCONV_CANDIDATE_TILE ||
        valid_outputs != CAMPP_QCONV_CANDIDATE_OUTPUT_TILE ||
        input_channels == 0u ||
        input_channels % CAMPP_QCONV_CANDIDATE_INPUT_BLOCK != 0u ||
        input_dtype != CAMPP_DTYPE_UINT8 ||
        weight_dtype != CAMPP_DTYPE_INT8 ||
        input_zero < 0 || input_zero > UINT8_MAX ||
        (uint64_t)kernel_elements * input_channels * UINT64_C(65025) >
            INT32_MAX) {
        return 0;
    }
    for (lane = 0u; lane < CAMPP_QCONV_CANDIDATE_OUTPUT_TILE; ++lane) {
        if (weight_zero[lane] < INT8_MIN ||
            weight_zero[lane] > INT8_MAX) {
            return 0;
        }
    }
    for (kernel = 0u; kernel < kernel_elements; ++kernel) {
        for (tile = 0u; tile < CAMPP_QCONV_CANDIDATE_TILE; ++tile) {
            if (input_points[kernel][tile] == NULL) return 0;
        }
    }
    return 1;
}

CamppQconvMac4x8Result campp_qconv_mac_4x8_try_tile(
    const uint8_t *const input_points
        [CAMPP_QCONV_CANDIDATE_MAX_KERNEL_ELEMENTS]
        [CAMPP_QCONV_CANDIDATE_TILE],
    uint32_t kernel_elements, uint32_t tile_count,
    const uint8_t *const packed_weights[2], uint32_t input_channels,
    uint8_t input_dtype, uint8_t weight_dtype, int32_t input_zero,
    const int32_t weight_zero[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    const int32_t bias[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    uint32_t valid_outputs, CamppQconvMac4x8Implementation implementation,
    int32_t accumulators[CAMPP_QCONV_CANDIDATE_TILE]
        [CAMPP_QCONV_CANDIDATE_OUTPUT_TILE])
{
    uint32_t half;
    uint32_t tile;

    if (implementation != CAMPP_QCONV_MAC_4X8_INTRINSICS &&
        implementation != CAMPP_QCONV_MAC_4X8_ASSEMBLY) {
        return CAMPP_QCONV_MAC_4X8_FAILED;
    }
    if (!campp_qconv_mac_4x8_is_supported(
            input_points, kernel_elements, tile_count, packed_weights,
            input_channels, input_dtype, weight_dtype, input_zero,
            weight_zero, bias, valid_outputs, accumulators)) {
        return CAMPP_QCONV_MAC_4X8_UNSUPPORTED;
    }

#if !defined(__aarch64__) || !defined(__ARM_NEON)
    (void)half;
    (void)tile;
    return CAMPP_QCONV_MAC_4X8_UNSUPPORTED;
#else
    for (half = 0u; half < 2u; ++half) {
        CamppQconvMac4x8Params params;
        int result;

        params.input_points = &input_points[0][half * 4u];
        params.packed_weight0 = packed_weights[0];
        params.packed_weight1 = packed_weights[1];
        params.weight_zero = weight_zero;
        params.output = &accumulators[half * 4u][0];
        params.kernel_elements = kernel_elements;
        params.input_channels = input_channels;
        params.input_zero = input_zero;
        if (implementation == CAMPP_QCONV_MAC_4X8_ASSEMBLY) {
            result = campp_qconv_mac_4x8_aarch64_raw(&params);
        } else {
            result = campp_qconv_mac_4x8_intrinsics_raw(&params);
        }
        if (result != 0) return CAMPP_QCONV_MAC_4X8_FAILED;
    }
    for (tile = 0u; tile < CAMPP_QCONV_CANDIDATE_TILE; ++tile) {
        uint32_t lane;
        for (lane = 0u; lane < CAMPP_QCONV_CANDIDATE_OUTPUT_TILE; ++lane) {
            const int64_t value = (int64_t)accumulators[tile][lane]
                + bias[lane];
            if (value < INT32_MIN || value > INT32_MAX) {
                return CAMPP_QCONV_MAC_4X8_FAILED;
            }
            accumulators[tile][lane] = (int32_t)value;
        }
    }
    return CAMPP_QCONV_MAC_4X8_OK;
#endif
}
