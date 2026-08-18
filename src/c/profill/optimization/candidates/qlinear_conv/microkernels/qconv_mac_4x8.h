#ifndef CAMPP_PROFILL_QCONV_MAC_4X8_H
#define CAMPP_PROFILL_QCONV_MAC_4X8_H

#include <stdint.h>

#include "qconv_mac_neon.h"

typedef enum CamppQconvMac4x8Implementation {
    CAMPP_QCONV_MAC_4X8_INTRINSICS = 0,
    CAMPP_QCONV_MAC_4X8_ASSEMBLY = 1
} CamppQconvMac4x8Implementation;

typedef enum CamppQconvMac4x8Result {
    CAMPP_QCONV_MAC_4X8_OK = 0,
    CAMPP_QCONV_MAC_4X8_UNSUPPORTED = 1,
    CAMPP_QCONV_MAC_4X8_FAILED = 2
} CamppQconvMac4x8Result;

/*
 * Compact ABI shared by the fixed intrinsics and AArch64 assembly kernels.
 * input_points addresses four spatial positions. Consecutive kernel rows are
 * separated by CAMPP_QCONV_CANDIDATE_TILE pointers.
 */
typedef struct CamppQconvMac4x8Params {
    const uint8_t *const *input_points;
    const uint8_t *packed_weight0;
    const uint8_t *packed_weight1;
    const int32_t *weight_zero;
    int32_t *output;
    uint32_t kernel_elements;
    uint32_t input_channels;
    int32_t input_zero;
} CamppQconvMac4x8Params;

/*
 * Tries the E7-specialized UINT8 x INT8 O4I4 path. Unsupported border, tail,
 * dtype and layout cases are reported to the caller so MAC v2 can handle them.
 */
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
        [CAMPP_QCONV_CANDIDATE_OUTPUT_TILE]);

int campp_qconv_mac_4x8_intrinsics_raw(
    const CamppQconvMac4x8Params *params);

#if defined(__aarch64__)
int campp_qconv_mac_4x8_aarch64_raw(
    const CamppQconvMac4x8Params *params);
#endif

#endif /* CAMPP_PROFILL_QCONV_MAC_4X8_H */
