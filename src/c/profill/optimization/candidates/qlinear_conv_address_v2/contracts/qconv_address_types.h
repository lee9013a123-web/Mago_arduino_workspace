#ifndef CAMPP_PROFILL_QCONV_ADDRESS_V2_TYPES_H
#define CAMPP_PROFILL_QCONV_ADDRESS_V2_TYPES_H

#include <stdint.h>

#define CAMPP_QCONV_ADDRESS_TILE_MAX 8u
#define CAMPP_QCONV_ADDRESS_KERNEL_MAX 9u

typedef enum CamppQconvAddressPath {
    CAMPP_QCONV_ADDRESS_PATH_ONE_BY_ONE = 0,
    CAMPP_QCONV_ADDRESS_PATH_THREE_BY_THREE_INTERIOR = 1,
    CAMPP_QCONV_ADDRESS_PATH_GENERIC = 2
} CamppQconvAddressPath;

/*
 * Runtime-neutral geometry. The V4 adapter constructs this once from its
 * execution plan; address kernels do not inspect model/operator descriptors.
 */
typedef struct CamppQconvAddressGeometry {
    const uint8_t *input_base;
    uint64_t storage_span_bytes;
    uint64_t batch_stride_bytes;
    uint64_t channel_stride_bytes;
    uint64_t input_spatial_stride_bytes[2];
    uint32_t input_spatial[2];
    uint32_t output_spatial[2];
    uint32_t kernel_shape[2];
    int64_t convolution_stride[2];
    int64_t dilation[2];
    int64_t pad_begin[2];
    uint32_t input_channels_per_group;
    uint8_t spatial_rank;
} CamppQconvAddressGeometry;

/* Values derived once and reused by every output-channel block. */
typedef struct CamppQconvAddressPlan {
    CamppQconvAddressGeometry geometry;
    uint64_t output_step_bytes[2];
    uint64_t kernel_step_bytes[2];
    uint32_t interior_begin[2];
    uint32_t interior_end[2];
    uint32_t output_spatial_count;
    uint8_t kernel_elements;
    CamppQconvAddressPath preferred_path;
} CamppQconvAddressPlan;

/*
 * Neutral producer/consumer contract. Address fills it; MAC consumes it.
 * A NULL point denotes a padded input and must contribute input_zero.
 */
typedef struct CamppQconvAddressTile {
    const uint8_t *input_points[CAMPP_QCONV_ADDRESS_KERNEL_MAX]
        [CAMPP_QCONV_ADDRESS_TILE_MAX];
    uint32_t output_linear[CAMPP_QCONV_ADDRESS_TILE_MAX];
    uint8_t tile_count;
    uint8_t kernel_elements;
    CamppQconvAddressPath path;
} CamppQconvAddressTile;

#endif /* CAMPP_PROFILL_QCONV_ADDRESS_V2_TYPES_H */
