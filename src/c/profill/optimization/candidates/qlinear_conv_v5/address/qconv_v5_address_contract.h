#ifndef CAMPP_PROFILL_QCONV_V5_ADDRESS_CONTRACT_H
#define CAMPP_PROFILL_QCONV_V5_ADDRESS_CONTRACT_H

#include <stdint.h>

#define CAMPP_QCONV_V5_ADDRESS_TILE_MAX 8u
#define CAMPP_QCONV_V5_ADDRESS_KERNEL_MAX 9u

typedef enum CamppQconvV5AddressPath {
    CAMPP_QCONV_V5_ADDRESS_PATH_ONE_BY_ONE = 0,
    CAMPP_QCONV_V5_ADDRESS_PATH_THREE_BY_THREE_INTERIOR = 1,
    CAMPP_QCONV_V5_ADDRESS_PATH_GENERIC = 2
} CamppQconvV5AddressPath;

typedef struct CamppQconvV5AddressGeometry {
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
    uint32_t batch_count;
    uint32_t input_channels;
    uint32_t input_channels_per_group;
    uint8_t spatial_rank;
} CamppQconvV5AddressGeometry;

typedef struct CamppQconvV5AddressPlan {
    CamppQconvV5AddressGeometry geometry;
    uint64_t output_step_bytes[2];
    uint64_t kernel_step_bytes[2];
    uint32_t interior_begin[2];
    uint32_t interior_end[2];
    uint32_t output_spatial_count;
    uint8_t kernel_elements;
    CamppQconvV5AddressPath preferred_path;
} CamppQconvV5AddressPlan;

/* A NULL input point denotes quantized padding and contributes input_zero. */
typedef struct CamppQconvV5AddressTile {
    const uint8_t *input_points[CAMPP_QCONV_V5_ADDRESS_KERNEL_MAX]
        [CAMPP_QCONV_V5_ADDRESS_TILE_MAX];
    uint32_t output_linear[CAMPP_QCONV_V5_ADDRESS_TILE_MAX];
    uint8_t tile_count;
    uint8_t kernel_elements;
    CamppQconvV5AddressPath path;
} CamppQconvV5AddressTile;

typedef struct CamppQconvV5AddressCursor {
    uint32_t output_row;
    uint32_t output_column;
} CamppQconvV5AddressCursor;

/* Compact schedule entry; it deliberately does not cache the 9x8 pointers. */
typedef struct CamppQconvV5AddressWorkItem {
    uint32_t output_row;
    uint32_t output_column;
    uint8_t tile_count;
    CamppQconvV5AddressPath path;
} CamppQconvV5AddressWorkItem;

#endif /* CAMPP_PROFILL_QCONV_V5_ADDRESS_CONTRACT_H */
