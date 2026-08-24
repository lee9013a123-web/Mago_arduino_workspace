#ifndef CAMPP_RUNTIME_MODEL_PACKAGE_FORMAT_H
#define CAMPP_RUNTIME_MODEL_PACKAGE_FORMAT_H

/*
 * .camppmodel on-disk ABI.  The package contains an execution plan, packed
 * weights, and versioned contracts for the future audio/scoring pipeline.
 * All integers are little-endian and sections are 64-byte aligned.
 *
 * v1: one section per type; every section flags field is zero.
 * v2: per-bucket types may repeat, and their section flags carry the bucket
 *     frame count.  Three optional types were added so that weight windowing
 *     and the layer-hybrid dispatch table can travel inside the model.  v1
 *     files still load -- their single plan belongs to the header bucket.
 */

#include <stdint.h>

#define CAMPP_MODEL_PACKAGE_MAGIC_SIZE 8u
#define CAMPP_MODEL_PACKAGE_FORMAT_VERSION 2u
#define CAMPP_MODEL_PACKAGE_MIN_READ_VERSION 1u
#define CAMPP_MODEL_PACKAGE_HEADER_SIZE 96u
#define CAMPP_MODEL_PACKAGE_SECTION_SIZE 64u
#define CAMPP_MODEL_PACKAGE_ALIGNMENT 64u
#define CAMPP_MODEL_PACKAGE_CHECKSUM_SIZE 32u
#define CAMPP_MODEL_PACKAGE_MAX_SECTIONS 32u
#define CAMPP_MODEL_PACKAGE_FLAG_LITTLE_ENDIAN 1u

#define CAMPP_MODEL_PACKAGE_MAGIC_OFFSET 0u
#define CAMPP_MODEL_PACKAGE_VERSION_OFFSET 8u
#define CAMPP_MODEL_PACKAGE_HEADER_SIZE_OFFSET 12u
#define CAMPP_MODEL_PACKAGE_SECTION_COUNT_OFFSET 16u
#define CAMPP_MODEL_PACKAGE_FLAGS_OFFSET 20u
#define CAMPP_MODEL_PACKAGE_BUCKET_OFFSET 24u
#define CAMPP_MODEL_PACKAGE_RESERVED_OFFSET 28u
#define CAMPP_MODEL_PACKAGE_FILE_SIZE_OFFSET 32u
#define CAMPP_MODEL_PACKAGE_TABLE_OFFSET_OFFSET 40u
#define CAMPP_MODEL_PACKAGE_PAYLOAD_OFFSET_OFFSET 48u
#define CAMPP_MODEL_PACKAGE_CHECKSUM_OFFSET 56u
#define CAMPP_MODEL_PACKAGE_RESERVED_TAIL_OFFSET 88u

#define CAMPP_MODEL_SECTION_TYPE_OFFSET 0u
#define CAMPP_MODEL_SECTION_FLAGS_OFFSET 4u
#define CAMPP_MODEL_SECTION_OFFSET_OFFSET 8u
#define CAMPP_MODEL_SECTION_SIZE_OFFSET 16u
#define CAMPP_MODEL_SECTION_CHECKSUM_OFFSET 24u
#define CAMPP_MODEL_SECTION_RESERVED_OFFSET 56u

static const uint8_t CAMPP_MODEL_PACKAGE_MAGIC[CAMPP_MODEL_PACKAGE_MAGIC_SIZE] = {
    'C', 'A', 'M', 'P', 'M', 'D', 'L', '\0'
};

typedef enum CamppModelPackageSectionType {
    CAMPP_MODEL_SECTION_EXECUTION_PLAN = 1,
    CAMPP_MODEL_SECTION_PACKED_WEIGHTS = 2,
    CAMPP_MODEL_SECTION_METADATA_JSON = 3,
    CAMPP_MODEL_SECTION_FRONTEND_JSON = 4,
    CAMPP_MODEL_SECTION_POSTPROCESS_JSON = 5,
    /* v2 optional sections. */
    CAMPP_MODEL_SECTION_STREAMING_WEIGHTS = 6,
    CAMPP_MODEL_SECTION_WEIGHT_STREAM_SCHEDULE = 7,
    CAMPP_MODEL_SECTION_KERNEL_DISPATCH_TABLE = 8
} CamppModelPackageSectionType;

#define CAMPP_MODEL_SECTION_TYPE_MIN CAMPP_MODEL_SECTION_EXECUTION_PLAN
#define CAMPP_MODEL_SECTION_TYPE_MAX CAMPP_MODEL_SECTION_KERNEL_DISPATCH_TABLE

/* Types whose section flags hold a bucket frame count and which may repeat. */
static inline int campp_model_section_is_per_bucket(uint32_t type)
{
    return type == (uint32_t)CAMPP_MODEL_SECTION_EXECUTION_PLAN ||
        type == (uint32_t)CAMPP_MODEL_SECTION_WEIGHT_STREAM_SCHEDULE ||
        type == (uint32_t)CAMPP_MODEL_SECTION_KERNEL_DISPATCH_TABLE;
}

#endif /* CAMPP_RUNTIME_MODEL_PACKAGE_FORMAT_H */
