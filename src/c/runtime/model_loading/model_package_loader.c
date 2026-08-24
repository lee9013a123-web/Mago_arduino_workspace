/* Load a self-contained .camppmodel into the existing RuntimeModel ABI. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "binary_section_reader.h"
#include "campp_runtime/model_package_format.h"
#include "compiled_model_validator.h"
#include "internal/runtime_model.h"

typedef struct CamppModelSectionView {
    uint32_t type;
    uint64_t offset;
    CamppByteSpan bytes;
} CamppModelSectionView;

static CamppStatus campp_package_read_file(
    const char *path, uint8_t **out_bytes, size_t *out_size)
{
    FILE *handle;
    long length;
    size_t size;
    uint8_t *buffer;

    if (path == NULL || out_bytes == NULL || out_size == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    handle = fopen(path, "rb");
    if (handle == NULL) return CAMPP_STATUS_FILE_NOT_FOUND;
    if (fseek(handle, 0L, SEEK_END) != 0 || (length = ftell(handle)) < 0L ||
        fseek(handle, 0L, SEEK_SET) != 0) {
        fclose(handle);
        return CAMPP_STATUS_FILE_READ_FAILED;
    }
    size = (size_t)length;
    buffer = (uint8_t *)malloc(size == 0u ? 1u : size);
    if (buffer == NULL) {
        fclose(handle);
        return CAMPP_STATUS_OUT_OF_MEMORY;
    }
    if (size != 0u && fread(buffer, 1u, size, handle) != size) {
        free(buffer);
        fclose(handle);
        return CAMPP_STATUS_FILE_READ_FAILED;
    }
    fclose(handle);
    *out_bytes = buffer;
    *out_size = size;
    return CAMPP_STATUS_OK;
}

static CamppStatus campp_package_align_u64(uint64_t value, uint64_t *out_value)
{
    const uint64_t mask = CAMPP_MODEL_PACKAGE_ALIGNMENT - 1u;
    uint64_t adjusted;

    if (out_value == NULL ||
        campp_add_overflows_u64(value, mask)) {
        return CAMPP_STATUS_BUFFER_OVERFLOW;
    }
    adjusted = value + mask;
    *out_value = adjusted & ~mask;
    return CAMPP_STATUS_OK;
}

static int campp_package_sections_overlap(
    const CamppModelSectionView *sections, uint32_t count,
    uint64_t offset, uint64_t size)
{
    uint32_t index;
    uint64_t end = offset + size;

    for (index = 0u; index < count; ++index) {
        uint64_t prior_end = sections[index].offset + sections[index].bytes.size;
        if (offset < prior_end && sections[index].offset < end) return 1;
    }
    return 0;
}

static CamppStatus campp_package_parse(
    const uint8_t *bytes, size_t size, uint32_t *out_bucket,
    CamppModelSectionView *out_plan, CamppModelSectionView *out_weights)
{
    CamppByteSpan package;
    CamppStatus status;
    uint8_t magic[CAMPP_MODEL_PACKAGE_MAGIC_SIZE];
    uint8_t payload_checksum[CAMPP_MODEL_PACKAGE_CHECKSUM_SIZE];
    uint32_t version, header_size, section_count, flags, bucket, reserved;
    uint64_t file_size, table_offset, payload_offset, reserved_tail;
    uint64_t table_size, table_end, expected_payload_offset;
    CamppModelSectionView sections[CAMPP_MODEL_PACKAGE_MAX_SECTIONS];
    uint32_t seen_mask = 0u;
    uint32_t index;

    if (bytes == NULL || out_bucket == NULL || out_plan == NULL ||
        out_weights == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    status = campp_span_make(bytes, (uint64_t)size, &package);
    if (status != CAMPP_STATUS_OK) return status;
    if (package.size < CAMPP_MODEL_PACKAGE_HEADER_SIZE) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    status = campp_span_read_bytes(
        &package, CAMPP_MODEL_PACKAGE_MAGIC_OFFSET, magic, sizeof(magic));
    if (status != CAMPP_STATUS_OK) return status;
    if (memcmp(magic, CAMPP_MODEL_PACKAGE_MAGIC, sizeof(magic)) != 0) {
        return CAMPP_STATUS_INVALID_MAGIC;
    }
#define CAMPP_READ_PACKAGE_U32(field, target)                                  \
    do {                                                                        \
        status = campp_span_read_u32(&package, (field), &(target));             \
        if (status != CAMPP_STATUS_OK) return status;                           \
    } while (0)
#define CAMPP_READ_PACKAGE_U64(field, target)                                  \
    do {                                                                        \
        status = campp_span_read_u64(&package, (field), &(target));             \
        if (status != CAMPP_STATUS_OK) return status;                           \
    } while (0)
    CAMPP_READ_PACKAGE_U32(CAMPP_MODEL_PACKAGE_VERSION_OFFSET, version);
    CAMPP_READ_PACKAGE_U32(CAMPP_MODEL_PACKAGE_HEADER_SIZE_OFFSET, header_size);
    CAMPP_READ_PACKAGE_U32(CAMPP_MODEL_PACKAGE_SECTION_COUNT_OFFSET, section_count);
    CAMPP_READ_PACKAGE_U32(CAMPP_MODEL_PACKAGE_FLAGS_OFFSET, flags);
    CAMPP_READ_PACKAGE_U32(CAMPP_MODEL_PACKAGE_BUCKET_OFFSET, bucket);
    CAMPP_READ_PACKAGE_U32(CAMPP_MODEL_PACKAGE_RESERVED_OFFSET, reserved);
    CAMPP_READ_PACKAGE_U64(CAMPP_MODEL_PACKAGE_FILE_SIZE_OFFSET, file_size);
    CAMPP_READ_PACKAGE_U64(CAMPP_MODEL_PACKAGE_TABLE_OFFSET_OFFSET, table_offset);
    CAMPP_READ_PACKAGE_U64(CAMPP_MODEL_PACKAGE_PAYLOAD_OFFSET_OFFSET, payload_offset);
    CAMPP_READ_PACKAGE_U64(CAMPP_MODEL_PACKAGE_RESERVED_TAIL_OFFSET, reserved_tail);
#undef CAMPP_READ_PACKAGE_U32
#undef CAMPP_READ_PACKAGE_U64
    if (version < CAMPP_MODEL_PACKAGE_MIN_READ_VERSION ||
        version > CAMPP_MODEL_PACKAGE_FORMAT_VERSION) {
        return CAMPP_STATUS_UNSUPPORTED_VERSION;
    }
    if (header_size != CAMPP_MODEL_PACKAGE_HEADER_SIZE ||
        section_count == 0u ||
        section_count > CAMPP_MODEL_PACKAGE_MAX_SECTIONS ||
        flags != CAMPP_MODEL_PACKAGE_FLAG_LITTLE_ENDIAN || bucket == 0u ||
        reserved != 0u || reserved_tail != 0u || file_size != package.size ||
        table_offset != CAMPP_MODEL_PACKAGE_HEADER_SIZE) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    status = campp_checked_mul_u64(
        section_count, CAMPP_MODEL_PACKAGE_SECTION_SIZE, &table_size);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_checked_add_u64(table_offset, table_size, &table_end);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_package_align_u64(table_end, &expected_payload_offset);
    if (status != CAMPP_STATUS_OK) return status;
    if (payload_offset != expected_payload_offset || payload_offset > file_size) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    status = campp_span_read_bytes(
        &package, CAMPP_MODEL_PACKAGE_CHECKSUM_OFFSET, payload_checksum,
        sizeof(payload_checksum));
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_validate_sha256(
        bytes + header_size, package.size - header_size, payload_checksum);
    if (status != CAMPP_STATUS_OK) return status;

    memset(sections, 0, sizeof(sections));
    memset(out_plan, 0, sizeof(*out_plan));
    memset(out_weights, 0, sizeof(*out_weights));
    for (index = 0u; index < section_count; ++index) {
        CamppByteSpan entry;
        uint32_t type, section_flags;
        int per_bucket;
        uint64_t offset, section_size, section_reserved;
        uint8_t checksum[CAMPP_MODEL_PACKAGE_CHECKSUM_SIZE];

        status = campp_span_slice(
            &package, table_offset + index * CAMPP_MODEL_PACKAGE_SECTION_SIZE,
            CAMPP_MODEL_PACKAGE_SECTION_SIZE, &entry);
        if (status != CAMPP_STATUS_OK) return status;
        status = campp_span_read_u32(
            &entry, CAMPP_MODEL_SECTION_TYPE_OFFSET, &type);
        if (status != CAMPP_STATUS_OK) return status;
        status = campp_span_read_u32(
            &entry, CAMPP_MODEL_SECTION_FLAGS_OFFSET, &section_flags);
        if (status != CAMPP_STATUS_OK) return status;
        status = campp_span_read_u64(
            &entry, CAMPP_MODEL_SECTION_OFFSET_OFFSET, &offset);
        if (status != CAMPP_STATUS_OK) return status;
        status = campp_span_read_u64(
            &entry, CAMPP_MODEL_SECTION_SIZE_OFFSET, &section_size);
        if (status != CAMPP_STATUS_OK) return status;
        status = campp_span_read_bytes(
            &entry, CAMPP_MODEL_SECTION_CHECKSUM_OFFSET,
            checksum, sizeof(checksum));
        if (status != CAMPP_STATUS_OK) return status;
        status = campp_span_read_u64(
            &entry, CAMPP_MODEL_SECTION_RESERVED_OFFSET, &section_reserved);
        if (status != CAMPP_STATUS_OK) return status;
        /* v2는 per-bucket 타입이 반복될 수 있고 그 flags가 bucket을 담는다.
         * 그 외 타입은 v1과 같이 한 번만, flags 0으로 엄격히 본다. */
        per_bucket = version >= 2u && campp_model_section_is_per_bucket(type);
        if (type < (uint32_t)CAMPP_MODEL_SECTION_TYPE_MIN ||
            type > (uint32_t)CAMPP_MODEL_SECTION_TYPE_MAX ||
            section_reserved != 0u ||
            (!per_bucket && section_flags != 0u) ||
            (per_bucket && section_flags == 0u) ||
            section_size == 0u || offset < payload_offset ||
            (offset % CAMPP_MODEL_PACKAGE_ALIGNMENT) != 0u ||
            (!per_bucket && (seen_mask & (1u << type)) != 0u)) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
        status = campp_span_check(&package, offset, section_size);
        if (status != CAMPP_STATUS_OK) return status;
        if (campp_package_sections_overlap(
                sections, index, offset, section_size)) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
        sections[index].type = type;
        sections[index].offset = offset;
        status = campp_span_slice(
            &package, offset, section_size, &sections[index].bytes);
        if (status != CAMPP_STATUS_OK) return status;
        status = campp_validate_sha256(
            sections[index].bytes.data, sections[index].bytes.size, checksum);
        if (status != CAMPP_STATUS_OK) return status;
        seen_mask |= 1u << type;
        if (type == (uint32_t)CAMPP_MODEL_SECTION_EXECUTION_PLAN) {
            /* per-bucket일 때는 헤더 bucket과 짝이 맞는 plan만 고른다.
             * v1은 flags가 0이라 그대로 받는다. */
            if (!per_bucket || section_flags == bucket) {
                *out_plan = sections[index];
            }
        } else if (type == (uint32_t)CAMPP_MODEL_SECTION_PACKED_WEIGHTS) {
            *out_weights = sections[index];
        }
    }
    if ((seen_mask & 0x3eu) != 0x3eu || out_plan->bytes.data == NULL ||
        out_weights->bytes.data == NULL) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    *out_bucket = bucket;
    return CAMPP_STATUS_OK;
}

CamppStatus campp_runtime_model_load_package(
    const char *package_path, CamppRuntimeModel *model)
{
    uint8_t *package_bytes = NULL;
    size_t package_size = 0u;
    uint8_t *plan_bytes = NULL;
    uint8_t *weight_bytes = NULL;
    CamppModelSectionView plan;
    CamppModelSectionView weights;
    CamppRuntimeModel staging;
    uint32_t bucket = 0u;
    CamppStatus status;

    if (package_path == NULL || model == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    memset(&staging, 0, sizeof(staging));
    status = campp_package_read_file(
        package_path, &package_bytes, &package_size);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_package_parse(
        package_bytes, package_size, &bucket, &plan, &weights);
    if (status != CAMPP_STATUS_OK) goto failed;
#if SIZE_MAX < UINT64_MAX
    if (plan.bytes.size > (uint64_t)SIZE_MAX ||
        weights.bytes.size > (uint64_t)SIZE_MAX) {
        status = CAMPP_STATUS_BUFFER_OVERFLOW;
        goto failed;
    }
#endif
    plan_bytes = (uint8_t *)malloc((size_t)plan.bytes.size);
    weight_bytes = (uint8_t *)malloc((size_t)weights.bytes.size);
    if (plan_bytes == NULL || weight_bytes == NULL) {
        status = CAMPP_STATUS_OUT_OF_MEMORY;
        goto failed;
    }
    memcpy(plan_bytes, plan.bytes.data, (size_t)plan.bytes.size);
    memcpy(weight_bytes, weights.bytes.data, (size_t)weights.bytes.size);
    status = campp_runtime_model_adopt(
        plan_bytes, (size_t)plan.bytes.size, true,
        weight_bytes, (size_t)weights.bytes.size, true, &staging);
    if (status != CAMPP_STATUS_OK) goto failed;
    plan_bytes = NULL;
    weight_bytes = NULL;
    if (staging.bucket_frames != bucket) {
        campp_runtime_model_release(&staging);
        status = CAMPP_STATUS_BUCKET_MISMATCH;
        goto failed;
    }
    free(package_bytes);
    *model = staging;
    return CAMPP_STATUS_OK;

failed:
    free(plan_bytes);
    free(weight_bytes);
    free(package_bytes);
    return status;
}
