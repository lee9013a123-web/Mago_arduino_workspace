/*
 * plan과 weights 바이트를 안전하게 읽는 저수준 도구의 구현이다.
 * 규칙과 의도는 binary_section_reader.h를 본다.
 */

#include "binary_section_reader.h"

#include <string.h>

uint16_t campp_read_u16_le(const uint8_t *bytes)
{
    return (uint16_t)((uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8));
}

uint32_t campp_read_u32_le(const uint8_t *bytes)
{
    return (uint32_t)bytes[0]
        | ((uint32_t)bytes[1] << 8)
        | ((uint32_t)bytes[2] << 16)
        | ((uint32_t)bytes[3] << 24);
}

uint64_t campp_read_u64_le(const uint8_t *bytes)
{
    return (uint64_t)bytes[0]
        | ((uint64_t)bytes[1] << 8)
        | ((uint64_t)bytes[2] << 16)
        | ((uint64_t)bytes[3] << 24)
        | ((uint64_t)bytes[4] << 32)
        | ((uint64_t)bytes[5] << 40)
        | ((uint64_t)bytes[6] << 48)
        | ((uint64_t)bytes[7] << 56);
}

bool campp_add_overflows_u64(uint64_t left, uint64_t right)
{
    return right > (UINT64_MAX - left);
}

CamppStatus campp_checked_add_u64(uint64_t left, uint64_t right, uint64_t *out_sum)
{
    if (out_sum == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    if (campp_add_overflows_u64(left, right)) {
        return CAMPP_STATUS_BUFFER_OVERFLOW;
    }
    *out_sum = left + right;
    return CAMPP_STATUS_OK;
}

CamppStatus campp_checked_mul_u64(
    uint64_t left, uint64_t right, uint64_t *out_product)
{
    if (out_product == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    if (left != 0u && right > (UINT64_MAX / left)) {
        return CAMPP_STATUS_BUFFER_OVERFLOW;
    }
    *out_product = left * right;
    return CAMPP_STATUS_OK;
}

CamppStatus campp_span_make(
    const void *data, uint64_t size, CamppByteSpan *out_span)
{
    if (out_span == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    if (data == NULL && size != 0u) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    out_span->data = (const uint8_t *)data;
    out_span->size = size;
    return CAMPP_STATUS_OK;
}

CamppStatus campp_span_check(
    const CamppByteSpan *span, uint64_t offset, uint64_t size)
{
    uint64_t end;

    if (span == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    if (span->data == NULL && span->size != 0u) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    /* wrap-around로 범위 검사를 통과하는 조작된 offset을 먼저 막는다. */
    if (campp_add_overflows_u64(offset, size)) {
        return CAMPP_STATUS_BUFFER_OVERFLOW;
    }
    end = offset + size;
    if (end > span->size) {
        return CAMPP_STATUS_BUFFER_OVERFLOW;
    }
    return CAMPP_STATUS_OK;
}

CamppStatus campp_span_check_aligned(
    const CamppByteSpan *span, uint64_t offset, uint64_t size,
    uint32_t alignment)
{
    CamppStatus status;

    if (alignment == 0u || (alignment & (alignment - 1u)) != 0u) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    status = campp_span_check(span, offset, size);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    if ((offset & (uint64_t)(alignment - 1u)) != 0u) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    return CAMPP_STATUS_OK;
}

CamppStatus campp_span_slice(
    const CamppByteSpan *span, uint64_t offset, uint64_t size,
    CamppByteSpan *out_span)
{
    CamppStatus status;

    if (out_span == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    status = campp_span_check(span, offset, size);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    out_span->data = (size == 0u) ? NULL : (span->data + offset);
    out_span->size = size;
    return CAMPP_STATUS_OK;
}

CamppStatus campp_span_read_u8(
    const CamppByteSpan *span, uint64_t offset, uint8_t *out_value)
{
    CamppStatus status;

    if (out_value == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    status = campp_span_check(span, offset, 1u);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    *out_value = span->data[offset];
    return CAMPP_STATUS_OK;
}

CamppStatus campp_span_read_u16(
    const CamppByteSpan *span, uint64_t offset, uint16_t *out_value)
{
    CamppStatus status;

    if (out_value == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    status = campp_span_check(span, offset, 2u);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    *out_value = campp_read_u16_le(span->data + offset);
    return CAMPP_STATUS_OK;
}

CamppStatus campp_span_read_u32(
    const CamppByteSpan *span, uint64_t offset, uint32_t *out_value)
{
    CamppStatus status;

    if (out_value == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    status = campp_span_check(span, offset, 4u);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    *out_value = campp_read_u32_le(span->data + offset);
    return CAMPP_STATUS_OK;
}

CamppStatus campp_span_read_u64(
    const CamppByteSpan *span, uint64_t offset, uint64_t *out_value)
{
    CamppStatus status;

    if (out_value == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    status = campp_span_check(span, offset, 8u);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    *out_value = campp_read_u64_le(span->data + offset);
    return CAMPP_STATUS_OK;
}

CamppStatus campp_span_read_i64(
    const CamppByteSpan *span, uint64_t offset, int64_t *out_value)
{
    CamppStatus status;
    uint64_t raw;
    int64_t value;

    if (out_value == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    status = campp_span_read_u64(span, offset, &raw);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    /*
     * 구현 정의 동작을 피하려고 cast 대신 복사한다. 두 타입 모두 8 byte이고
     * plan은 2의 보수로 기록하므로 표현이 그대로 옮겨진다.
     */
    memcpy(&value, &raw, sizeof(value));
    *out_value = value;
    return CAMPP_STATUS_OK;
}

CamppStatus campp_span_read_f64(
    const CamppByteSpan *span, uint64_t offset, double *out_value)
{
    CamppStatus status;
    uint64_t raw;
    double value;

    if (out_value == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    status = campp_span_read_u64(span, offset, &raw);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    memcpy(&value, &raw, sizeof(value));
    *out_value = value;
    return CAMPP_STATUS_OK;
}

CamppStatus campp_span_read_bytes(
    const CamppByteSpan *span, uint64_t offset, void *destination,
    uint64_t size)
{
    CamppStatus status;

    if (destination == NULL && size != 0u) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    status = campp_span_check(span, offset, size);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    if (size != 0u) {
        memcpy(destination, span->data + offset, (size_t)size);
    }
    return CAMPP_STATUS_OK;
}
