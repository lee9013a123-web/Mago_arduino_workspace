#ifndef CAMPP_RUNTIME_MODEL_LOADING_BINARY_SECTION_READER_H
#define CAMPP_RUNTIME_MODEL_LOADING_BINARY_SECTION_READER_H

/*
 * plan과 weights 바이트를 안전하게 읽는 저수준 도구다.
 *
 * 디스크 데이터를 구조체 pointer로 cast하지 않는다. 파일이 어떤 정렬로
 * 놓였는지, host가 어떤 byte order인지와 무관하게 같은 값을 얻기 위해서다.
 * 값은 항상 little-endian으로 한 byte씩 조립한다.
 *
 * 모든 읽기는 span 범위 검사를 먼저 통과한다. offset + size가 uint64를 넘는
 * 경우까지 걸러내므로, 조작된 파일이 wrap-around로 범위 검사를 통과하는 일이
 * 없다. 이 파일의 함수를 거치지 않고 plan 바이트를 직접 만지는 코드는 없어야
 * 한다.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "campp_runtime/status_code.h"

/* 읽기 전용 바이트 구간. data가 NULL이면 size도 0이어야 한다. */
typedef struct CamppByteSpan {
    const uint8_t *data;
    uint64_t size;
} CamppByteSpan;

/* --- 원시 정수 읽기 (호출자가 이미 범위를 확인한 경우) --- */

uint16_t campp_read_u16_le(const uint8_t *bytes);
uint32_t campp_read_u32_le(const uint8_t *bytes);
uint64_t campp_read_u64_le(const uint8_t *bytes);

/* --- overflow를 검사하는 산술 --- */

bool campp_add_overflows_u64(uint64_t left, uint64_t right);
CamppStatus campp_checked_add_u64(uint64_t left, uint64_t right, uint64_t *out_sum);
CamppStatus campp_checked_mul_u64(
    uint64_t left, uint64_t right, uint64_t *out_product);

/* --- span 검사 --- */

CamppStatus campp_span_make(
    const void *data, uint64_t size, CamppByteSpan *out_span);

/* [offset, offset + size)가 span 안에 들어가는지 본다. */
CamppStatus campp_span_check(
    const CamppByteSpan *span, uint64_t offset, uint64_t size);

/* 범위 검사에 더해 offset이 alignment의 배수인지 본다. */
CamppStatus campp_span_check_aligned(
    const CamppByteSpan *span, uint64_t offset, uint64_t size,
    uint32_t alignment);

/* 부분 구간을 잘라낸다. 잘라낸 span의 offset은 다시 0부터 시작한다. */
CamppStatus campp_span_slice(
    const CamppByteSpan *span, uint64_t offset, uint64_t size,
    CamppByteSpan *out_span);

/* --- 범위를 확인하며 읽기 --- */

CamppStatus campp_span_read_u16(
    const CamppByteSpan *span, uint64_t offset, uint16_t *out_value);
CamppStatus campp_span_read_u32(
    const CamppByteSpan *span, uint64_t offset, uint32_t *out_value);
CamppStatus campp_span_read_u64(
    const CamppByteSpan *span, uint64_t offset, uint64_t *out_value);
CamppStatus campp_span_read_u8(
    const CamppByteSpan *span, uint64_t offset, uint8_t *out_value);
CamppStatus campp_span_read_i64(
    const CamppByteSpan *span, uint64_t offset, int64_t *out_value);

/* IEEE-754 double을 8 byte little-endian에서 읽는다. */
CamppStatus campp_span_read_f64(
    const CamppByteSpan *span, uint64_t offset, double *out_value);

CamppStatus campp_span_read_bytes(
    const CamppByteSpan *span, uint64_t offset, void *destination,
    uint64_t size);

#endif /* CAMPP_RUNTIME_MODEL_LOADING_BINARY_SECTION_READER_H */
