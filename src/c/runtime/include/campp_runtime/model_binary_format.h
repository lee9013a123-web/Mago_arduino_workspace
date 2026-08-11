#ifndef CAMPP_RUNTIME_MODEL_BINARY_FORMAT_H
#define CAMPP_RUNTIME_MODEL_BINARY_FORMAT_H

/*
 * Python binary_format_schema.py와 공유하는 execution-plan header ABI다.
 *
 * 디스크 규칙:
 * - 모든 다중 바이트 정수는 little-endian이다.
 * - header는 정확히 80바이트다.
 * - checksum은 header 뒤 payload 전체의 SHA-256이다.
 * - loader는 임의 정렬 pointer cast 대신 little-endian reader를 사용한다.
 */

#include <stddef.h>
#include <stdint.h>

#define CAMPP_PLAN_MAGIC_SIZE 8u
#define CAMPP_PLAN_CHECKSUM_SIZE 32u
#define CAMPP_PLAN_FORMAT_VERSION 1u
#define CAMPP_PLAN_HEADER_SIZE 80u
#define CAMPP_PLAN_SECTION_ALIGNMENT 8u

#define CAMPP_PLAN_MAGIC_OFFSET 0u
#define CAMPP_PLAN_FORMAT_VERSION_OFFSET 8u
#define CAMPP_PLAN_BUCKET_FRAMES_OFFSET 12u
#define CAMPP_PLAN_TENSOR_COUNT_OFFSET 16u
#define CAMPP_PLAN_OPERATOR_COUNT_OFFSET 20u
#define CAMPP_PLAN_TENSOR_TABLE_OFFSET_OFFSET 24u
#define CAMPP_PLAN_OPERATOR_TABLE_OFFSET_OFFSET 32u
#define CAMPP_PLAN_ATTRIBUTE_SECTION_OFFSET_OFFSET 40u
#define CAMPP_PLAN_CHECKSUM_OFFSET 48u

static const uint8_t CAMPP_PLAN_MAGIC[CAMPP_PLAN_MAGIC_SIZE] = {
    'C', 'A', 'M', 'P', 'L', 'A', 'N', '\0'
};

/*
 * loader가 little-endian 필드를 host 값으로 변환한 뒤 사용하는 표현이다.
 * 현재 QRB2210은 little-endian이지만, 디스크 데이터를 구조체로 직접 cast하지 않는다.
 */
typedef struct CamppPlanHeader {
    uint8_t magic[CAMPP_PLAN_MAGIC_SIZE];
    uint32_t format_version;
    uint32_t bucket_frames;
    uint32_t tensor_count;
    uint32_t operator_count;
    uint64_t tensor_table_offset;
    uint64_t operator_table_offset;
    uint64_t attribute_section_offset;
    uint8_t checksum[CAMPP_PLAN_CHECKSUM_SIZE];
} CamppPlanHeader;

#if defined(__cplusplus)
#define CAMPP_BINARY_STATIC_ASSERT(condition, message) static_assert(condition, message)
#else
#define CAMPP_BINARY_STATIC_ASSERT(condition, message) _Static_assert(condition, message)
#endif

CAMPP_BINARY_STATIC_ASSERT(
    sizeof(CamppPlanHeader) == CAMPP_PLAN_HEADER_SIZE,
    "CamppPlanHeader must remain exactly 80 bytes");
CAMPP_BINARY_STATIC_ASSERT(
    offsetof(CamppPlanHeader, format_version) == CAMPP_PLAN_FORMAT_VERSION_OFFSET,
    "format_version offset does not match the disk ABI");
CAMPP_BINARY_STATIC_ASSERT(
    offsetof(CamppPlanHeader, bucket_frames) == CAMPP_PLAN_BUCKET_FRAMES_OFFSET,
    "bucket_frames offset does not match the disk ABI");
CAMPP_BINARY_STATIC_ASSERT(
    offsetof(CamppPlanHeader, tensor_count) == CAMPP_PLAN_TENSOR_COUNT_OFFSET,
    "tensor_count offset does not match the disk ABI");
CAMPP_BINARY_STATIC_ASSERT(
    offsetof(CamppPlanHeader, operator_count) == CAMPP_PLAN_OPERATOR_COUNT_OFFSET,
    "operator_count offset does not match the disk ABI");
CAMPP_BINARY_STATIC_ASSERT(
    offsetof(CamppPlanHeader, tensor_table_offset) ==
        CAMPP_PLAN_TENSOR_TABLE_OFFSET_OFFSET,
    "tensor_table_offset offset does not match the disk ABI");
CAMPP_BINARY_STATIC_ASSERT(
    offsetof(CamppPlanHeader, operator_table_offset) ==
        CAMPP_PLAN_OPERATOR_TABLE_OFFSET_OFFSET,
    "operator_table_offset offset does not match the disk ABI");
CAMPP_BINARY_STATIC_ASSERT(
    offsetof(CamppPlanHeader, attribute_section_offset) ==
        CAMPP_PLAN_ATTRIBUTE_SECTION_OFFSET_OFFSET,
    "attribute_section_offset offset does not match the disk ABI");
CAMPP_BINARY_STATIC_ASSERT(
    offsetof(CamppPlanHeader, checksum) == CAMPP_PLAN_CHECKSUM_OFFSET,
    "checksum offset does not match the disk ABI");

#undef CAMPP_BINARY_STATIC_ASSERT

#endif /* CAMPP_RUNTIME_MODEL_BINARY_FORMAT_H */
