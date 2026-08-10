#ifndef CAMPP_RUNTIME_OPERATOR_DESCRIPTOR_H
#define CAMPP_RUNTIME_OPERATOR_DESCRIPTOR_H

/*
 * Python binary_format_schema.py와 공유하는 Operator descriptor ABI다.
 * 현재 동결된 CAM++ graph의 최대 입력은 9개이고 모든 Runtime node의
 * 출력은 1개이므로 고정 64바이트 명령으로 표현한다.
 */

#include <stddef.h>
#include <stdint.h>

#include "tensor_descriptor.h"

#define CAMPP_OPERATOR_INPUT_CAPACITY 9u
#define CAMPP_OPERATOR_OUTPUT_CAPACITY 1u
#define CAMPP_OPERATOR_DESCRIPTOR_SIZE 64u
#define CAMPP_DEFAULT_KERNEL_ID 0u

/* Python OperatorCode와 숫자값을 동일하게 유지한다. */
typedef enum CamppOperatorCode {
    CAMPP_OP_INVALID = 0,
    CAMPP_OP_QLINEAR_CONV = 1,
    CAMPP_OP_QUANTIZE_LINEAR = 2,
    CAMPP_OP_DEQUANTIZE_LINEAR = 3,
    CAMPP_OP_BATCH_NORMALIZATION = 4,
    CAMPP_OP_RELU = 5,
    CAMPP_OP_SIGMOID = 6,
    CAMPP_OP_AVERAGE_POOL = 7,
    CAMPP_OP_REDUCE_MEAN = 8,
    CAMPP_OP_ADD = 9,
    CAMPP_OP_MUL = 10,
    CAMPP_OP_SUB = 11,
    CAMPP_OP_DIV = 12,
    CAMPP_OP_SQRT = 13,
    CAMPP_OP_CONCAT = 14,
    CAMPP_OP_EXPAND = 15,
    CAMPP_OP_SLICE = 16,
    CAMPP_OP_RESHAPE = 17,
    CAMPP_OP_TRANSPOSE = 18,
    CAMPP_OP_SQUEEZE = 19,
    CAMPP_OP_UNSQUEEZE = 20
} CamppOperatorCode;

/* 실행 kernel을 제공하는 backend를 식별한다. */
typedef enum CamppBackendId {
    CAMPP_BACKEND_AUTO = 0,
    CAMPP_BACKEND_CPU_REFERENCE = 1,
    CAMPP_BACKEND_CPU_AARCH64 = 2,
    CAMPP_BACKEND_GPU_OPENCL = 3,
    CAMPP_BACKEND_GPU_VULKAN = 4
} CamppBackendId;

/*
 * 고정 64바이트 Operator descriptor.
 *
 * 사용하지 않는 input/output 슬롯은 CAMPP_INVALID_TENSOR_ID로 채운다.
 * attribute_offset은 plan 파일 시작이 아니라 PlanHeader가 가리키는
 * attribute section 시작부터의 상대 byte offset이다. attribute가 없으면
 * attribute_offset과 attribute_size를 모두 0으로 기록한다.
 */
typedef struct CamppOperatorDescriptor {
    uint32_t operator_id;
    uint16_t opcode;
    uint8_t input_count;
    uint8_t output_count;
    uint32_t input_tensor_ids[CAMPP_OPERATOR_INPUT_CAPACITY];
    uint32_t output_tensor_ids[CAMPP_OPERATOR_OUTPUT_CAPACITY];
    uint64_t attribute_offset;
    uint32_t attribute_size;
    uint16_t backend_id;
    uint16_t kernel_id;
} CamppOperatorDescriptor;

#if defined(__cplusplus)
#define CAMPP_OPERATOR_STATIC_ASSERT(condition, message) static_assert(condition, message)
#else
#define CAMPP_OPERATOR_STATIC_ASSERT(condition, message) _Static_assert(condition, message)
#endif

CAMPP_OPERATOR_STATIC_ASSERT(
    sizeof(CamppOperatorDescriptor) == CAMPP_OPERATOR_DESCRIPTOR_SIZE,
    "CamppOperatorDescriptor must remain exactly 64 bytes");
CAMPP_OPERATOR_STATIC_ASSERT(
    offsetof(CamppOperatorDescriptor, operator_id) == 0u,
    "operator_id offset does not match the disk ABI");
CAMPP_OPERATOR_STATIC_ASSERT(
    offsetof(CamppOperatorDescriptor, opcode) == 4u,
    "opcode offset does not match the disk ABI");
CAMPP_OPERATOR_STATIC_ASSERT(
    offsetof(CamppOperatorDescriptor, input_count) == 6u,
    "input_count offset does not match the disk ABI");
CAMPP_OPERATOR_STATIC_ASSERT(
    offsetof(CamppOperatorDescriptor, output_count) == 7u,
    "output_count offset does not match the disk ABI");
CAMPP_OPERATOR_STATIC_ASSERT(
    offsetof(CamppOperatorDescriptor, input_tensor_ids) == 8u,
    "input_tensor_ids offset does not match the disk ABI");
CAMPP_OPERATOR_STATIC_ASSERT(
    offsetof(CamppOperatorDescriptor, output_tensor_ids) == 44u,
    "output_tensor_ids offset does not match the disk ABI");
CAMPP_OPERATOR_STATIC_ASSERT(
    offsetof(CamppOperatorDescriptor, attribute_offset) == 48u,
    "attribute_offset offset does not match the disk ABI");
CAMPP_OPERATOR_STATIC_ASSERT(
    offsetof(CamppOperatorDescriptor, attribute_size) == 56u,
    "attribute_size offset does not match the disk ABI");
CAMPP_OPERATOR_STATIC_ASSERT(
    offsetof(CamppOperatorDescriptor, backend_id) == 60u,
    "backend_id offset does not match the disk ABI");
CAMPP_OPERATOR_STATIC_ASSERT(
    offsetof(CamppOperatorDescriptor, kernel_id) == 62u,
    "kernel_id offset does not match the disk ABI");

#undef CAMPP_OPERATOR_STATIC_ASSERT

#endif /* CAMPP_RUNTIME_OPERATOR_DESCRIPTOR_H */
