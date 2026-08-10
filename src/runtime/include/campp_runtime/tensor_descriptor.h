#ifndef CAMPP_RUNTIME_TENSOR_DESCRIPTOR_H
#define CAMPP_RUNTIME_TENSOR_DESCRIPTOR_H

/*
 * Python binary_format_schema.py와 공유하는 Tensor descriptor ABI다.
 *
 * descriptor에는 실제 data pointer나 Tensor 이름을 저장하지 않는다.
 * 저장 위치는 storage_type과 data_offset으로 표현하고 Runtime이 로드 시
 * weights base, arena base 또는 alias Tensor 주소와 결합한다.
 */

#include <stddef.h>
#include <stdint.h>

#define CAMPP_TENSOR_MAX_RANK 4u
#define CAMPP_TENSOR_DESCRIPTOR_SIZE 80u
#define CAMPP_INVALID_TENSOR_ID UINT32_MAX
#define CAMPP_INVALID_QUANTIZATION_INDEX UINT32_MAX
#define CAMPP_INVALID_OPERATOR_INDEX UINT32_MAX
#define CAMPP_INVALID_DATA_OFFSET UINT64_MAX

/* Python TensorDType와 숫자값을 동일하게 유지한다. */
typedef enum CamppTensorDType {
    CAMPP_DTYPE_INVALID = 0,
    CAMPP_DTYPE_FLOAT32 = 1,
    CAMPP_DTYPE_UINT8 = 2,
    CAMPP_DTYPE_INT8 = 3,
    CAMPP_DTYPE_INT32 = 4,
    CAMPP_DTYPE_INT64 = 5,
    CAMPP_DTYPE_BOOL = 6,
    CAMPP_DTYPE_FLOAT16 = 7
} CamppTensorDType;

/* data_offset의 기준 주소를 결정한다. */
typedef enum CamppTensorStorageType {
    CAMPP_TENSOR_STORAGE_INVALID = 0,
    CAMPP_TENSOR_STORAGE_INPUT = 1,
    CAMPP_TENSOR_STORAGE_OUTPUT = 2,
    CAMPP_TENSOR_STORAGE_CONSTANT = 3,
    CAMPP_TENSOR_STORAGE_ACTIVATION = 4,
    CAMPP_TENSOR_STORAGE_VIEW = 5
} CamppTensorStorageType;

/* 하나의 uint8_t에 조합하여 저장하는 Tensor 속성이다. */
typedef enum CamppTensorFlags {
    CAMPP_TENSOR_FLAG_NONE = 0,
    CAMPP_TENSOR_FLAG_READ_ONLY = 1u << 0,
    CAMPP_TENSOR_FLAG_CONTIGUOUS = 1u << 1,
    CAMPP_TENSOR_FLAG_EXTERNAL = 1u << 2,
    CAMPP_TENSOR_FLAG_ALIASED = 1u << 3,
    CAMPP_TENSOR_FLAG_DENSE_SLAB = 1u << 4
} CamppTensorFlags;

/*
 * 고정 80바이트 Tensor descriptor.
 *
 * dimensions에서 rank 이후의 값은 1, byte_strides의 미사용 값은 0이다.
 * byte_strides는 원소 단위가 아닌 byte 단위다. Expand view는 stride 0을
 * 사용할 수 있고, Transpose view는 dimensions와 다른 순서의 stride를 쓴다.
 *
 * data_offset 의미:
 * - CONSTANT: weights section 시작부터의 byte offset
 * - ACTIVATION/OUTPUT: Tensor Arena 시작부터의 byte offset
 * - VIEW: alias_of_tensor_id가 가리키는 Tensor 시작부터의 byte offset
 * - INPUT: 외부에서 pointer를 연결하므로 CAMPP_INVALID_DATA_OFFSET
 */
typedef struct CamppTensorDescriptor {
    uint32_t tensor_id;
    uint8_t dtype;
    uint8_t rank;
    uint8_t storage_type;
    uint8_t flags;
    uint32_t dimensions[CAMPP_TENSOR_MAX_RANK];
    uint32_t byte_strides[CAMPP_TENSOR_MAX_RANK];
    uint64_t data_offset;
    uint64_t logical_byte_size;
    uint64_t storage_span_bytes;
    uint32_t alias_of_tensor_id;
    uint32_t quantization_index;
    uint32_t first_use;
    uint32_t last_use;
} CamppTensorDescriptor;

#if defined(__cplusplus)
#define CAMPP_TENSOR_STATIC_ASSERT(condition, message) static_assert(condition, message)
#else
#define CAMPP_TENSOR_STATIC_ASSERT(condition, message) _Static_assert(condition, message)
#endif

CAMPP_TENSOR_STATIC_ASSERT(
    sizeof(CamppTensorDescriptor) == CAMPP_TENSOR_DESCRIPTOR_SIZE,
    "CamppTensorDescriptor must remain exactly 80 bytes");
CAMPP_TENSOR_STATIC_ASSERT(
    offsetof(CamppTensorDescriptor, tensor_id) == 0u,
    "tensor_id offset does not match the disk ABI");
CAMPP_TENSOR_STATIC_ASSERT(
    offsetof(CamppTensorDescriptor, dtype) == 4u,
    "dtype offset does not match the disk ABI");
CAMPP_TENSOR_STATIC_ASSERT(
    offsetof(CamppTensorDescriptor, rank) == 5u,
    "rank offset does not match the disk ABI");
CAMPP_TENSOR_STATIC_ASSERT(
    offsetof(CamppTensorDescriptor, storage_type) == 6u,
    "storage_type offset does not match the disk ABI");
CAMPP_TENSOR_STATIC_ASSERT(
    offsetof(CamppTensorDescriptor, flags) == 7u,
    "flags offset does not match the disk ABI");
CAMPP_TENSOR_STATIC_ASSERT(
    offsetof(CamppTensorDescriptor, dimensions) == 8u,
    "dimensions offset does not match the disk ABI");
CAMPP_TENSOR_STATIC_ASSERT(
    offsetof(CamppTensorDescriptor, byte_strides) == 24u,
    "byte_strides offset does not match the disk ABI");
CAMPP_TENSOR_STATIC_ASSERT(
    offsetof(CamppTensorDescriptor, data_offset) == 40u,
    "data_offset offset does not match the disk ABI");
CAMPP_TENSOR_STATIC_ASSERT(
    offsetof(CamppTensorDescriptor, logical_byte_size) == 48u,
    "logical_byte_size offset does not match the disk ABI");
CAMPP_TENSOR_STATIC_ASSERT(
    offsetof(CamppTensorDescriptor, storage_span_bytes) == 56u,
    "storage_span_bytes offset does not match the disk ABI");
CAMPP_TENSOR_STATIC_ASSERT(
    offsetof(CamppTensorDescriptor, alias_of_tensor_id) == 64u,
    "alias_of_tensor_id offset does not match the disk ABI");
CAMPP_TENSOR_STATIC_ASSERT(
    offsetof(CamppTensorDescriptor, quantization_index) == 68u,
    "quantization_index offset does not match the disk ABI");
CAMPP_TENSOR_STATIC_ASSERT(
    offsetof(CamppTensorDescriptor, first_use) == 72u,
    "first_use offset does not match the disk ABI");
CAMPP_TENSOR_STATIC_ASSERT(
    offsetof(CamppTensorDescriptor, last_use) == 76u,
    "last_use offset does not match the disk ABI");

#undef CAMPP_TENSOR_STATIC_ASSERT

#endif /* CAMPP_RUNTIME_TENSOR_DESCRIPTOR_H */
