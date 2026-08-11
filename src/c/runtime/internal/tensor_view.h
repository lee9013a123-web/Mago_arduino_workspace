#ifndef CAMPP_RUNTIME_INTERNAL_TENSOR_VIEW_H
#define CAMPP_RUNTIME_INTERNAL_TENSOR_VIEW_H

/*
 * Kernel이 실제로 받는 Tensor 표현이다.
 *
 * CamppTensorDescriptor는 디스크 위의 정적 기술이라 data pointer가 없고
 * data_offset과 storage_type만 들고 있다. kernel이 그때마다 weights base나
 * activation buffer를 찾아 더하게 하면 storage 규칙이 kernel마다 흩어진다.
 * 그래서 executor가 descriptor를 한 번 풀어 pointer가 박힌 view를 만들고,
 * kernel은 실제 Tensor 접근에는 view만 사용한다. Tensor ID와 storage 종류에
 * 따른 주소 해석은 executor에서 끝나며 kernel마다 반복하지 않는다.
 *
 * view는 값 타입이다. 실행 중에 executor가 채워 넣고 kernel은 읽기만 한다.
 * 소유권은 없다. data가 가리키는 메모리는 model이나 context가 소유한다.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "campp_runtime/tensor_descriptor.h"

/*
 * 하나의 Tensor를 가리키는 view.
 *
 * dimensions와 byte_strides는 descriptor와 같은 규칙을 따른다. rank 이후의
 * dimensions는 1, byte_strides는 0이다. byte_strides는 원소 수가 아니라
 * byte 단위이므로 kernel은 dtype 크기를 다시 곱하지 않는다.
 *
 * 입력은 const CamppTensorView *로 넘긴다. view 자체는 kernel이 바꿀 수 없고,
 * 출력 view의 data만 쓰기 대상이다.
 *
 * flags의 CAMPP_TENSOR_FLAG_CONTIGUOUS는 executor가
 * campp_tensor_view_is_contiguous()로 직접 계산해 넣는다. descriptor의 값을
 * 그대로 복사하지 않는다. 연속성의 근거를 stride 하나로 두어야 flag와 실제
 * stride가 어긋난 view가 생기지 않는다.
 */
typedef struct CamppTensorView {
    void *data;
    uint8_t dtype;
    uint8_t rank;
    uint8_t flags;
    /* 명시적 padding. 항상 0으로 채운다. 나중에 필드로 승격할 자리다. */
    uint8_t reserved;
    uint32_t dimensions[CAMPP_TENSOR_MAX_RANK];
    uint32_t byte_strides[CAMPP_TENSOR_MAX_RANK];
    /* shape와 dtype으로 표현되는 Tensor 전체의 논리적 크기. */
    uint64_t logical_byte_size;
    /* data부터 실제로 접근 가능한 저장 영역. 비연속 VIEW의 범위 검사에 쓴다. */
    uint64_t storage_span_bytes;
} CamppTensorView;

/* dtype 하나가 차지하는 byte 수. 알 수 없는 dtype이면 0을 돌려준다. */
static inline uint32_t campp_dtype_byte_size(uint8_t dtype)
{
    switch (dtype) {
    case CAMPP_DTYPE_FLOAT32: return 4u;
    case CAMPP_DTYPE_UINT8: return 1u;
    case CAMPP_DTYPE_INT8: return 1u;
    case CAMPP_DTYPE_INT32: return 4u;
    case CAMPP_DTYPE_INT64: return 8u;
    case CAMPP_DTYPE_BOOL: return 1u;
    case CAMPP_DTYPE_FLOAT16: return 2u;
    default: return 0u;
    }
}

/* rank까지의 dimensions를 곱한 원소 수. rank 0 스칼라는 1이다. */
static inline uint64_t campp_tensor_view_element_count(const CamppTensorView *view)
{
    uint64_t count = 1u;
    uint8_t axis;

    if (view == NULL) {
        return 0u;
    }
    for (axis = 0u; axis < view->rank; ++axis) {
        count *= (uint64_t)view->dimensions[axis];
    }
    return count;
}

/* 마지막 축이 촘촘하고 앞 축이 그 배수면 연속이다. */
static inline bool campp_tensor_view_is_contiguous(const CamppTensorView *view)
{
    uint32_t expected;
    uint8_t axis;

    if (view == NULL) {
        return false;
    }
    expected = campp_dtype_byte_size(view->dtype);
    if (expected == 0u) {
        return false;
    }
    for (axis = view->rank; axis > 0u; --axis) {
        if (view->byte_strides[axis - 1u] != expected) {
            return false;
        }
        expected *= view->dimensions[axis - 1u];
    }
    return true;
}

/* 축별 index를 byte offset으로 바꾼다. 범위 검사는 호출자 몫이다. */
static inline uint64_t campp_tensor_view_byte_offset(
    const CamppTensorView *view, const uint32_t *indices)
{
    uint64_t offset = 0u;
    uint8_t axis;

    if (view == NULL || indices == NULL) {
        return 0u;
    }
    for (axis = 0u; axis < view->rank; ++axis) {
        offset += (uint64_t)indices[axis] * (uint64_t)view->byte_strides[axis];
    }
    return offset;
}

/* 두 view의 dtype과 dimensions가 같은지 본다. stride는 보지 않는다. */
static inline bool campp_tensor_view_same_shape(
    const CamppTensorView *first, const CamppTensorView *second)
{
    uint8_t axis;

    if (first == NULL || second == NULL) {
        return false;
    }
    if (first->dtype != second->dtype || first->rank != second->rank) {
        return false;
    }
    for (axis = 0u; axis < first->rank; ++axis) {
        if (first->dimensions[axis] != second->dimensions[axis]) {
            return false;
        }
    }
    return true;
}

#endif /* CAMPP_RUNTIME_INTERNAL_TENSOR_VIEW_H */
