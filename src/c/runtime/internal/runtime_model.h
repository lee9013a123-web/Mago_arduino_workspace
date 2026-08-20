#ifndef CAMPP_RUNTIME_INTERNAL_RUNTIME_MODEL_H
#define CAMPP_RUNTIME_INTERNAL_RUNTIME_MODEL_H

/*
 * 로드가 끝난 정적 모델이다.
 *
 * 여기 담긴 값은 로드 이후 절대 바뀌지 않는다. inference를 몇 번 돌리든,
 * 몇 개의 thread가 동시에 돌리든 model은 읽기 전용이다. 바뀌는 것은 모두
 * CamppRuntimeContext에 있다. 이 경계가 흐려지면 bucket을 바꿔 끼우거나
 * 같은 model로 context 여러 개를 돌릴 수 없게 된다.
 *
 * plan_*.bin 한 개와 weights.bin 한 개가 model 하나를 이룬다. weights.bin은
 * 모든 bucket이 공유하므로, bucket을 바꾸려면 plan만 다시 로드하면 된다.
 *
 * 디스크 바이트를 구조체로 직접 cast하지 않는다. loader가 little-endian
 * reader로 읽어 아래 배열을 채우고, model이 그 배열을 소유한다. 그래서 plan
 * 파일의 정렬이나 host의 byte order와 무관하게 동작한다.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "campp_runtime/model_binary_format.h"
#include "campp_runtime/operator_descriptor.h"
#include "campp_runtime/status_code.h"
#include "campp_runtime/tensor_descriptor.h"

/* 한 bucket의 실행 계획과 그 계획이 참조하는 weight 전체. */
typedef struct CamppRuntimeModel {
    /* --- 원본 바이트 --- */
    /* plan 파일 전체. attribute section이 이 안을 가리키므로 살아 있어야 한다. */
    uint8_t *plan_bytes;
    size_t plan_size;
    bool owns_plan_bytes;

    /* weights.bin. mmap한 경우 owns_weights는 false이고 해제는 mapper가 한다. */
    const uint8_t *weights;
    size_t weights_size;
    bool owns_weights;

    /* --- header에서 읽은 값 --- */
    CamppPlanHeader header;
    uint32_t bucket_frames;

    /* --- 디코드된 표 (model이 소유) --- */
    CamppTensorDescriptor *tensors;
    uint32_t tensor_count;
    CamppOperatorDescriptor *operators;
    uint32_t operator_count;

    /* --- plan_bytes 안을 가리키는 뷰 (소유하지 않음) --- */
    const uint8_t *attribute_section;
    size_t attribute_section_size;

    /* --- graph 경계 --- */
    /* plan에는 이름이 없으므로 외부와 주고받을 창구는 이 ID들뿐이다. */
    uint32_t *input_tensor_ids;
    uint32_t input_count;
    uint32_t *output_tensor_ids;
    uint32_t output_count;
} CamppRuntimeModel;

/* 되읽은 attribute 값 하나. 값은 종류와 무관하게 8바이트로 저장되어 있다. */
typedef struct CamppAttributeValues {
    uint16_t key;
    uint8_t value_type;
    uint8_t count;
    const uint8_t *raw;
} CamppAttributeValues;

/*
 * 로드와 해제.
 *
 * campp_runtime_model_load는 plan과 weights를 읽고, magic, version, checksum,
 * 표의 정합성까지 확인한 뒤에야 성공을 돌려준다. 실패하면 model은 손대지 않은
 * 상태로 남고 부분적으로 할당된 자원은 스스로 정리한다.
 */
CamppStatus campp_runtime_model_load(
    const char *plan_path, const char *weights_path, CamppRuntimeModel *model);

/* Versioned .camppmodel container에서 plan과 weights를 함께 로드한다. */
CamppStatus campp_runtime_model_load_package(
    const char *package_path, CamppRuntimeModel *model);

CamppStatus campp_runtime_model_adopt(
    uint8_t *plan_bytes, size_t plan_size, bool take_plan_ownership,
    const uint8_t *weights, size_t weights_size, bool take_weights_ownership,
    CamppRuntimeModel *model);

void campp_runtime_model_release(CamppRuntimeModel *model);

/*
 * 조회.
 *
 * ID는 plan이 정한 조밀한 정수라 배열 index로 바로 쓰지만, 범위를 벗어난 ID를
 * 조용히 넘기지 않도록 항상 검사한다.
 */
CamppStatus campp_runtime_model_tensor(
    const CamppRuntimeModel *model, uint32_t tensor_id,
    const CamppTensorDescriptor **out_descriptor);

CamppStatus campp_runtime_model_operator(
    const CamppRuntimeModel *model, uint32_t operator_id,
    const CamppOperatorDescriptor **out_descriptor);

/*
 * CONSTANT Tensor의 실제 주소를 돌려준다.
 *
 * data_offset은 weights.bin 시작 기준이고, bucket 전용 상수는 bucket마다 다른
 * offset을 가리킨다. 그 차이는 plan이 이미 반영해 두었으므로 여기서는 bucket을
 * 따질 필요가 없다. CONSTANT가 아닌 Tensor면 CAMPP_STATUS_INVALID_ARGUMENT다.
 */
CamppStatus campp_runtime_model_constant_data(
    const CamppRuntimeModel *model, const CamppTensorDescriptor *descriptor,
    const void **out_data);

/*
 * Operator의 attribute를 읽는다.
 *
 * attribute가 없는 operator에 대해 조회하면 CAMPP_STATUS_MISSING_ATTRIBUTE다.
 * kernel은 자기가 필요한 key만 뽑아 쓰고, 없는 key는 오류로 다룬다. exporter가
 * ONNX 기본값까지 명시적으로 채워 넣으므로 kernel이 기본값을 추론할 일은 없다.
 */
CamppStatus campp_runtime_model_attribute(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    uint16_t key, CamppAttributeValues *out_values);

CamppStatus campp_runtime_model_attribute_ints(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    uint16_t key, int64_t *values, uint8_t capacity, uint8_t *out_count);

CamppStatus campp_runtime_model_attribute_double(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    uint16_t key, double *out_value);

#endif /* CAMPP_RUNTIME_INTERNAL_RUNTIME_MODEL_H */
