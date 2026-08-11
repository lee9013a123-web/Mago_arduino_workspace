#ifndef CAMPP_RUNTIME_MODEL_LOADING_COMPILED_MODEL_VALIDATOR_H
#define CAMPP_RUNTIME_MODEL_LOADING_COMPILED_MODEL_VALIDATOR_H

/*
 * plan 바이트가 실행해도 되는 형태인지 확인한다.
 *
 * 원칙은 하나다. 파일을 믿지 않는다. loader가 descriptor를 풀기 전에 header와
 * 구역 경계를 확인하고, 푼 다음에는 표 안의 모든 ID와 offset이 자기 범위 안에
 * 있는지 확인한다. 여기서 걸러내지 못한 값은 executor에서 곧바로 잘못된 주소
 * 접근이 된다.
 *
 * 검사는 로드 시점에 한 번만 한다. 실행 중에는 다시 확인하지 않는다.
 */

#include <stdint.h>

#include "binary_section_reader.h"
#include "campp_runtime/model_binary_format.h"
#include "campp_runtime/operator_descriptor.h"
#include "campp_runtime/status_code.h"
#include "campp_runtime/tensor_descriptor.h"

struct CamppRuntimeModel;

/* validator가 확인해 준 구역 위치. loader는 이 값만 믿고 표를 푼다. */
typedef struct CamppPlanLayout {
    CamppPlanHeader header;
    CamppByteSpan tensor_table;
    CamppByteSpan operator_table;
    CamppByteSpan attribute_section;
} CamppPlanLayout;

/*
 * header를 읽고 magic, format version, 구역 경계, checksum까지 확인한다.
 *
 * checksum은 header 뒤 payload 전체의 SHA-256이다. 계산 비용이 있지만 로드
 * 시점 한 번이고, 이 검사가 없으면 잘린 파일과 온전한 파일을 구분할 수 없다.
 */
CamppStatus campp_validate_plan(
    const CamppByteSpan *plan, CamppPlanLayout *out_layout);

/* 표를 푼 뒤 descriptor 하나가 자기 규칙을 지키는지 본다. */
CamppStatus campp_validate_tensor_descriptor(
    const CamppTensorDescriptor *descriptor, uint32_t expected_id,
    uint32_t tensor_count, uint32_t operator_count);

CamppStatus campp_validate_operator_descriptor(
    const CamppOperatorDescriptor *descriptor, uint32_t expected_id,
    uint32_t tensor_count, uint64_t attribute_section_size);

/*
 * 표 전체의 관계를 본다.
 *
 * descriptor 하나만으로는 알 수 없는 것들을 여기서 확인한다. CONSTANT가
 * weights.bin 안을 가리키는지, 모든 Tensor를 누군가 생산하는지, 그리고 어떤
 * operator도 아직 만들어지지 않은 Tensor를 읽지 않는지다. 마지막 검사가 없으면
 * 순서가 뒤집힌 plan이 쓰레기 값을 읽으면서도 조용히 끝난다.
 */
CamppStatus campp_validate_model(const struct CamppRuntimeModel *model);

#endif /* CAMPP_RUNTIME_MODEL_LOADING_COMPILED_MODEL_VALIDATOR_H */
