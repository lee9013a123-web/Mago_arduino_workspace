/*
 * Reference Runtime의 첫 실행 프로그램이다.
 *
 * 이 단계에서는 추론하지 않는다. Python이 만든 bundle을 C가 같은 값으로 읽는지
 * 확인하는 것이 전부다. 그래서 출력은 사람이 읽는 요약과, Python 쪽 dump와
 * 그대로 비교할 수 있는 descriptor 목록 두 가지다.
 *
 *   campp_compiled_model_inspect <plan.bin> <weights.bin>
 *   campp_compiled_model_inspect <plan.bin> <weights.bin> --dump-tensors
 *   campp_compiled_model_inspect <plan.bin> <weights.bin> --dump-operators
 *
 * plan에는 Tensor 이름이 없다. 이름은 exporter와 manifest에만 있고 Runtime은
 * 정수 ID로만 Tensor를 찾는다. 그래서 입출력 Tensor는 이름 대신 ID와 shape로
 * 밝힌다. 이름을 찍으려면 형식에 문자열 구역을 넣어야 하는데, 보드에서 쓰지
 * 않을 데이터를 매 plan마다 싣게 된다.
 */

#include <stdio.h>
#include <inttypes.h>
#include <string.h>

#include "internal/runtime_model.h"
#include "internal/tensor_view.h"

static const char *campp_dtype_name(uint8_t dtype)
{
    switch (dtype) {
    case CAMPP_DTYPE_FLOAT32: return "FLOAT32";
    case CAMPP_DTYPE_UINT8: return "UINT8";
    case CAMPP_DTYPE_INT8: return "INT8";
    case CAMPP_DTYPE_INT32: return "INT32";
    case CAMPP_DTYPE_INT64: return "INT64";
    case CAMPP_DTYPE_BOOL: return "BOOL";
    case CAMPP_DTYPE_FLOAT16: return "FLOAT16";
    default: return "INVALID";
    }
}

static const char *campp_storage_name(uint8_t storage_type)
{
    switch (storage_type) {
    case CAMPP_TENSOR_STORAGE_INPUT: return "INPUT";
    case CAMPP_TENSOR_STORAGE_OUTPUT: return "OUTPUT";
    case CAMPP_TENSOR_STORAGE_CONSTANT: return "CONSTANT";
    case CAMPP_TENSOR_STORAGE_ACTIVATION: return "ACTIVATION";
    case CAMPP_TENSOR_STORAGE_VIEW: return "VIEW";
    default: return "INVALID";
    }
}

static void campp_print_shape(const CamppTensorDescriptor *descriptor)
{
    uint8_t axis;

    printf("[");
    for (axis = 0u; axis < descriptor->rank; ++axis) {
        printf("%s%" PRIu32, (axis == 0u) ? "" : ", ",
               descriptor->dimensions[axis]);
    }
    printf("]");
}

static void campp_print_summary(const CamppRuntimeModel *model)
{
    uint64_t constant_bytes = 0u;
    uint32_t index;

    for (index = 0u; index < model->tensor_count; ++index) {
        if (model->tensors[index].storage_type
            == CAMPP_TENSOR_STORAGE_CONSTANT) {
            constant_bytes += model->tensors[index].logical_byte_size;
        }
    }

    printf("Loaded plan: %" PRIu32 " frames\n", model->bucket_frames);
    printf("Tensor count: %" PRIu32 "\n", model->tensor_count);
    printf("Operator count: %" PRIu32 "\n", model->operator_count);
    printf("Weight bytes: %zu\n", model->weights_size);
    printf("Constant bytes in use: %" PRIu64 "\n", constant_bytes);
    printf("Attribute section bytes: %zu\n", model->attribute_section_size);

    for (index = 0u; index < model->input_count; ++index) {
        const CamppTensorDescriptor *descriptor =
            &model->tensors[model->input_tensor_ids[index]];

        printf("Input tensor: id %" PRIu32 " %s ",
               descriptor->tensor_id,
               campp_dtype_name(descriptor->dtype));
        campp_print_shape(descriptor);
        printf(" %" PRIu64 " bytes\n", descriptor->logical_byte_size);
    }
    for (index = 0u; index < model->output_count; ++index) {
        const CamppTensorDescriptor *descriptor =
            &model->tensors[model->output_tensor_ids[index]];

        printf("Output tensor: id %" PRIu32 " %s ",
               descriptor->tensor_id,
               campp_dtype_name(descriptor->dtype));
        campp_print_shape(descriptor);
        printf(" %" PRIu64 " bytes\n", descriptor->logical_byte_size);
    }
}

/* Python dump와 문자 단위로 같아야 하는 줄이다. 형식을 바꾸면 양쪽을 함께 바꾼다. */
static void campp_dump_tensors(const CamppRuntimeModel *model)
{
    uint32_t index;
    uint8_t axis;

    for (index = 0u; index < model->tensor_count; ++index) {
        const CamppTensorDescriptor *descriptor = &model->tensors[index];

        printf("T %" PRIu32 " %s rank=%u %s flags=0x%02x dims=",
               descriptor->tensor_id,
               campp_dtype_name(descriptor->dtype),
               (unsigned int)descriptor->rank,
               campp_storage_name(descriptor->storage_type),
               (unsigned int)descriptor->flags);
        for (axis = 0u; axis < CAMPP_TENSOR_MAX_RANK; ++axis) {
            printf("%s%" PRIu32, (axis == 0u) ? "" : ",",
                   descriptor->dimensions[axis]);
        }
        printf(" strides=");
        for (axis = 0u; axis < CAMPP_TENSOR_MAX_RANK; ++axis) {
            printf("%s%" PRIu32, (axis == 0u) ? "" : ",",
                   descriptor->byte_strides[axis]);
        }
        printf(" offset=%" PRIu64 " logical=%" PRIu64 " span=%" PRIu64
               " alias=%" PRIu32 " quant=%" PRIu32
               " first=%" PRIu32 " last=%" PRIu32 "\n",
               descriptor->data_offset,
               descriptor->logical_byte_size,
               descriptor->storage_span_bytes,
               descriptor->alias_of_tensor_id,
               descriptor->quantization_index,
               descriptor->first_use,
               descriptor->last_use);
    }
}

static void campp_dump_operators(const CamppRuntimeModel *model)
{
    uint32_t index;
    uint8_t slot;

    for (index = 0u; index < model->operator_count; ++index) {
        const CamppOperatorDescriptor *descriptor = &model->operators[index];

        printf("O %" PRIu32 " opcode=%u backend=%u kernel=%u in=",
               descriptor->operator_id,
               (unsigned int)descriptor->opcode,
               (unsigned int)descriptor->backend_id,
               (unsigned int)descriptor->kernel_id);
        for (slot = 0u; slot < descriptor->input_count; ++slot) {
            printf("%s%" PRIu32, (slot == 0u) ? "" : ",",
                   descriptor->input_tensor_ids[slot]);
        }
        printf(" out=%" PRIu32 " attr=%" PRIu32 "@%" PRIu64 "\n",
               descriptor->output_tensor_ids[0],
               descriptor->attribute_size,
               descriptor->attribute_offset);
    }
}

static void campp_print_usage(const char *program)
{
    fprintf(stderr,
            "usage: %s <plan.bin> <weights.bin>"
            " [--dump-tensors] [--dump-operators]\n",
            (program == NULL) ? "campp_compiled_model_inspect" : program);
}

int main(int argc, char **argv)
{
    CamppRuntimeModel model;
    CamppStatus status;
    int dump_tensors = 0;
    int dump_operators = 0;
    int index;

    if (argc < 3) {
        campp_print_usage(argv[0]);
        return 2;
    }
    for (index = 3; index < argc; ++index) {
        if (strcmp(argv[index], "--dump-tensors") == 0) {
            dump_tensors = 1;
        } else if (strcmp(argv[index], "--dump-operators") == 0) {
            dump_operators = 1;
        } else {
            fprintf(stderr, "unknown option: %s\n", argv[index]);
            campp_print_usage(argv[0]);
            return 2;
        }
    }

    memset(&model, 0, sizeof(model));
    status = campp_runtime_model_load(argv[1], argv[2], &model);
    if (status != CAMPP_STATUS_OK) {
        fprintf(stderr, "load failed: %s (%d)\n", campp_status_name(status),
                (int)status);
        return 1;
    }

    if (dump_tensors) {
        campp_dump_tensors(&model);
    }
    if (dump_operators) {
        campp_dump_operators(&model);
    }
    if (!dump_tensors && !dump_operators) {
        campp_print_summary(&model);
    }

    campp_runtime_model_release(&model);
    return 0;
}
