/*
 * plan 검증 구현. 규칙과 의도는 compiled_model_validator.h를 본다.
 *
 * SHA-256은 이 파일 안에 static으로 둔다. checksum 확인이 유일한 용도이고,
 * 다른 모듈이 해시를 쓸 일이 생기기 전에 공용 모듈로 빼면 쓰이지 않는 API가
 * 하나 늘 뿐이다.
 */

#include "compiled_model_validator.h"

#include <stdlib.h>
#include <string.h>

#include "internal/runtime_model.h"
#include "internal/tensor_view.h"

/* --------------------------------------------------------------------- */
/* SHA-256                                                               */
/* --------------------------------------------------------------------- */

typedef struct CamppSha256 {
    uint32_t state[8];
    uint64_t bit_count;
    uint8_t block[64];
    size_t block_length;
} CamppSha256;

static const uint32_t CAMPP_SHA256_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

static uint32_t campp_rotr32(uint32_t value, unsigned int bits)
{
    return (value >> bits) | (value << (32u - bits));
}

static void campp_sha256_init(CamppSha256 *context)
{
    context->state[0] = 0x6a09e667u;
    context->state[1] = 0xbb67ae85u;
    context->state[2] = 0x3c6ef372u;
    context->state[3] = 0xa54ff53au;
    context->state[4] = 0x510e527fu;
    context->state[5] = 0x9b05688cu;
    context->state[6] = 0x1f83d9abu;
    context->state[7] = 0x5be0cd19u;
    context->bit_count = 0u;
    context->block_length = 0u;
}

static void campp_sha256_compress(CamppSha256 *context, const uint8_t *block)
{
    uint32_t schedule[64];
    uint32_t a, b, c, d, e, f, g, h;
    unsigned int index;

    for (index = 0u; index < 16u; ++index) {
        schedule[index] = ((uint32_t)block[index * 4u] << 24)
            | ((uint32_t)block[index * 4u + 1u] << 16)
            | ((uint32_t)block[index * 4u + 2u] << 8)
            | (uint32_t)block[index * 4u + 3u];
    }
    for (index = 16u; index < 64u; ++index) {
        uint32_t s0 = campp_rotr32(schedule[index - 15u], 7u)
            ^ campp_rotr32(schedule[index - 15u], 18u)
            ^ (schedule[index - 15u] >> 3);
        uint32_t s1 = campp_rotr32(schedule[index - 2u], 17u)
            ^ campp_rotr32(schedule[index - 2u], 19u)
            ^ (schedule[index - 2u] >> 10);
        schedule[index] = schedule[index - 16u] + s0 + schedule[index - 7u] + s1;
    }

    a = context->state[0];
    b = context->state[1];
    c = context->state[2];
    d = context->state[3];
    e = context->state[4];
    f = context->state[5];
    g = context->state[6];
    h = context->state[7];

    for (index = 0u; index < 64u; ++index) {
        uint32_t s1 = campp_rotr32(e, 6u) ^ campp_rotr32(e, 11u)
            ^ campp_rotr32(e, 25u);
        uint32_t choice = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + s1 + choice + CAMPP_SHA256_K[index]
            + schedule[index];
        uint32_t s0 = campp_rotr32(a, 2u) ^ campp_rotr32(a, 13u)
            ^ campp_rotr32(a, 22u);
        uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = s0 + majority;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    context->state[0] += a;
    context->state[1] += b;
    context->state[2] += c;
    context->state[3] += d;
    context->state[4] += e;
    context->state[5] += f;
    context->state[6] += g;
    context->state[7] += h;
}

static void campp_sha256_update(
    CamppSha256 *context, const uint8_t *data, uint64_t size)
{
    uint64_t consumed = 0u;

    context->bit_count += size * 8u;
    while (consumed < size) {
        size_t room = sizeof(context->block) - context->block_length;
        uint64_t remaining = size - consumed;
        size_t take = (remaining < (uint64_t)room) ? (size_t)remaining : room;

        memcpy(context->block + context->block_length, data + consumed, take);
        context->block_length += take;
        consumed += (uint64_t)take;

        if (context->block_length == sizeof(context->block)) {
            campp_sha256_compress(context, context->block);
            context->block_length = 0u;
        }
    }
}

static void campp_sha256_final(CamppSha256 *context, uint8_t digest[32])
{
    uint64_t bit_count = context->bit_count;
    unsigned int index;

    context->block[context->block_length] = 0x80u;
    context->block_length += 1u;
    if (context->block_length > 56u) {
        memset(context->block + context->block_length, 0,
               sizeof(context->block) - context->block_length);
        campp_sha256_compress(context, context->block);
        context->block_length = 0u;
    }
    memset(context->block + context->block_length, 0,
           56u - context->block_length);
    for (index = 0u; index < 8u; ++index) {
        context->block[56u + index] =
            (uint8_t)((bit_count >> (56u - 8u * index)) & 0xffu);
    }
    campp_sha256_compress(context, context->block);

    for (index = 0u; index < 8u; ++index) {
        digest[index * 4u] = (uint8_t)((context->state[index] >> 24) & 0xffu);
        digest[index * 4u + 1u] = (uint8_t)((context->state[index] >> 16) & 0xffu);
        digest[index * 4u + 2u] = (uint8_t)((context->state[index] >> 8) & 0xffu);
        digest[index * 4u + 3u] = (uint8_t)(context->state[index] & 0xffu);
    }
}

/* 길이가 고정이라 이른 종료 없이 전부 비교한다. */
static bool campp_digest_equal(const uint8_t *left, const uint8_t *right)
{
    uint8_t difference = 0u;
    unsigned int index;

    for (index = 0u; index < CAMPP_PLAN_CHECKSUM_SIZE; ++index) {
        difference |= (uint8_t)(left[index] ^ right[index]);
    }
    return difference == 0u;
}

/* --------------------------------------------------------------------- */
/* header와 구역                                                          */
/* --------------------------------------------------------------------- */

static CamppStatus campp_read_plan_header(
    const CamppByteSpan *plan, CamppPlanHeader *out_header)
{
    CamppStatus status;

    status = campp_span_read_bytes(
        plan, CAMPP_PLAN_MAGIC_OFFSET, out_header->magic,
        CAMPP_PLAN_MAGIC_SIZE);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    if (memcmp(out_header->magic, CAMPP_PLAN_MAGIC, CAMPP_PLAN_MAGIC_SIZE) != 0) {
        return CAMPP_STATUS_INVALID_MAGIC;
    }

    status = campp_span_read_u32(
        plan, CAMPP_PLAN_FORMAT_VERSION_OFFSET, &out_header->format_version);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    if (out_header->format_version != CAMPP_PLAN_FORMAT_VERSION) {
        return CAMPP_STATUS_UNSUPPORTED_VERSION;
    }

    status = campp_span_read_u32(
        plan, CAMPP_PLAN_BUCKET_FRAMES_OFFSET, &out_header->bucket_frames);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_span_read_u32(
        plan, CAMPP_PLAN_TENSOR_COUNT_OFFSET, &out_header->tensor_count);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_span_read_u32(
        plan, CAMPP_PLAN_OPERATOR_COUNT_OFFSET, &out_header->operator_count);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_span_read_u64(
        plan, CAMPP_PLAN_TENSOR_TABLE_OFFSET_OFFSET,
        &out_header->tensor_table_offset);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_span_read_u64(
        plan, CAMPP_PLAN_OPERATOR_TABLE_OFFSET_OFFSET,
        &out_header->operator_table_offset);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_span_read_u64(
        plan, CAMPP_PLAN_ATTRIBUTE_SECTION_OFFSET_OFFSET,
        &out_header->attribute_section_offset);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    return campp_span_read_bytes(
        plan, CAMPP_PLAN_CHECKSUM_OFFSET, out_header->checksum,
        CAMPP_PLAN_CHECKSUM_SIZE);
}

static CamppStatus campp_validate_plan_checksum(
    const CamppByteSpan *plan, const CamppPlanHeader *header)
{
    CamppSha256 context;
    uint8_t digest[CAMPP_PLAN_CHECKSUM_SIZE];
    CamppStatus status;
    CamppByteSpan payload;

    status = campp_span_slice(
        plan, CAMPP_PLAN_HEADER_SIZE, plan->size - CAMPP_PLAN_HEADER_SIZE,
        &payload);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }

    campp_sha256_init(&context);
    if (payload.size != 0u) {
        campp_sha256_update(&context, payload.data, payload.size);
    }
    campp_sha256_final(&context, digest);

    if (!campp_digest_equal(digest, header->checksum)) {
        return CAMPP_STATUS_CHECKSUM_MISMATCH;
    }
    return CAMPP_STATUS_OK;
}

CamppStatus campp_validate_plan(
    const CamppByteSpan *plan, CamppPlanLayout *out_layout)
{
    CamppStatus status;
    CamppPlanHeader header;
    uint64_t tensor_bytes;
    uint64_t operator_bytes;
    uint64_t tensor_end;
    uint64_t operator_end;

    if (plan == NULL || out_layout == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    if (plan->size < CAMPP_PLAN_HEADER_SIZE) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }

    memset(&header, 0, sizeof(header));
    status = campp_read_plan_header(plan, &header);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }

    if (header.bucket_frames == 0u || header.tensor_count == 0u
        || header.operator_count == 0u) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }

    status = campp_checked_mul_u64(
        (uint64_t)header.tensor_count, CAMPP_TENSOR_DESCRIPTOR_SIZE,
        &tensor_bytes);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_checked_mul_u64(
        (uint64_t)header.operator_count, CAMPP_OPERATOR_DESCRIPTOR_SIZE,
        &operator_bytes);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }

    /* 세 구역은 header 뒤에서 겹치지 않고 이 순서로 놓인다. */
    if (header.tensor_table_offset < CAMPP_PLAN_HEADER_SIZE) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    status = campp_span_check_aligned(
        plan, header.tensor_table_offset, tensor_bytes,
        CAMPP_PLAN_SECTION_ALIGNMENT);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_checked_add_u64(
        header.tensor_table_offset, tensor_bytes, &tensor_end);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    if (header.operator_table_offset < tensor_end) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    status = campp_span_check_aligned(
        plan, header.operator_table_offset, operator_bytes,
        CAMPP_PLAN_SECTION_ALIGNMENT);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_checked_add_u64(
        header.operator_table_offset, operator_bytes, &operator_end);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    if (header.attribute_section_offset < operator_end) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    status = campp_span_check_aligned(
        plan, header.attribute_section_offset, 0u,
        CAMPP_PLAN_SECTION_ALIGNMENT);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }

    status = campp_validate_plan_checksum(plan, &header);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }

    out_layout->header = header;
    status = campp_span_slice(
        plan, header.tensor_table_offset, tensor_bytes,
        &out_layout->tensor_table);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_span_slice(
        plan, header.operator_table_offset, operator_bytes,
        &out_layout->operator_table);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    return campp_span_slice(
        plan, header.attribute_section_offset,
        plan->size - header.attribute_section_offset,
        &out_layout->attribute_section);
}

/* --------------------------------------------------------------------- */
/* descriptor                                                            */
/* --------------------------------------------------------------------- */

static bool campp_dtype_supported(uint8_t dtype)
{
    return campp_dtype_byte_size(dtype) != 0u;
}

static bool campp_storage_supported(uint8_t storage_type)
{
    return storage_type >= CAMPP_TENSOR_STORAGE_INPUT
        && storage_type <= CAMPP_TENSOR_STORAGE_VIEW;
}

static bool campp_opcode_supported(uint16_t opcode)
{
    return opcode >= CAMPP_OP_QLINEAR_CONV && opcode <= CAMPP_OP_UNSQUEEZE;
}

CamppStatus campp_validate_tensor_descriptor(
    const CamppTensorDescriptor *descriptor, uint32_t expected_id,
    uint32_t tensor_count, uint32_t operator_count)
{
    static const uint8_t known_flags = (uint8_t)(
        CAMPP_TENSOR_FLAG_READ_ONLY | CAMPP_TENSOR_FLAG_CONTIGUOUS
        | CAMPP_TENSOR_FLAG_EXTERNAL | CAMPP_TENSOR_FLAG_ALIASED
        | CAMPP_TENSOR_FLAG_DENSE_SLAB);
    uint64_t elements = 1u;
    uint8_t axis;
    CamppStatus status;

    if (descriptor == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    /* ID는 표에서의 자리와 같아야 한다. 그래야 index로 바로 찾을 수 있다. */
    if (descriptor->tensor_id != expected_id) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    if (!campp_dtype_supported(descriptor->dtype)) {
        return CAMPP_STATUS_UNSUPPORTED_DTYPE;
    }
    if (descriptor->rank > CAMPP_TENSOR_MAX_RANK) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    if (!campp_storage_supported(descriptor->storage_type)) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    if ((descriptor->flags & (uint8_t)~known_flags) != 0u) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }

    for (axis = 0u; axis < CAMPP_TENSOR_MAX_RANK; ++axis) {
        if (axis < descriptor->rank) {
            if (descriptor->dimensions[axis] == 0u) {
                return CAMPP_STATUS_CORRUPT_PLAN;
            }
            status = campp_checked_mul_u64(
                elements, (uint64_t)descriptor->dimensions[axis], &elements);
            if (status != CAMPP_STATUS_OK) {
                return status;
            }
        } else {
            if (descriptor->dimensions[axis] != 1u
                || descriptor->byte_strides[axis] != 0u) {
                return CAMPP_STATUS_CORRUPT_PLAN;
            }
        }
    }

    /* shape와 dtype이 실제로 그 크기를 만드는지 대조한다. */
    {
        uint64_t expected_bytes;

        status = campp_checked_mul_u64(
            elements, (uint64_t)campp_dtype_byte_size(descriptor->dtype),
            &expected_bytes);
        if (status != CAMPP_STATUS_OK) {
            return status;
        }
        if (expected_bytes == 0u
            || descriptor->logical_byte_size != expected_bytes) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
    }
    if (descriptor->storage_span_bytes < descriptor->logical_byte_size) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }

    if (descriptor->storage_type == CAMPP_TENSOR_STORAGE_VIEW) {
        if (descriptor->alias_of_tensor_id >= tensor_count) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
        if ((descriptor->flags & CAMPP_TENSOR_FLAG_ALIASED) == 0u) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
    } else if (descriptor->alias_of_tensor_id != CAMPP_INVALID_TENSOR_ID) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }

    if (descriptor->first_use != CAMPP_INVALID_OPERATOR_INDEX) {
        if (descriptor->first_use >= operator_count) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
    }
    if (descriptor->last_use != CAMPP_INVALID_OPERATOR_INDEX) {
        if (descriptor->last_use >= operator_count) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
    }
    if (descriptor->first_use != CAMPP_INVALID_OPERATOR_INDEX
        && descriptor->last_use != CAMPP_INVALID_OPERATOR_INDEX
        && descriptor->first_use > descriptor->last_use) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    return CAMPP_STATUS_OK;
}

CamppStatus campp_validate_operator_descriptor(
    const CamppOperatorDescriptor *descriptor, uint32_t expected_id,
    uint32_t tensor_count, uint64_t attribute_section_size)
{
    uint8_t slot;
    CamppStatus status;

    if (descriptor == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    if (descriptor->operator_id != expected_id) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    if (!campp_opcode_supported(descriptor->opcode)) {
        return CAMPP_STATUS_UNSUPPORTED_OPCODE;
    }
    if (descriptor->input_count == 0u
        || descriptor->input_count > CAMPP_OPERATOR_INPUT_CAPACITY) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    if (descriptor->output_count != CAMPP_OPERATOR_OUTPUT_CAPACITY) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    if (descriptor->backend_id > CAMPP_BACKEND_GPU_VULKAN) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }

    for (slot = 0u; slot < CAMPP_OPERATOR_INPUT_CAPACITY; ++slot) {
        if (slot < descriptor->input_count) {
            if (descriptor->input_tensor_ids[slot] >= tensor_count) {
                return CAMPP_STATUS_CORRUPT_PLAN;
            }
        } else if (descriptor->input_tensor_ids[slot]
                   != CAMPP_INVALID_TENSOR_ID) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
    }
    if (descriptor->output_tensor_ids[0] >= tensor_count) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }

    if (descriptor->attribute_size == 0u) {
        if (descriptor->attribute_offset != 0u) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
    } else {
        uint64_t end;

        if ((descriptor->attribute_offset
             & (uint64_t)(CAMPP_PLAN_SECTION_ALIGNMENT - 1u)) != 0u) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
        status = campp_checked_add_u64(
            descriptor->attribute_offset, (uint64_t)descriptor->attribute_size,
            &end);
        if (status != CAMPP_STATUS_OK) {
            return status;
        }
        if (end > attribute_section_size) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
    }
    return CAMPP_STATUS_OK;
}

/* --------------------------------------------------------------------- */
/* 표 전체                                                                */
/* --------------------------------------------------------------------- */

CamppStatus campp_validate_model(const struct CamppRuntimeModel *model)
{
    const CamppRuntimeModel *target = model;
    uint32_t index;
    uint8_t slot;
    CamppStatus status;
    uint8_t *produced;

    if (target == NULL || target->tensors == NULL
        || target->operators == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }

    /* CONSTANT는 weights.bin 안을 가리켜야 한다. */
    for (index = 0u; index < target->tensor_count; ++index) {
        const CamppTensorDescriptor *descriptor = &target->tensors[index];
        uint64_t end;

        if (descriptor->storage_type != CAMPP_TENSOR_STORAGE_CONSTANT) {
            continue;
        }
        status = campp_checked_add_u64(
            descriptor->data_offset, descriptor->storage_span_bytes, &end);
        if (status != CAMPP_STATUS_OK) {
            return status;
        }
        if (end > (uint64_t)target->weights_size) {
            return CAMPP_STATUS_CORRUPT_WEIGHTS;
        }
    }

    /*
     * 표를 위에서 아래로 읽었을 때 실행이 성립하는지 본다. exporter가 이미
     * 보장하지만, 여기서 다시 보는 이유는 파일이 도중에 바뀌었을 수 있기
     * 때문이다. 순서가 뒤집힌 plan은 초기화되지 않은 버퍼를 읽으면서도
     * 조용히 끝나므로 실행 전에 잡아야 한다.
     */
    produced = (uint8_t *)calloc((size_t)target->tensor_count, sizeof(uint8_t));
    if (produced == NULL) {
        return CAMPP_STATUS_OUT_OF_MEMORY;
    }
    for (index = 0u; index < target->tensor_count; ++index) {
        uint8_t storage = target->tensors[index].storage_type;

        if (storage == CAMPP_TENSOR_STORAGE_INPUT
            || storage == CAMPP_TENSOR_STORAGE_CONSTANT) {
            produced[index] = 1u;
        }
    }

    status = CAMPP_STATUS_OK;
    for (index = 0u; index < target->operator_count && status == CAMPP_STATUS_OK;
         ++index) {
        const CamppOperatorDescriptor *descriptor = &target->operators[index];

        for (slot = 0u; slot < descriptor->input_count; ++slot) {
            if (produced[descriptor->input_tensor_ids[slot]] == 0u) {
                status = CAMPP_STATUS_CORRUPT_PLAN;
                break;
            }
        }
        if (status == CAMPP_STATUS_OK) {
            uint32_t output_id = descriptor->output_tensor_ids[0];

            if (produced[output_id] != 0u) {
                /* 같은 Tensor를 두 번 생산하면 어느 값이 맞는지 알 수 없다. */
                status = CAMPP_STATUS_CORRUPT_PLAN;
            } else {
                produced[output_id] = 1u;
            }
        }
    }

    if (status == CAMPP_STATUS_OK) {
        for (index = 0u; index < target->tensor_count; ++index) {
            if (produced[index] == 0u) {
                status = CAMPP_STATUS_CORRUPT_PLAN;
                break;
            }
        }
    }

    free(produced);
    return status;
}
