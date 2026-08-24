#include "sha256.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

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
        schedule[index] = schedule[index - 16u] + s0
            + schedule[index - 7u] + s1;
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
        size_t take = remaining < (uint64_t)room ? (size_t)remaining : room;

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

    context->block[context->block_length++] = 0x80u;
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
        digest[index * 4u] = (uint8_t)(context->state[index] >> 24);
        digest[index * 4u + 1u] = (uint8_t)(context->state[index] >> 16);
        digest[index * 4u + 2u] = (uint8_t)(context->state[index] >> 8);
        digest[index * 4u + 3u] = (uint8_t)context->state[index];
    }
}

static bool campp_digest_equal(const uint8_t *left, const uint8_t *right)
{
    uint8_t difference = 0u;
    unsigned int index;

    for (index = 0u; index < CAMPP_PLAN_CHECKSUM_SIZE; ++index) {
        difference |= (uint8_t)(left[index] ^ right[index]);
    }
    return difference == 0u;
}

static int campp_hex_nibble(char value)
{
    if (value >= '0' && value <= '9') {
        return value - '0';
    }
    if (value >= 'a' && value <= 'f') {
        return value - 'a' + 10;
    }
    if (value >= 'A' && value <= 'F') {
        return value - 'A' + 10;
    }
    return -1;
}

CamppStatus campp_validate_sha256(
    const uint8_t *data, uint64_t size,
    const uint8_t expected[CAMPP_PLAN_CHECKSUM_SIZE])
{
    CamppSha256 context;
    uint8_t digest[CAMPP_PLAN_CHECKSUM_SIZE];

    if ((data == NULL && size != 0u) || expected == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    campp_sha256_init(&context);
    if (size != 0u) {
        campp_sha256_update(&context, data, size);
    }
    campp_sha256_final(&context, digest);
    return campp_digest_equal(digest, expected)
        ? CAMPP_STATUS_OK : CAMPP_STATUS_CHECKSUM_MISMATCH;
}

CamppStatus campp_sha256_file(
    const char *path, uint8_t digest[CAMPP_PLAN_CHECKSUM_SIZE])
{
    FILE *file;
    CamppSha256 context;
    uint8_t buffer[64u * 1024u];

    if (path == NULL || digest == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return CAMPP_STATUS_FILE_NOT_FOUND;
    }
    campp_sha256_init(&context);
    for (;;) {
        size_t count = fread(buffer, 1u, sizeof(buffer), file);
        if (count != 0u) {
            campp_sha256_update(&context, buffer, (uint64_t)count);
        }
        if (count != sizeof(buffer)) {
            if (ferror(file) != 0) {
                fclose(file);
                return CAMPP_STATUS_FILE_READ_FAILED;
            }
            break;
        }
    }
    if (fclose(file) != 0) {
        return CAMPP_STATUS_FILE_READ_FAILED;
    }
    campp_sha256_final(&context, digest);
    return CAMPP_STATUS_OK;
}

CamppStatus campp_validate_file_sha256_hex(
    const char *path, const char expected_hex[65])
{
    uint8_t expected[CAMPP_PLAN_CHECKSUM_SIZE];
    uint8_t digest[CAMPP_PLAN_CHECKSUM_SIZE];
    unsigned int index;
    CamppStatus status;

    if (path == NULL || expected_hex == NULL || strlen(expected_hex) != 64u) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    for (index = 0u; index < CAMPP_PLAN_CHECKSUM_SIZE; ++index) {
        int high = campp_hex_nibble(expected_hex[index * 2u]);
        int low = campp_hex_nibble(expected_hex[index * 2u + 1u]);
        if (high < 0 || low < 0) {
            return CAMPP_STATUS_INVALID_ARGUMENT;
        }
        expected[index] = (uint8_t)((high << 4) | low);
    }
    status = campp_sha256_file(path, digest);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    return campp_digest_equal(digest, expected)
        ? CAMPP_STATUS_OK : CAMPP_STATUS_CHECKSUM_MISMATCH;
}
