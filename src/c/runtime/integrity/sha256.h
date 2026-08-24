#ifndef CAMPP_RUNTIME_INTEGRITY_SHA256_H
#define CAMPP_RUNTIME_INTEGRITY_SHA256_H

#include <stdint.h>

#include "campp_runtime/model_binary_format.h"
#include "campp_runtime/status_code.h"

#ifdef __cplusplus
extern "C" {
#endif

/* In-memory plan/package sections and external deployment assets share this. */
CamppStatus campp_validate_sha256(
    const uint8_t *data, uint64_t size,
    const uint8_t expected[CAMPP_PLAN_CHECKSUM_SIZE]);

/* Computes a file digest incrementally, without mapping the whole weight file. */
CamppStatus campp_sha256_file(
    const char *path, uint8_t digest[CAMPP_PLAN_CHECKSUM_SIZE]);

/* Validates a file against the canonical 64-character hexadecimal form. */
CamppStatus campp_validate_file_sha256_hex(
    const char *path, const char expected_hex[65]);

#ifdef __cplusplus
}
#endif

#endif /* CAMPP_RUNTIME_INTEGRITY_SHA256_H */
