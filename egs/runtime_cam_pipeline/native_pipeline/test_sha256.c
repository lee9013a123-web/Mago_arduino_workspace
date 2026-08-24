#include "integrity/sha256.h"

#include <stdio.h>
#include <string.h>
#include <unistd.h>

int main(void)
{
    static const char expected[] =
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad";
    static const uint8_t expected_bytes[32] = {
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
        0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
        0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
        0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad
    };
    char path[128];
    FILE *file;

    if (campp_validate_sha256(
            (const uint8_t *)"abc", 3u, expected_bytes) != CAMPP_STATUS_OK) {
        fputs("FAIL in-memory SHA-256 vector\n", stderr);
        return 1;
    }
    snprintf(path, sizeof(path), "/tmp/campp_sha256_%ld.bin", (long)getpid());
    file = fopen(path, "wb");
    if (file == NULL || fwrite("abc", 1u, 3u, file) != 3u || fclose(file) != 0) {
        fputs("FAIL SHA-256 test file setup\n", stderr);
        return 1;
    }
    if (campp_validate_file_sha256_hex(path, expected) != CAMPP_STATUS_OK) {
        remove(path);
        fputs("FAIL streaming SHA-256 vector\n", stderr);
        return 1;
    }
    if (campp_validate_file_sha256_hex(
            path,
            "aa7816bf8f01cfea414140de5dae2223"
            "b00361a396177a9cb410ff61f20015ad") !=
            CAMPP_STATUS_CHECKSUM_MISMATCH) {
        remove(path);
        fputs("FAIL SHA-256 mismatch gate\n", stderr);
        return 1;
    }
    remove(path);
    puts("PASS test_sha256");
    return 0;
}
