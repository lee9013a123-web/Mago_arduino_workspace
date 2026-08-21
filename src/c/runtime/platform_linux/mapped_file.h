#ifndef CAMPP_RUNTIME_PLATFORM_LINUX_MAPPED_FILE_H
#define CAMPP_RUNTIME_PLATFORM_LINUX_MAPPED_FILE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "campp_runtime/status_code.h"

typedef struct CamppMappedFile {
    const uint8_t *data;
    size_t size;
    int descriptor;
    bool mapped;
} CamppMappedFile;

CamppStatus campp_mapped_file_open_readonly(
    const char *path, CamppMappedFile *mapping);
void campp_mapped_file_close(CamppMappedFile *mapping);
CamppStatus campp_mapped_file_advise_sequential(
    const CamppMappedFile *mapping);
CamppStatus campp_mapped_file_prefetch(
    const CamppMappedFile *mapping, uint64_t offset, uint64_t size);
CamppStatus campp_mapped_file_discard(
    const CamppMappedFile *mapping, uint64_t offset, uint64_t size);

#endif /* CAMPP_RUNTIME_PLATFORM_LINUX_MAPPED_FILE_H */
