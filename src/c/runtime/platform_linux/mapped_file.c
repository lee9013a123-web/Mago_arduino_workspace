#define _GNU_SOURCE

/* Read-only model mapping and page-cache advice for Linux deployments. */

#include "platform_linux/mapped_file.h"

#include <fcntl.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

CamppStatus campp_mapped_file_open_readonly(
    const char *path, CamppMappedFile *mapping)
{
    struct stat information;
    void *address;
    int descriptor;

    if (path == NULL || mapping == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    memset(mapping, 0, sizeof(*mapping));
    mapping->descriptor = -1;
    descriptor = open(path, O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
        return CAMPP_STATUS_FILE_NOT_FOUND;
    }
    if (fstat(descriptor, &information) != 0 || information.st_size <= 0) {
        close(descriptor);
        return CAMPP_STATUS_FILE_READ_FAILED;
    }
    if ((uintmax_t)information.st_size > (uintmax_t)SIZE_MAX) {
        close(descriptor);
        return CAMPP_STATUS_BUFFER_OVERFLOW;
    }
    address = mmap(
        NULL, (size_t)information.st_size, PROT_READ, MAP_SHARED,
        descriptor, 0);
    if (address == MAP_FAILED) {
        close(descriptor);
        return CAMPP_STATUS_OUT_OF_MEMORY;
    }
    mapping->data = (const uint8_t *)address;
    mapping->size = (size_t)information.st_size;
    mapping->descriptor = descriptor;
    mapping->mapped = true;
    return CAMPP_STATUS_OK;
}

void campp_mapped_file_close(CamppMappedFile *mapping)
{
    if (mapping == NULL) {
        return;
    }
    if (mapping->mapped && mapping->data != NULL && mapping->size != 0u) {
        (void)munmap((void *)(uintptr_t)mapping->data, mapping->size);
    }
    if (mapping->descriptor >= 0) {
        (void)close(mapping->descriptor);
    }
    memset(mapping, 0, sizeof(*mapping));
    mapping->descriptor = -1;
}

CamppStatus campp_mapped_file_advise_sequential(
    const CamppMappedFile *mapping)
{
    if (mapping == NULL || !mapping->mapped || mapping->data == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    return madvise(
        (void *)(uintptr_t)mapping->data,
        mapping->size,
        MADV_SEQUENTIAL) == 0
        ? CAMPP_STATUS_OK : CAMPP_STATUS_FILE_READ_FAILED;
}

/* THP가 [always]면 파일 매핑이 2 MiB PMD로 backing된다.  그러면 128 KiB~512 KiB
 * 단위 MADV_DONTNEED가 huge page를 쪼개지 못해 통째로 상주한 채 남는다.  windowing을
 * 쓰려면 이 매핑만 huge page 대상에서 빼야 4 KiB 단위 반납이 실제로 먹는다. */
CamppStatus campp_mapped_file_advise_no_huge_page(
    const CamppMappedFile *mapping)
{
    if (mapping == NULL || !mapping->mapped || mapping->data == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
#ifdef MADV_NOHUGEPAGE
    if (madvise((void *)(uintptr_t)mapping->data, mapping->size,
                MADV_NOHUGEPAGE) != 0) {
        return CAMPP_STATUS_FILE_READ_FAILED;
    }
#endif
    return CAMPP_STATUS_OK;
}

CamppStatus campp_mapped_file_prefetch(
    const CamppMappedFile *mapping, uint64_t offset, uint64_t size)
{
    if (mapping == NULL || !mapping->mapped || mapping->descriptor < 0 ||
        offset > mapping->size || size > mapping->size - offset) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    return posix_fadvise(
        mapping->descriptor, (off_t)offset, (off_t)size,
        POSIX_FADV_WILLNEED) == 0
        ? CAMPP_STATUS_OK : CAMPP_STATUS_FILE_READ_FAILED;
}

CamppStatus campp_mapped_file_discard(
    const CamppMappedFile *mapping, uint64_t offset, uint64_t size)
{
    if (mapping == NULL || !mapping->mapped || mapping->data == NULL ||
        offset > mapping->size || size > mapping->size - offset) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    return madvise(
        (void *)(uintptr_t)(mapping->data + offset),
        (size_t)size,
        MADV_DONTNEED) == 0
        ? CAMPP_STATUS_OK : CAMPP_STATUS_FILE_READ_FAILED;
}
