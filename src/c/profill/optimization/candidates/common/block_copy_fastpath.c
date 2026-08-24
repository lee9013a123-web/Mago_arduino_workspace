#include "block_copy_fastpath.h"

#include <string.h>

void campp_copy_repeated_block(
    const uint8_t *source, size_t block_size, uint8_t *destination,
    size_t destination_stride, uint32_t repeat_count)
{
    uint32_t repeat;
    for (repeat = 0u; repeat < repeat_count; ++repeat) {
        memcpy(destination + (size_t)repeat * destination_stride,
               source, block_size);
    }
}
