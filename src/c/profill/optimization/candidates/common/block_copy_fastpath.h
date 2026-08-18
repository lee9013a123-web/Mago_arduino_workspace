#ifndef CAMPP_PROFILL_BLOCK_COPY_FASTPATH_H
#define CAMPP_PROFILL_BLOCK_COPY_FASTPATH_H

#include <stddef.h>
#include <stdint.h>

void campp_copy_repeated_block(
    const uint8_t *source, size_t block_size, uint8_t *destination,
    size_t destination_stride, uint32_t repeat_count);

#endif /* CAMPP_PROFILL_BLOCK_COPY_FASTPATH_H */
