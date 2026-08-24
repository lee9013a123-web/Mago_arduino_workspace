#ifndef CAMPP_PROFILL_RUNTIME_FIXTURE_H
#define CAMPP_PROFILL_RUNTIME_FIXTURE_H

#include <stddef.h>
#include <stdint.h>

#include "internal/kernel_registry.h"
#include "internal/runtime_model.h"

int campp_profill_read_entire_file(
    const char *path, uint8_t **out_data, size_t *out_size);

int campp_profill_infer_input_dimensions(
    const CamppRuntimeModel *model,
    const CamppTensorDescriptor *descriptor, size_t input_size,
    uint32_t dimensions[CAMPP_TENSOR_MAX_RANK]);

const CamppKernelRegistry *campp_profill_registry_for_model(
    const CamppRuntimeModel *model);

#endif /* CAMPP_PROFILL_RUNTIME_FIXTURE_H */
