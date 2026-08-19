#ifndef CAMPP_PROFILL_QCONV_V4_TILE_PLAN_H
#define CAMPP_PROFILL_QCONV_V4_TILE_PLAN_H

#include <stdbool.h>
#include <stdint.h>

#include "qconv_v4_execution_plan.h"
#include "qconv_mac_neon.h"

typedef struct CamppQconvV4TilePlan {
    uint32_t output_coordinates[CAMPP_QCONV_CANDIDATE_TILE][2];
    const uint8_t *input_points[CAMPP_QCONV_CANDIDATE_MAX_KERNEL_ELEMENTS]
        [CAMPP_QCONV_CANDIDATE_TILE];
    uint32_t tile_count;
    CamppQconvV4Path path;
    bool full_spatial_tile;
} CamppQconvV4TilePlan;

CamppStatus campp_qconv_v4_tile_plan_create(
    const CamppQconvV4ExecutionPlan *plan, uint32_t batch,
    uint32_t group_channel, uint32_t tile_start, uint32_t tile_count,
    CamppQconvV4TilePlan *out_tile);

#endif /* CAMPP_PROFILL_QCONV_V4_TILE_PLAN_H */
