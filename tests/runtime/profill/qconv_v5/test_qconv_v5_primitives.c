#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "instrumentation/qconv_v5_stage_tags.h"
#include "packing/qconv_v5_weight_pack.h"
#include "parameters/qconv_v5_zero_point_fastpath.h"
#include "planning/qconv_v5_validated_raw_plan.h"

#define CHECK_TRUE(expression)                                        \
    do {                                                              \
        if (!(expression)) {                                          \
            fprintf(stderr, "CHECK failed at line %d: %s\n",       \
                    __LINE__, #expression);                           \
            return 1;                                                 \
        }                                                             \
    } while (0)

static int test_zero_point_helpers(void)
{
    const int32_t all_zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int32_t mixed[8] = {0, 0, 1, 0, 0, 0, 0, 0};
    CHECK_TRUE(campp_qconv_v5_weight_zero_all_zero(all_zero, 8u));
    CHECK_TRUE(campp_qconv_v5_weight_zero_all_zero(all_zero, 0u));
    CHECK_TRUE(!campp_qconv_v5_weight_zero_all_zero(mixed, 8u));
    CHECK_TRUE(!campp_qconv_v5_weight_zero_all_zero(NULL, 8u));
    CHECK_TRUE(!campp_qconv_v5_weight_zero_all_zero(all_zero, 9u));
    CHECK_TRUE(
        campp_qconv_v5_correct_bias_for_input_zero(17, 3, -5) == 32);
    return 0;
}

static int test_weight_pack_view(void)
{
    uint8_t bytes[4] = {1u, 2u, 3u, 4u};
    CamppQconvV5WeightPackView view =
        campp_qconv_v5_weight_pack_view(bytes, sizeof(bytes), true);
    CHECK_TRUE(view.data == bytes);
    CHECK_TRUE(view.byte_size == sizeof(bytes));
    CHECK_TRUE(view.symmetric_zero_point);
    return 0;
}

static int test_stage_tags(void)
{
    CHECK_TRUE(strcmp(
        campp_qconv_v5_stage_tag_name(
            CAMPP_QCONV_V5_STAGE_VALIDATED_RAW),
        "validated_raw") == 0);
    CHECK_TRUE(strcmp(
        campp_qconv_v5_stage_tag_name(
            (CamppQconvV5StageTag)99),
        "invalid") == 0);
    return 0;
}

static int test_validated_raw_plan(void)
{
    CamppQconvV4ExecutionPlan plan;
    CamppQconvV4ParameterBlock parameters;
    CamppQconvV5ValidatedRawPlan raw_plan;

    memset(&plan, 0, sizeof(plan));
    memset(&parameters, 0, sizeof(parameters));
    plan.fixed_mac_plan_eligible = true;
    plan.inputs_per_group = 32u;
    parameters.fixed_mac_block_eligible = true;
    parameters.valid_outputs = 8u;

    raw_plan = campp_qconv_v5_validated_raw_plan_create(
        &plan, &parameters);
    CHECK_TRUE(raw_plan.enabled);
    CHECK_TRUE(raw_plan.weight_zero_is_all_zero);
    CHECK_TRUE(raw_plan.input_channels == 32u);
    CHECK_TRUE(raw_plan.valid_outputs == 8u);

    parameters.weight_zero[3] = 1;
    raw_plan = campp_qconv_v5_validated_raw_plan_create(
        &plan, &parameters);
    CHECK_TRUE(raw_plan.enabled);
    CHECK_TRUE(!raw_plan.weight_zero_is_all_zero);

    parameters.valid_outputs = 7u;
    raw_plan = campp_qconv_v5_validated_raw_plan_create(
        &plan, &parameters);
    CHECK_TRUE(!raw_plan.enabled);
    CHECK_TRUE(raw_plan.valid_outputs == 7u);
    return 0;
}

int main(void)
{
    CHECK_TRUE(test_zero_point_helpers() == 0);
    CHECK_TRUE(test_weight_pack_view() == 0);
    CHECK_TRUE(test_stage_tags() == 0);
    CHECK_TRUE(test_validated_raw_plan() == 0);
    puts("QConv v5 primitives: PASS");
    return 0;
}
