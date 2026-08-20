#include <stdio.h>
#include <string.h>

#include "conv_layer_hybrid_plan.h"

#define CHECK_TRUE(condition)                                          \
    do {                                                               \
        if (!(condition)) {                                            \
            fprintf(stderr, "CHECK failed at line %d: %s\n",       \
                    __LINE__, #condition);                             \
            return 1;                                                  \
        }                                                              \
    } while (0)

static int test_qconv_plan(void)
{
    CamppRuntimeModel model;
    CamppOperatorDescriptor op;

    memset(&model, 0, sizeof(model));
    memset(&op, 0, sizeof(op));
    model.bucket_frames = 98u;

    op.operator_id = 3u;
    CHECK_TRUE(campp_conv_layer_hybrid_select_qconv(&model, &op) ==
               CAMPP_QCONV_CANDIDATE_MAC_FIXED);
    op.operator_id = 216u;
    CHECK_TRUE(campp_conv_layer_hybrid_select_qconv(&model, &op) ==
               CAMPP_QCONV_CANDIDATE_V4);
    op.operator_id = 825u;
    CHECK_TRUE(campp_conv_layer_hybrid_select_qconv(&model, &op) ==
               CAMPP_QCONV_CANDIDATE_V5);
    op.operator_id = 999u;
    CHECK_TRUE(campp_conv_layer_hybrid_select_qconv(&model, &op) ==
               CAMPP_QCONV_CANDIDATE_V4);
    model.bucket_frames = 298u;
    op.operator_id = 825u;
    CHECK_TRUE(campp_conv_layer_hybrid_select_qconv(&model, &op) ==
               CAMPP_QCONV_CANDIDATE_V4);
    return 0;
}

static int test_fused_plan(void)
{
    CamppRuntimeModel model;
    CamppOperatorDescriptor op;

    memset(&model, 0, sizeof(model));
    memset(&op, 0, sizeof(op));
    model.bucket_frames = 98u;

    op.operator_id = 42u;
    CHECK_TRUE(campp_conv_layer_hybrid_select_fused_qconv(&model, &op) ==
               CAMPP_FUSED_QCONV_CANDIDATE_COMBINED_FIXED);
    op.operator_id = 10u;
    CHECK_TRUE(campp_conv_layer_hybrid_select_fused_qconv(&model, &op) ==
               CAMPP_FUSED_QCONV_CANDIDATE_COMBINED_V5);
    op.operator_id = 829u;
    CHECK_TRUE(campp_conv_layer_hybrid_select_fused_qconv(&model, &op) ==
               CAMPP_FUSED_QCONV_CANDIDATE_COMBINED_HYBRID);
    op.operator_id = 999u;
    CHECK_TRUE(campp_conv_layer_hybrid_select_fused_qconv(&model, &op) ==
               CAMPP_FUSED_QCONV_CANDIDATE_COMBINED_HYBRID);
    return 0;
}

int main(void)
{
    CHECK_TRUE(test_qconv_plan() == 0);
    CHECK_TRUE(test_fused_plan() == 0);
    puts("conv layer-hybrid plan: PASS");
    return 0;
}
