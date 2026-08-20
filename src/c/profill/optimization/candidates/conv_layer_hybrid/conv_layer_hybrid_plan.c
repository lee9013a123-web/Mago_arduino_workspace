#include "conv_layer_hybrid_plan.h"

#include <stddef.h>
#include <stdint.h>

/* Generated from conv_hybrid_plan.json, 2026-08-20, min margin 1%. */
static const uint32_t CAMPP_QCONV_MAC_FIXED_IDS[] = {3u, 18u};

static const uint32_t CAMPP_QCONV_V5_IDS[] = {
    2u, 6u, 12u, 17u, 21u, 27u, 39u, 51u, 54u, 66u, 69u, 81u,
    84u, 96u, 99u, 111u, 114u, 126u, 129u, 141u, 144u, 156u,
    159u, 171u, 174u, 186u, 189u, 201u, 204u, 219u, 222u, 234u,
    237u, 249u, 252u, 264u, 267u, 279u, 282u, 294u, 297u, 309u,
    312u, 324u, 327u, 339u, 342u, 354u, 357u, 369u, 372u, 384u,
    387u, 399u, 402u, 414u, 417u, 429u, 432u, 444u, 447u, 459u,
    462u, 474u, 477u, 489u, 492u, 504u, 507u, 519u, 522u, 534u,
    537u, 549u, 552u, 564u, 567u, 579u, 582u, 585u, 597u, 600u,
    612u, 615u, 627u, 630u, 642u, 645u, 657u, 660u, 672u, 675u,
    687u, 690u, 702u, 705u, 717u, 720u, 732u, 735u, 747u, 750u,
    762u, 765u, 777u, 780u, 792u, 795u, 807u, 810u, 822u, 825u
};

static const uint32_t CAMPP_FUSED_QCONV_FIXED_IDS[] = {
    42u, 192u, 300u, 633u, 648u
};

static const uint32_t CAMPP_FUSED_QCONV_V5_IDS[] = {
    10u, 25u, 31u, 49u, 64u, 79u, 94u, 109u, 124u, 139u, 154u,
    169u, 184u, 199u, 214u, 232u, 247u, 262u, 277u, 292u, 307u,
    322u, 337u, 352u, 367u, 382u, 397u, 412u, 427u, 442u, 457u,
    472u, 487u, 502u, 517u, 532u, 547u, 562u, 577u, 595u, 610u,
    625u, 640u, 655u, 670u, 685u, 700u, 715u, 730u, 745u, 760u,
    775u, 790u, 805u, 820u
};

static int campp_id_in_sorted_table(
    uint32_t operator_id, const uint32_t *ids, size_t count)
{
    size_t lower = 0u;
    size_t upper = count;

    while (lower < upper) {
        const size_t middle = lower + (upper - lower) / 2u;
        if (ids[middle] < operator_id) {
            lower = middle + 1u;
        } else {
            upper = middle;
        }
    }
    return lower < count && ids[lower] == operator_id;
}

CamppQconvCandidateMode campp_conv_layer_hybrid_select_qconv(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op)
{
    if (model == NULL || op == NULL || model->bucket_frames != 98u) {
        return CAMPP_QCONV_CANDIDATE_V4;
    }
    if (campp_id_in_sorted_table(
            op->operator_id, CAMPP_QCONV_MAC_FIXED_IDS,
            sizeof(CAMPP_QCONV_MAC_FIXED_IDS) /
                sizeof(CAMPP_QCONV_MAC_FIXED_IDS[0]))) {
        return CAMPP_QCONV_CANDIDATE_MAC_FIXED;
    }
    if (campp_id_in_sorted_table(
            op->operator_id, CAMPP_QCONV_V5_IDS,
            sizeof(CAMPP_QCONV_V5_IDS) / sizeof(CAMPP_QCONV_V5_IDS[0]))) {
        return CAMPP_QCONV_CANDIDATE_V5;
    }
    return CAMPP_QCONV_CANDIDATE_V4;
}

CamppFusedQconvCandidateMode campp_conv_layer_hybrid_select_fused_qconv(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op)
{
    if (model == NULL || op == NULL || model->bucket_frames != 98u) {
        return CAMPP_FUSED_QCONV_CANDIDATE_COMBINED_HYBRID;
    }
    if (campp_id_in_sorted_table(
            op->operator_id, CAMPP_FUSED_QCONV_FIXED_IDS,
            sizeof(CAMPP_FUSED_QCONV_FIXED_IDS) /
                sizeof(CAMPP_FUSED_QCONV_FIXED_IDS[0]))) {
        return CAMPP_FUSED_QCONV_CANDIDATE_COMBINED_FIXED;
    }
    if (campp_id_in_sorted_table(
            op->operator_id, CAMPP_FUSED_QCONV_V5_IDS,
            sizeof(CAMPP_FUSED_QCONV_V5_IDS) /
                sizeof(CAMPP_FUSED_QCONV_V5_IDS[0]))) {
        return CAMPP_FUSED_QCONV_CANDIDATE_COMBINED_V5;
    }
    return CAMPP_FUSED_QCONV_CANDIDATE_COMBINED_HYBRID;
}
