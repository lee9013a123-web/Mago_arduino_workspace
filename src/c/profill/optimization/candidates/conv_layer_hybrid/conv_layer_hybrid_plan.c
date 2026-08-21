#include "conv_layer_hybrid_plan.h"

#include <stddef.h>
#include <stdint.h>

/* Operator 단위 latency 실측으로 고른 layer-hybrid plan.
 * 각 bucket에서 mac_fixed/v4/v5(그리고 fused 3종)를 직접 재고 incumbent보다
 * 1%% 이상 빠른 layer만 교체한다.  측정하지 않은 bucket은 테이블이 없어
 * dispatch가 v4 / combined_hybrid로 fallback한다.
 * 측정 완료 bucket: 98, 298, 498, 998 */

static const uint32_t CAMPP_QCONV_MAC_FIXED_IDS_98[] = {
    3u, 18u,
};

static const uint32_t CAMPP_QCONV_V5_IDS_98[] = {
    2u, 6u, 12u, 17u, 21u, 27u, 39u, 51u, 54u, 66u,
    69u, 81u, 84u, 96u, 99u, 111u, 114u, 126u, 129u, 141u,
    144u, 156u, 159u, 171u, 174u, 186u, 189u, 201u, 204u, 219u,
    222u, 234u, 237u, 249u, 252u, 264u, 267u, 279u, 282u, 294u,
    297u, 309u, 312u, 324u, 327u, 339u, 342u, 354u, 357u, 369u,
    372u, 384u, 387u, 399u, 402u, 414u, 417u, 429u, 432u, 444u,
    447u, 459u, 462u, 474u, 477u, 489u, 492u, 504u, 507u, 519u,
    522u, 534u, 537u, 549u, 552u, 564u, 567u, 579u, 582u, 585u,
    597u, 600u, 612u, 615u, 627u, 630u, 642u, 645u, 657u, 660u,
    672u, 675u, 687u, 690u, 702u, 705u, 717u, 720u, 732u, 735u,
    747u, 750u, 762u, 765u, 777u, 780u, 792u, 795u, 807u, 810u,
    822u, 825u,
};

static const uint32_t CAMPP_FUSED_QCONV_FIXED_IDS_98[] = {
    42u, 192u, 300u, 633u, 648u,
};

static const uint32_t CAMPP_FUSED_QCONV_V5_IDS_98[] = {
    10u, 25u, 31u, 49u, 64u, 79u, 94u, 109u, 124u, 139u,
    154u, 169u, 184u, 199u, 214u, 232u, 247u, 262u, 277u, 292u,
    307u, 322u, 337u, 352u, 367u, 382u, 397u, 412u, 427u, 442u,
    457u, 472u, 487u, 502u, 517u, 532u, 547u, 562u, 577u, 595u,
    610u, 625u, 640u, 655u, 670u, 685u, 700u, 715u, 730u, 745u,
    760u, 775u, 790u, 805u, 820u,
};

static const uint32_t CAMPP_QCONV_MAC_FIXED_IDS_298[] = {
    3u, 18u,
};

static const uint32_t CAMPP_QCONV_V5_IDS_298[] = {
    2u, 6u, 12u, 17u, 21u, 27u, 39u, 52u, 55u, 68u,
    71u, 84u, 87u, 100u, 103u, 116u, 119u, 132u, 135u, 148u,
    151u, 164u, 167u, 180u, 183u, 196u, 199u, 212u, 215u, 228u,
    231u, 234u, 247u, 250u, 263u, 266u, 279u, 282u, 295u, 298u,
    311u, 314u, 327u, 330u, 343u, 346u, 359u, 362u, 375u, 378u,
    391u, 394u, 407u, 410u, 423u, 426u, 439u, 442u, 455u, 458u,
    471u, 474u, 487u, 490u, 503u, 506u, 519u, 522u, 535u, 538u,
    551u, 554u, 567u, 570u, 583u, 586u, 599u, 602u, 615u, 618u,
    621u, 634u, 637u, 650u, 653u, 666u, 669u, 682u, 685u, 698u,
    701u, 714u, 717u, 730u, 733u, 746u, 749u, 762u, 765u, 778u,
    781u, 794u, 797u, 810u, 813u, 826u, 829u, 842u, 845u, 858u,
    861u, 874u, 877u,
};

static const uint32_t CAMPP_FUSED_QCONV_FIXED_IDS_298[] = {
    58u, 445u, 573u, 800u,
};

static const uint32_t CAMPP_FUSED_QCONV_V5_IDS_298[] = {
    10u, 25u, 31u, 50u, 66u, 82u, 98u, 114u, 130u, 146u,
    162u, 178u, 194u, 210u, 226u, 245u, 261u, 277u, 293u, 309u,
    325u, 341u, 357u, 373u, 389u, 405u, 421u, 437u, 453u, 469u,
    485u, 501u, 517u, 533u, 549u, 565u, 581u, 597u, 613u, 632u,
    648u, 664u, 680u, 696u, 712u, 728u, 744u, 760u, 776u, 792u,
    808u, 824u, 840u, 856u, 872u,
};

static const uint32_t CAMPP_QCONV_MAC_FIXED_IDS_498[] = {
    3u, 18u,
};

static const uint32_t CAMPP_QCONV_V5_IDS_498[] = {
    2u, 6u, 12u, 17u, 21u, 27u, 39u, 52u, 55u, 68u,
    71u, 84u, 87u, 100u, 103u, 116u, 119u, 132u, 135u, 148u,
    151u, 164u, 167u, 180u, 183u, 196u, 199u, 212u, 215u, 228u,
    231u, 234u, 247u, 250u, 263u, 266u, 279u, 282u, 295u, 298u,
    311u, 314u, 327u, 330u, 343u, 346u, 359u, 362u, 375u, 378u,
    391u, 394u, 407u, 410u, 423u, 426u, 439u, 442u, 455u, 458u,
    471u, 474u, 487u, 490u, 503u, 506u, 519u, 522u, 535u, 538u,
    551u, 554u, 567u, 570u, 583u, 586u, 599u, 602u, 615u, 618u,
    621u, 634u, 637u, 650u, 653u, 666u, 669u, 682u, 685u, 698u,
    701u, 714u, 717u, 730u, 733u, 746u, 749u, 762u, 765u, 778u,
    781u, 794u, 797u, 810u, 813u, 826u, 829u, 842u, 845u, 858u,
    861u, 874u, 877u,
};

static const uint32_t CAMPP_FUSED_QCONV_FIXED_IDS_498[] = {
    0u,
};

static const uint32_t CAMPP_FUSED_QCONV_V5_IDS_498[] = {
    10u, 25u, 31u, 50u, 66u, 82u, 98u, 114u, 130u, 146u,
    162u, 178u, 194u, 210u, 226u, 245u, 261u, 277u, 293u, 309u,
    325u, 341u, 357u, 373u, 389u, 405u, 421u, 437u, 453u, 469u,
    485u, 501u, 517u, 533u, 549u, 565u, 581u, 597u, 613u, 632u,
    648u, 664u, 680u, 696u, 712u, 728u, 744u, 760u, 776u, 792u,
    808u, 824u, 840u, 856u, 872u,
};

static const uint32_t CAMPP_QCONV_MAC_FIXED_IDS_998[] = {
    3u, 18u,
};

static const uint32_t CAMPP_QCONV_V5_IDS_998[] = {
    2u, 6u, 12u, 17u, 21u, 27u, 39u, 52u, 55u, 68u,
    71u, 84u, 87u, 100u, 103u, 116u, 119u, 132u, 135u, 148u,
    151u, 164u, 167u, 180u, 183u, 196u, 199u, 212u, 215u, 228u,
    231u, 234u, 247u, 250u, 263u, 266u, 279u, 282u, 295u, 298u,
    311u, 314u, 327u, 330u, 343u, 346u, 359u, 362u, 375u, 378u,
    391u, 394u, 407u, 410u, 423u, 426u, 439u, 442u, 455u, 458u,
    471u, 474u, 487u, 490u, 503u, 506u, 519u, 522u, 535u, 538u,
    551u, 554u, 567u, 570u, 583u, 586u, 599u, 615u, 618u, 621u,
    634u, 637u, 650u, 653u, 666u, 669u, 682u, 685u, 698u, 701u,
    714u, 717u, 730u, 733u, 746u, 749u, 762u, 765u, 778u, 781u,
    794u, 797u, 810u, 813u, 826u, 829u, 842u, 845u, 858u, 874u,
    877u,
};

static const uint32_t CAMPP_FUSED_QCONV_FIXED_IDS_998[] = {
    154u, 445u, 477u, 509u, 704u, 768u,
};

static const uint32_t CAMPP_FUSED_QCONV_V5_IDS_998[] = {
    10u, 25u, 31u, 50u, 66u, 82u, 98u, 114u, 130u, 146u,
    162u, 178u, 194u, 210u, 226u, 245u, 261u, 277u, 293u, 309u,
    325u, 341u, 357u, 373u, 389u, 405u, 421u, 437u, 453u, 469u,
    485u, 501u, 517u, 533u, 549u, 565u, 581u, 597u, 613u, 632u,
    648u, 664u, 680u, 696u, 712u, 728u, 744u, 760u, 776u, 792u,
    808u, 824u, 840u, 856u, 872u,
};

typedef struct {
    uint32_t bucket_frames;
    const uint32_t *qconv_mac_fixed_ids;
    size_t qconv_mac_fixed_count;
    const uint32_t *qconv_v5_ids;
    size_t qconv_v5_count;
    const uint32_t *fused_fixed_ids;
    size_t fused_fixed_count;
    const uint32_t *fused_v5_ids;
    size_t fused_v5_count;
} CamppConvLayerHybridBucketPlan;

static const CamppConvLayerHybridBucketPlan CAMPP_BUCKET_PLANS[] = {
    {
        98u,
        CAMPP_QCONV_MAC_FIXED_IDS_98, sizeof(CAMPP_QCONV_MAC_FIXED_IDS_98) / sizeof(CAMPP_QCONV_MAC_FIXED_IDS_98[0]),
        CAMPP_QCONV_V5_IDS_98, sizeof(CAMPP_QCONV_V5_IDS_98) / sizeof(CAMPP_QCONV_V5_IDS_98[0]),
        CAMPP_FUSED_QCONV_FIXED_IDS_98, sizeof(CAMPP_FUSED_QCONV_FIXED_IDS_98) / sizeof(CAMPP_FUSED_QCONV_FIXED_IDS_98[0]),
        CAMPP_FUSED_QCONV_V5_IDS_98, sizeof(CAMPP_FUSED_QCONV_V5_IDS_98) / sizeof(CAMPP_FUSED_QCONV_V5_IDS_98[0]),
    },
    {
        298u,
        CAMPP_QCONV_MAC_FIXED_IDS_298, sizeof(CAMPP_QCONV_MAC_FIXED_IDS_298) / sizeof(CAMPP_QCONV_MAC_FIXED_IDS_298[0]),
        CAMPP_QCONV_V5_IDS_298, sizeof(CAMPP_QCONV_V5_IDS_298) / sizeof(CAMPP_QCONV_V5_IDS_298[0]),
        CAMPP_FUSED_QCONV_FIXED_IDS_298, sizeof(CAMPP_FUSED_QCONV_FIXED_IDS_298) / sizeof(CAMPP_FUSED_QCONV_FIXED_IDS_298[0]),
        CAMPP_FUSED_QCONV_V5_IDS_298, sizeof(CAMPP_FUSED_QCONV_V5_IDS_298) / sizeof(CAMPP_FUSED_QCONV_V5_IDS_298[0]),
    },
    {
        498u,
        CAMPP_QCONV_MAC_FIXED_IDS_498, sizeof(CAMPP_QCONV_MAC_FIXED_IDS_498) / sizeof(CAMPP_QCONV_MAC_FIXED_IDS_498[0]),
        CAMPP_QCONV_V5_IDS_498, sizeof(CAMPP_QCONV_V5_IDS_498) / sizeof(CAMPP_QCONV_V5_IDS_498[0]),
        CAMPP_FUSED_QCONV_FIXED_IDS_498, sizeof(CAMPP_FUSED_QCONV_FIXED_IDS_498) / sizeof(CAMPP_FUSED_QCONV_FIXED_IDS_498[0]),
        CAMPP_FUSED_QCONV_V5_IDS_498, sizeof(CAMPP_FUSED_QCONV_V5_IDS_498) / sizeof(CAMPP_FUSED_QCONV_V5_IDS_498[0]),
    },
    {
        998u,
        CAMPP_QCONV_MAC_FIXED_IDS_998, sizeof(CAMPP_QCONV_MAC_FIXED_IDS_998) / sizeof(CAMPP_QCONV_MAC_FIXED_IDS_998[0]),
        CAMPP_QCONV_V5_IDS_998, sizeof(CAMPP_QCONV_V5_IDS_998) / sizeof(CAMPP_QCONV_V5_IDS_998[0]),
        CAMPP_FUSED_QCONV_FIXED_IDS_998, sizeof(CAMPP_FUSED_QCONV_FIXED_IDS_998) / sizeof(CAMPP_FUSED_QCONV_FIXED_IDS_998[0]),
        CAMPP_FUSED_QCONV_V5_IDS_998, sizeof(CAMPP_FUSED_QCONV_V5_IDS_998) / sizeof(CAMPP_FUSED_QCONV_V5_IDS_998[0]),
    },
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

static const CamppConvLayerHybridBucketPlan *campp_find_bucket_plan(
    uint32_t bucket_frames)
{
    size_t index;
    for (index = 0u; index < sizeof(CAMPP_BUCKET_PLANS) /
             sizeof(CAMPP_BUCKET_PLANS[0]); ++index) {
        if (CAMPP_BUCKET_PLANS[index].bucket_frames == bucket_frames) {
            return &CAMPP_BUCKET_PLANS[index];
        }
    }
    return NULL;
}

int campp_conv_layer_hybrid_has_bucket_plan(uint32_t bucket_frames)
{
    return campp_find_bucket_plan(bucket_frames) != NULL;
}

size_t campp_conv_layer_hybrid_bucket_plan_count(void)
{
    return sizeof(CAMPP_BUCKET_PLANS) / sizeof(CAMPP_BUCKET_PLANS[0]);
}

uint32_t campp_conv_layer_hybrid_bucket_plan_at(size_t index)
{
    return index < campp_conv_layer_hybrid_bucket_plan_count()
        ? CAMPP_BUCKET_PLANS[index].bucket_frames
        : 0u;
}

CamppQconvCandidateMode campp_conv_layer_hybrid_select_qconv(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op)
{
    const CamppConvLayerHybridBucketPlan *plan;
    if (model == NULL || op == NULL) {
        return CAMPP_QCONV_CANDIDATE_V4;
    }
    plan = campp_find_bucket_plan(model->bucket_frames);
    if (plan == NULL) {
        return CAMPP_QCONV_CANDIDATE_V4;
    }
    if (campp_id_in_sorted_table(op->operator_id,
            plan->qconv_mac_fixed_ids, plan->qconv_mac_fixed_count)) {
        return CAMPP_QCONV_CANDIDATE_MAC_FIXED;
    }
    if (campp_id_in_sorted_table(op->operator_id,
            plan->qconv_v5_ids, plan->qconv_v5_count)) {
        return CAMPP_QCONV_CANDIDATE_V5;
    }
    return CAMPP_QCONV_CANDIDATE_V4;
}

CamppFusedQconvCandidateMode campp_conv_layer_hybrid_select_fused_qconv(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op)
{
    const CamppConvLayerHybridBucketPlan *plan;
    if (model == NULL || op == NULL) {
        return CAMPP_FUSED_QCONV_CANDIDATE_COMBINED_HYBRID;
    }
    plan = campp_find_bucket_plan(model->bucket_frames);
    if (plan == NULL) {
        return CAMPP_FUSED_QCONV_CANDIDATE_COMBINED_HYBRID;
    }
    if (campp_id_in_sorted_table(op->operator_id,
            plan->fused_fixed_ids, plan->fused_fixed_count)) {
        return CAMPP_FUSED_QCONV_CANDIDATE_COMBINED_FIXED;
    }
    if (campp_id_in_sorted_table(op->operator_id,
            plan->fused_v5_ids, plan->fused_v5_count)) {
        return CAMPP_FUSED_QCONV_CANDIDATE_COMBINED_V5;
    }
    return CAMPP_FUSED_QCONV_CANDIDATE_COMBINED_HYBRID;
}
