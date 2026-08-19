#ifndef CAMPP_PROFILL_QCONV_V4_ADDRESS_V2_ADAPTER_H
#define CAMPP_PROFILL_QCONV_V4_ADDRESS_V2_ADAPTER_H

#include "../../api/qconv_address_provider.h"
#include "planning/qconv_v4_execution_plan.h"

CamppQconvAddressStatus campp_qconv_v4_address_v2_geometry(
    const CamppQconvV4ExecutionPlan *v4_plan,
    CamppQconvAddressGeometry *out_geometry);

#endif /* CAMPP_PROFILL_QCONV_V4_ADDRESS_V2_ADAPTER_H */
