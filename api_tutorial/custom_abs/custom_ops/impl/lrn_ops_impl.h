#ifndef __EXTERN_CUSTOM_OPSETS_IMPL_LRN_OPS_IMPL_H
#define __EXTERN_CUSTOM_OPSETS_IMPL_LRN_OPS_IMPL_H

#include "ot_avp_custom_utils.h"
#include "ot_type.h"

#ifdef __cplusplus
extern "C" {
#endif

ot_s32 lrn_infershape(ot_avp_custom_handle custom_op);
ot_bool lrn_check_supported(ot_avp_custom_handle custom_op);
ot_s32 lrn_compute(ot_avp_custom_handle custom_op);

#ifdef __cplusplus
};
#endif

#endif
