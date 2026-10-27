/* LDRA_EXCLUDE_START 554 S */
#include "extern_schema_sets.h"
#include "opsets.h"

namespace oritekgraph {

extern "C" void RegisterCustomOp(void)
{
    RegisterExternCustomOpsetSchema();
}
/* LDRA_EXCLUDE_START 78 S */
/* LDRA_EXCLUDE 67 S */
#define ADD_EXTERN_CUSTOM_OP_IMPL(opName, inferShape, checkSupported, compute) \
    opSetImpls->emplace_back(std::make_pair(#opName, ot_avp_custom_op_impl{#compute, OPIMPL_PRI_EXTERN, inferShape, checkSupported, compute}))
/* LDRA_EXCLUDE_END  78 S */
extern "C" void GetOpSetImpls(void* opSetImplsHandle) {
    auto opSetImpls = reinterpret_cast<std::vector<std::pair<std::string, ot_avp_custom_op_impl>>*>(opSetImplsHandle);
    ADD_EXTERN_CUSTOM_OP_IMPL(Abs, abs_infershape, abs_check_supported_float, abs_compute_float);
    ADD_EXTERN_CUSTOM_OP_IMPL(Abs, abs_infershape, abs_check_supported_int32, abs_compute_int32);
}
}
/* LDRA_EXCLUDE_END 554 S */
