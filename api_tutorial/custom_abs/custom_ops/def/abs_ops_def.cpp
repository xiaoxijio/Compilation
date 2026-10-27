#include "op_schema.h"

namespace oritekgraph {
EXTERN_CUSTOM_OPERATOR_SET_SCHEMA(
    Abs,
    1,
    OpSchema()
    .Input(0, "X", "N-D tensor", "T")
    .Output(0, "Y", "N-D tensor", "T")
    .TypeConstraint("T", {"float", "int32"}, "only fp32 int32 supported")
    .SetDoc("Y=abs(X)")
    );
}
