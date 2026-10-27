#include "op_schema.h"

namespace oritekgraph {
EXTERN_CUSTOM_OPERATOR_SET_SCHEMA(
    LRN, 1,
    OpSchema()
    .Input(0, "X", "Input 4-dim or 1-dim(C=1) tensor of shape NCHW", "T")
    .Output(0, "Y", "Output", "T")
    .TypeConstraint("T", {"float"}, "Constraint input and output types")
    .Attr("alpha", "Scaling parameter", OpSchema::AttributeType::FLOAT, false)
    .Attr("beta", "The exponent", OpSchema::AttributeType::FLOAT, false)
    .Attr("bias", "Bias", OpSchema::AttributeType::FLOAT, false)
    .Attr("size", "The number of channels to sum over", OpSchema::AttributeType::INT, true)
    .SetDoc("LRN normalizes over local input regions. Y = X / ((bias + (alpha / nsize) * square_sum) ** beta), square_sum = local_sum(x ** 2)")
    );
}