#ifndef __EXTERN_CUSTOM_OPSETS_EXTERN_SCHEMA_SETS_H
#define __EXTERN_CUSTOM_OPSETS_EXTERN_SCHEMA_SETS_H

#include <functional>
#include "op_schema.h"

namespace oritekgraph {

class ORITEK_OPERATOR_SET_SCHEMA_CLASS_NAME(extern_cpu, 1, Abs);

class OpSet_Extern_ver1 {
public:
    static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
        fn(GetOpSchema<ORITEK_OPERATOR_SET_SCHEMA_CLASS_NAME(extern_cpu, 1, Abs)>());
    }
};


void RegisterExternCustomOpsetSchema() {
    RegisterOpSetSchema<OpSet_Extern_ver1>();
}

}
#endif
