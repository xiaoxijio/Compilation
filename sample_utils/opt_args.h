#ifndef __SAMPLE_UTILS_OPT_ARGS_H
#define __SAMPLE_UTILS_OPT_ARGS_H
#include "ot_avp_npu_common.h"

typedef struct {
    char* model_path;
    char* input_path;
    char* output_path;
    bool perf_mode;
    bool dump_layer;
    bool is_folder;
    uint32_t repeat;
    uint32_t core_ids;
    uint32_t buffer_type;
    uint32_t output_type;
    bool multi_core;
} arg_config;

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

int parse_optarg(int argc, char** argv, arg_config* opt_arg);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
