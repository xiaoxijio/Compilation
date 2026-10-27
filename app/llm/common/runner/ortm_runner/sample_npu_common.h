#ifndef SAMPLE_NPU_COMMON_H
#define SAMPLE_NPU_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "ot_avp_npu_rts.h"
/* LDRA_EXCLUDE_START 78 S */
#define SAMPLE_NPU_CHECK_RETURN(val, ret)       \
    do {                                        \
        if (val) {                              \
            return ret;                         \
        }                                       \
    } while(false);

#define SAMPLE_NPU_CHECK_RETURN_WITH_LOG(val, ret, ...)    \
    do {                                        \
        if (val) {                              \
            printf(__VA_ARGS__);                \
            return ret;                         \
        }                                       \
    } while(false);

#define SAMPLE_NPU_CHECK_GOTO(val, label)       \
    do {                                        \
        if (val) {                              \
            goto label;                         \
        }                                       \
    } while(false);

#define SAMPLE_NPU_CHECK_GOTO_WITH_LOG(val, label, ...)    \
    do {                                        \
        if (val) {                              \
            printf(__VA_ARGS__);                \
            goto label;                         \
        }                                       \
    } while(false);
/* LDRA_EXCLUDE_END  78 S */
#define OT_TEST_MAX_INPUT_OUTPUT_NUM 64

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

typedef struct {
    void *handle;
    ot_u32 input_num;
    ot_u32 output_num;
    ot_avp_tensor input_tensor[OT_TEST_MAX_INPUT_OUTPUT_NUM];
    ot_avp_tensor output_tensor[OT_TEST_MAX_INPUT_OUTPUT_NUM];
    ot_avp_npu_dataset* input_dataset;
    ot_avp_npu_dataset* output_dataset;
} model_task_info;

int ot_avp_create_tensor(ot_avp_tensor *tensor, ot_u32 size, ot_avp_npu_shape shape, ot_avp_data_type dtype, ot_u32 stride);
int ot_avp_destroy_tensor(ot_avp_tensor *tensor);

void prepare_input_output_tensor(void *handle, model_task_info* info);
void destory_input_output_tensor(void *handle, model_task_info* info);

ot_float data_convert_fp16_to_fp32(const ot_u16 in);
ot_u16 data_convert_fp32_to_fp16(const ot_float in);

int32_t sample_avp_npu_malloc(void **ptr, ot_phys_addr *phys_addr, uint64_t size);
int32_t sample_avp_npu_free(void *ptr);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
