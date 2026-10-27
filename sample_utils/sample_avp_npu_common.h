#ifndef SAMPLE_AVP_NPU_COMMON_H
#define SAMPLE_AVP_NPU_COMMON_H

#include "ot_avp_npu_rts.h"
#ifdef SIMULATOR
#include "simulator_utils.h"
#else
#include "platform_utils.h"
#endif

#include "file_utils.h"
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <complex.h>
#include <float.h>
#include <time.h>
#include <signal.h>     /* for signal */

#define RETURN_NONE
#define SAMPLE_AVP_NPU_MAX_BUFF_NUM 256
#define SAMPLE_AVP_NPU_MAX_BBOX_NUM 300
#define CURRENT_DIM         640

#undef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#undef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))

typedef struct {
    uint64_t virt_addr;
    uint64_t phys_addr;
    uint32_t size;
} sample_avp_mem_info_t;

typedef struct {
    uint64_t virt_addr;
    uint64_t phys_addr;
    uint32_t size;

    ot_avp_npu_shape shape;
    ot_avp_data_type dtype;
    uint32_t byte_width;

    int32_t stride; // 只记录最后一维的跨度，其他维跨度，可以通过其他参数计算。
} sample_avp_tensor_t;

typedef struct {
    void *handle;
    uint32_t input_num;
    uint32_t output_num;
    sample_avp_tensor_t input_tensor[SAMPLE_AVP_NPU_MAX_BUFF_NUM];
    sample_avp_tensor_t output_tensor[SAMPLE_AVP_NPU_MAX_BUFF_NUM];
    ot_avp_npu_dataset* input_dataset;
    ot_avp_npu_dataset* output_dataset;
    void* runtime_buf;
    bool  with_sram;
    sample_avp_mem_info_t input_batch_buff;
} sample_avp_model_task_info_t;

typedef struct sample_rect {
    float x1;
    float y1;
    float x2;
    float y2;
    float conf;
    int class_id;
} sample_avp_npu_rect;

typedef struct sample_npu_context_t {
    sample_avp_model_task_info_t model_info;

    sample_avp_npu_rect out_bbox[SAMPLE_AVP_NPU_MAX_BBOX_NUM];
    ot_u32 out_bbox_num;
    ot_u32 out_valid_bbox_num;
} sample_npu_context_t;

typedef struct _sample_npu_rect_t {
    sample_avp_npu_rect out_rect[SAMPLE_AVP_NPU_MAX_BBOX_NUM];
    int num;
} sample_avp_npu_rect_t;

#define SAMPLE_AVP_CHECK_RETURN(val, ret)                  \
    do {                                        \
        if (val) {                              \
            return ret;                         \
        }                                       \
    } while(0);

#define SAMPLE_AVP_CHECK_RETURN_WITH_LOG(val, ret, ...)    \
    do {                                        \
        if (val) {                              \
            printf(__VA_ARGS__);                \
            return ret;                         \
        }                                       \
    } while(0);

#define SAMPLE_AVP_CHECK_GOTO(val, label)       \
    do {                                        \
        if (val) {                              \
            goto label;                         \
        }                                       \
    } while(0);

#define SAMPLE_AVP_CHECK_GOTO_WITH_LOG(val, label, ...)    \
    do {                                        \
        if (val) {                              \
            printf(__VA_ARGS__);                \
            goto label;                         \
        }                                       \
    } while(0);

#define SAMPLE_AVP_CHECK_CONTINUE(val)          \
    if (val) {                                  \
        continue;                               \
    }

#define SAMPLE_AVP_CHECK_BREAK(val)             \
    if (val) {                                  \
        break;                                  \
    }

typedef enum sample_avp_npu_log_level {
    SAMPLE_AVP_NPU_LOG_LEVEL_TRACE = 0,
    SAMPLE_AVP_NPU_LOG_LEVEL_DEBUG = 1,
    SAMPLE_AVP_NPU_LOG_LEVEL_INFO  = 2,
    SAMPLE_AVP_NPU_LOG_LEVEL_WARN  = 3,
    SAMPLE_AVP_NPU_LOG_LEVEL_ERROR = 4,
    SAMPLE_AVP_NPU_LOG_LEVEL_FATAL = 5,
} sample_avp_npu_log_level_e;


#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

#define SAMPLE_AVP_NPU_LOG_DEBUG(...) sample_avp_npu_print_log(SAMPLE_AVP_NPU_LOG_LEVEL_DEBUG, __FUNCTION__, __LINE__, __VA_ARGS__);
#define SAMPLE_AVP_NPU_LOG_INFO(...)  sample_avp_npu_print_log(SAMPLE_AVP_NPU_LOG_LEVEL_INFO, __FUNCTION__, __LINE__, __VA_ARGS__);
#define SAMPLE_AVP_NPU_LOG_WARN(...)  sample_avp_npu_print_log(SAMPLE_AVP_NPU_LOG_LEVEL_WARN, __FUNCTION__, __LINE__, __VA_ARGS__);
#define SAMPLE_AVP_NPU_LOG_ERROR(...) sample_avp_npu_print_log(SAMPLE_AVP_NPU_LOG_LEVEL_ERROR, __FUNCTION__, __LINE__, __VA_ARGS__);
#define SAMPLE_AVP_NPU_LOG_FATAL(...) sample_avp_npu_print_log(SAMPLE_AVP_NPU_LOG_LEVEL_FATAL, __FUNCTION__, __LINE__, __VA_ARGS__);

struct box{
    float confidence;
    int index;
};
void sample_avp_npu_print_log(sample_avp_npu_log_level_e level, const char* function, int line, const char* fmt, ...);

int32_t sample_avp_create_tensor(sample_avp_tensor_t *tensor, uint32_t size, ot_avp_npu_shape shape, ot_avp_data_type dtype, uint32_t stride);
int32_t sample_avp_destroy_tensor(sample_avp_tensor_t *tensor);

void sample_avp_create_datasets(void *handle, sample_avp_model_task_info_t* info);
void sample_avp_destroy_datasets(void *handle, sample_avp_model_task_info_t* info);

float data_convert_fp16_to_fp32(const uint16_t in);
int32_t sample_avp_execute_model(sample_avp_model_task_info_t* model_param, uint32_t repeat_times);
void transpose_nobubble_float32(float* matrix, int rows, int cols);
void transpose_nobubble_float16(uint16_t* matrix, int rows, int cols);

uint32_t sample_npu_yolo_nms(uint32_t out_bbox_num, float* inferData, float iou_thresh, float conf);
ot_void sample_npu_yolov7_nms(sample_npu_context_t *context);
float iou_compute(sample_avp_npu_rect* lbox, sample_avp_npu_rect* rbox);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif