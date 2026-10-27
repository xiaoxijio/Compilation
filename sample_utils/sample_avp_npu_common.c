#include "sample_avp_npu_common.h"
#if defined(__linux__)
#include <unistd.h>
#include <dirent.h>
#elif defined(_WIN32) || defined(_WIN64)
#include <Windows.h>
#include <direct.h>
#endif

#include <math.h>
#ifndef SIMULATOR
#include <arm_neon.h>
#endif

#include <stdarg.h>
#include <stdio.h>
#include <time.h>

static const char* DTYPE_ARRAY[] = {
    "F32",
    "F64",
    "INT8",
    "INT16",
    "INT32",
    "INT64",
    "UINT8",
    "UINT16",
    "UINT32",
    "UINT64",
    "F16",
    "INT4",
    "UINT4",
    "INT5",
    "UINT5",
    "INT6",
    "UINT6",
    "INT7",
    "UINT7",
    "BOOL",
    "INT10",
    "UINT10",
    "INT12",
    "UINT12",
    "BUTT"
};

bool sample_avp_to_shape_str(char str[], int str_size, const ot_avp_npu_shape* shape)
{
    uint32_t dim_size = shape->dim_size;
    if (dim_size < 1U) {
        fprintf(stderr, "shape size is 0\n");
        return false;
    }
    for (uint32_t i = 0; i < dim_size; i++) {
        int len = snprintf(str, str_size, "%d", shape->dims[i]);
        str = str+len;
        str_size = str_size - len;
        if ((i != (dim_size-1U)) && (str_size > 0)) {
            str[0] = 'x';
            str = str + 1;
            str_size--;
        }
    }
    return true;
}

int32_t sample_avp_create_tensor(sample_avp_tensor_t *tensor, uint32_t size,
    ot_avp_npu_shape shape, ot_avp_data_type dtype, uint32_t stride)
{
    tensor->size = size;
    tensor->shape = shape;
    tensor->dtype = dtype;
    tensor->stride = stride;

    int32_t ret = sample_avp_npu_malloc((ot_void **)&tensor->virt_addr, size);
    if (ret != 0) {
        return -1;
    }
    tensor->byte_width = ot_avp_npu_data_type_size(dtype) / 8;

    return 0;
}

// 计算后继索引
int getNext(int i, int rows, int cols) {
    return (i % cols) * rows + i / cols;
}

// 计算前驱索引
int getPre(int i, int rows, int cols) {
    return (i % rows) * cols + i / rows;
}

// 处理环置换
void moveData_float(float* matrix, int i, int rows, int cols) {
    float temp = matrix[i];  // 暂存当前值
    int cur = i;
    int pre = getPre(cur, rows, cols);

    while (pre != i) {
        matrix[cur] = matrix[pre];
        cur = pre;
        pre = getPre(cur, rows, cols);
    }
    matrix[cur] = temp;
}

void moveData_uint16(uint16_t* matrix, int i, int rows, int cols) {
    uint16_t temp = matrix[i];  // 暂存当前值
    int cur = i;
    int pre = getPre(cur, rows, cols);

    while (pre != i) {
        matrix[cur] = matrix[pre];
        cur = pre;
        pre = getPre(cur, rows, cols);
    }
    matrix[cur] = temp;
}

float data_convert_fp16_to_fp32(const uint16_t in)
{
    float out = 0.0f;

    uint32_t t1 = (uint32_t)in & 0x7fffU;                // Non-sign bits
    uint32_t t2 = (uint32_t)in & 0x8000U;                // Sign bit
    uint32_t t3 = (uint32_t)in & 0x7c00U;                // Exponent

    t1 <<= 13;                              // Align mantissa on MSB
    t2 <<= 16;                              // Shift sign bit into position

    t1 += 0x38000000U;                       // Adjust bias

    t1 = (t3 == 0U ? 0U : t1);              // Denormals-as-zero

    t1 |= t2;                               // Re-insert sign bit
    (void)memcpy(&out, &t1, sizeof(out));
    return out;
}

// 执行矩阵转置
void transpose_nobubble_float32(float* matrix, int rows, int cols)
{
    for (int i = 0; i < rows * cols; ++i) {
        int next = getNext(i, rows, cols);
        while (next > i)
            next = getNext(next, rows, cols);
        if (next == i)
            moveData_float(matrix, i, rows, cols);
    }
}

void transpose_nobubble_float16(uint16_t* matrix, int rows, int cols)
{
    for (int i = 0; i < rows * cols; ++i) {
        int next = getNext(i, rows, cols);
        while (next > i)
            next = getNext(next, rows, cols);
        if (next == i)
            moveData_uint16(matrix, i, rows, cols);
    }
}

int32_t sample_avp_destroy_tensor(sample_avp_tensor_t *tensor)
{
    sample_avp_npu_free((ot_void *)tensor->virt_addr);

    memset(tensor, 0, sizeof(sample_avp_tensor_t));

    return 0;
}

void sample_avp_create_datasets(void *handle, sample_avp_model_task_info_t* info)
{
    (void)ot_avp_npu_get_input_num(handle, &info->input_num);
    (void)ot_avp_npu_get_output_num(handle, &info->output_num);
    info->input_dataset = ot_avp_npu_create_input_dataset(handle);
    info->output_dataset = ot_avp_npu_create_output_dataset(handle);

    printf("inputs:\n");
    for (uint32_t i = 0; i < info->input_num; i++) {
        uint32_t stride = ot_avp_npu_get_input_default_stride(handle, i);
        uint32_t buf_size = ot_avp_npu_get_input_size_by_index(handle, i);
        ot_avp_npu_shape shape;
        (void)ot_avp_npu_get_input_shape_by_index(handle, i, &shape);
        ot_avp_data_type data_type;
        (void)ot_avp_npu_get_input_dtype_by_index(handle, i, &data_type);

        char shape_str[1024];
        (void)sample_avp_to_shape_str(shape_str, 1024, &shape);
        printf("\t%d: dtype: %s shape: %s stride: %d size: %d\n", i, DTYPE_ARRAY[data_type], shape_str, stride, buf_size);
        (void)sample_avp_create_tensor(&info->input_tensor[i], buf_size, shape, data_type, stride);
        (void)ot_avp_npu_set_buffer(info->input_dataset, i, (ot_u8 *)info->input_tensor[i].virt_addr,
            0, buf_size, stride);
    }
    printf("outputs:\n");
    for (uint32_t i = 0; i < info->output_num; i++) {
        uint32_t stride = ot_avp_npu_get_output_default_stride(handle, i);
        uint32_t buf_size = ot_avp_npu_get_output_size_by_index(handle, i);
        ot_avp_npu_shape shape;
        (void)ot_avp_npu_get_output_shape_by_index(handle, i, &shape);
        ot_avp_data_type data_type;
        (void)ot_avp_npu_get_output_dtype_by_index(handle, i, &data_type);

        char shape_str[1024];
        (void)sample_avp_to_shape_str(shape_str, 1024, &shape);
        printf("\t%d: dtype: %s shape: %s stride: %d size: %d\n", i, DTYPE_ARRAY[data_type], shape_str, stride, buf_size);
        (void)sample_avp_create_tensor(&info->output_tensor[i], buf_size, shape, data_type, stride);
        (void)ot_avp_npu_set_buffer(info->output_dataset, i, (ot_u8 *)info->output_tensor[i].virt_addr,
            0, buf_size, stride);
    }
}

void sample_avp_destroy_datasets(void *handle, sample_avp_model_task_info_t* info)
{
    (void)handle;
    (void)ot_avp_npu_destroy_dataset(info->input_dataset);
    (void)ot_avp_npu_destroy_dataset(info->output_dataset);
    for (uint32_t i = 0; i < info->input_num; i++) {
        (void)sample_avp_destroy_tensor(&info->input_tensor[i]);
    }
    for (uint32_t i = 0; i < info->output_num; i++) {
        (void)sample_avp_destroy_tensor(&info->output_tensor[i]);
    }
}

int32_t sample_avp_execute_model(sample_avp_model_task_info_t* model_param, uint32_t repeat_times)
{
#ifdef SIMULATOR
    (void)repeat_times;
    int32_t ret = ot_avp_npu_execute(model_param->handle, model_param->input_dataset, model_param->output_dataset);
    SAMPLE_AVP_CHECK_RETURN_WITH_LOG(ret != 0, ret, "execute fail.\n");
#else
    struct timespec start, end;
    double max_time = 0.0, min_time = DBL_MAX, sum_time = 0.0;
    uint32_t i;
    for (i = 0; i < repeat_times; i++) {
        clock_gettime(CLOCK_REALTIME, &start);
        int32_t ret = ot_avp_npu_execute(model_param->handle, model_param->input_dataset, model_param->output_dataset);
        SAMPLE_AVP_CHECK_RETURN_WITH_LOG(ret != 0, ret, "execute fail.\n");
        clock_gettime(CLOCK_REALTIME, &end);
        long long exec_time_ns = ((end.tv_sec - start.tv_sec) * 1000000000) + (end.tv_nsec - start.tv_nsec);
        double time_ms = exec_time_ns / 1000000.0;
        sum_time += time_ms;
        max_time = MAX(max_time, time_ms);
        min_time = MIN(min_time, time_ms);
        if (repeat_times == (uint32_t)-1 && ((i % 1000U) == 1U)) {
            printf("exec %d avp_time: %.3f\n", i, sum_time / i);
        }
    }
    printf("\n==== oritek execute result ====\n");
    printf("max_time: %.3lf ms, min_time: %.3lf ms, avg_time: %.3f ms, fps: %.3f, repeat: (%d/%d)\n\n",
        max_time, min_time, (repeat_times > 2) ? (sum_time - max_time - min_time) / (repeat_times - 2) : sum_time / repeat_times,
        1000.0 / sum_time * repeat_times,
        i, repeat_times);
#endif

    return 0;
}

float iou_compute(sample_avp_npu_rect* lbox, sample_avp_npu_rect* rbox)
{
    float tmp_rect[4] = {
        MAX(lbox->x1, rbox->x1),
        MAX(lbox->y1, rbox->y1),
        MIN(lbox->x2, rbox->x2),
        MIN(lbox->y2, rbox->y2)
    };

    if (tmp_rect[1] >= tmp_rect[3] || tmp_rect[0] >= tmp_rect[2]) {
        return 0.0f;
    }

    float interBoxS = (tmp_rect[2] - tmp_rect[0]) * (tmp_rect[3] - tmp_rect[1]);
    float lbox_area = (lbox->x2 - lbox->x1) * (lbox->y2 - lbox->y1);
    float rbox_area = (rbox->x2 - rbox->x1) * (rbox->y2 - rbox->y1);
    float extra = 0.0000000001;

    return interBoxS / (lbox_area + rbox_area - interBoxS + extra);
}

void sample_avp_npu_print_log(sample_avp_npu_log_level_e level, const char* function, int line, const char* fmt, ...)
{
    static sample_avp_npu_log_level_e logLevel = SAMPLE_AVP_NPU_LOG_LEVEL_INFO;
    static const char* LEVEL_STR[] = { "[SAMPLE] Trace", "[SAMPLE] Debug", "[SAMPLE] Info", "[SAMPLE] Warn", "[SAMPLE] Err", "[SAMPLE] Fatal" };
    if (level < logLevel) {
        return;
    }

    time_t t = time(NULL);
    va_list ap;
    va_start(ap, fmt);
    char buf[16];
    buf[strftime(buf, sizeof(buf), "%H:%M:%S", localtime(&t))] = '\0';
    fprintf(stdout, "%s %-5s %s:%d ", buf, LEVEL_STR[level], function, line);
    vfprintf(stdout, fmt, ap);
    fprintf(stdout, "\n");
    (void)fflush(stdout);
    va_end(ap);
}
