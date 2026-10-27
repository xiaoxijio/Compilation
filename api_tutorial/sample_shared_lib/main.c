#include "ot_avp_npu_rts.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#if defined(__linux__)
#include <unistd.h>
#include <dirent.h>
#define LIB_PATH "/../lib"
#elif defined(_WIN32) || defined(_WIN64)
#include <Windows.h>
#include <direct.h>
#define LIB_PATH "\\..\\bin"
#endif
#include "file_utils.h"
#include "opt_args.h"
#include <time.h>
#include <float.h>
#include <signal.h>
#ifndef SIMULATOR
#include "platform_utils.h"
#else
#include "simulator_utils.h"
#endif
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

typedef struct {
    void* buf;
    uint32_t buf_len;
    uint32_t stride;
} data_buf;


typedef struct {
    char* input_path;
    char* output_path;
    int input_num;
    int output_num;
    ot_avp_npu_dataset* input_dataset;
    ot_avp_npu_dataset* output_dataset;
    arg_config* opt_arg;
    data_buf* input_buf;
    data_buf* output_buf;
} exec_args;

#if defined(__linux__)
volatile sig_atomic_t g_exit_flag = 0;

static ot_void sample_avp_npu_handle_sig(ot_s32 signo)
{
    if (signo == SIGINT || signo == SIGTERM) {
        g_exit_flag = 1;
    }
}

static ot_void sample_avp_npu_register_sig_handler(ot_void (*sig_handle)(ot_s32))
{
    struct sigaction sa;

    (ot_void)memset(&sa, 0, sizeof(struct sigaction));
    sa.sa_handler = sig_handle;
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, OT_NULL);
    sigaction(SIGTERM, &sa, OT_NULL);
}
#else
int g_exit_flag = 0;
#endif

static ot_void save_model_results(ot_avp_handle handle, exec_args* args, char *output_filename_prefix, data_buf* output_data)
{
    if (args->output_path == NULL) {
        return;
    }
    ot_u32 output_num = (ot_u32)args->output_num;
    arg_config* opt_arg = args->opt_arg;
    for (ot_u32 i = 0; i < output_num; i++) {
        ot_avp_data_type dtype;
        (void)ot_avp_npu_get_output_dtype_by_index(handle, i, &dtype);
        ot_avp_npu_shape shape;
        (void)ot_avp_npu_get_output_shape_by_index(handle, i, &shape);
        ot_u32 stride = ot_avp_npu_get_output_default_stride(handle, i);
        // dump to file
        /* LDRA_EXCLUDE 621 S */
        char output_file[FILE_PATH_MAX] = {0};
        ot_s32 ret = 0;
        /* LDRA_EXCLUDE 589 S */
        char *suffix = (opt_arg->output_type == 0U || dtype == OT_AVP_DTYPE_UINT10 || dtype == OT_AVP_DTYPE_INT12) ? (char *)"bin" : (char *)"txt";
        ret = snprintf(output_file, FILE_PATH_MAX, "%s%c%s_%d.%s", args->output_path, PATH_SEPARATOR, output_filename_prefix, i, suffix);
        if (ret < 0) {
            printf("dump output fail, output file is too long\n");
            return;
        }
        printf("dump output path: %s\n", output_file);
        ret = (opt_arg->output_type == 0U)
            ? dump_data_to_binary_file(output_file, (ot_u8*)output_data[i].buf, shape, stride, dtype)
            : dump_data_to_txt_file(output_file, (ot_u8*)output_data[i].buf, shape, stride, dtype);
        if (ret != 0) {
            printf("dump output:%s failed\n", output_file);
        }
    }

}

static ot_s32 execute_model(ot_avp_handle handle, exec_args* args)
{
    ot_u32 input_num = (ot_u32)args->input_num;
    ot_u32 output_num = (ot_u32)args->output_num;
    ot_avp_npu_dataset* input_dataset = args->input_dataset;
    ot_avp_npu_dataset* output_dataset = args->output_dataset;
    data_buf* input_data = args->input_buf;
    data_buf* output_data = args->output_buf;
    ot_s32 ret = -1;
    char** input_file = (char**)malloc(input_num*sizeof(char*));
    for (ot_u32 i = 0; i < input_num; i++) {
        input_file[i] = (char*)malloc(FILE_PATH_MAX);
/* LDRA_EXCLUDE 72 D */
        (void)memset(input_file[i], 0, FILE_PATH_MAX);
    }
    if (args->input_path != NULL) {
        ret = get_input_file(args->input_path, input_file, input_num);
        if (ret != 0) {
            printf("get input file failed\n");
            goto exec_finish;
        }
    } else if (args->output_path != NULL) {
        printf("can not find input args, please reference ortm_run_model -m xxx.otrm -i xxx -o xxx\n");
        goto exec_finish;
    } else {
        /*non-execution*/
    }
    for (ot_u32 i = 0; i < (ot_u32)args->input_num; i++) {
        ot_avp_data_type dtype;
        (void)ot_avp_npu_get_input_dtype_by_index(handle, i, &dtype);
        ot_avp_npu_shape shape;
        (void)ot_avp_npu_get_input_shape_by_index(handle, i, &shape);
        ot_u32 stride = ot_avp_npu_get_input_default_stride(handle, i);
        if (stride == 0U) {
            printf("stride can't be 0\n");
            goto exec_finish;
        }
        if (args->input_path != NULL) {
            ret = load_data_from_file(input_file[i], input_data[i].buf, shape, stride, dtype);
            if (ret != 0) {
                printf("load data from file %s failed\n", input_file[i]);
                goto exec_finish;
            }
        }
    }
    for (ot_u32 i = 0; i < output_num; i++) {
        (void)memset((ot_u8*)output_data[i].buf, 0, output_data[i].buf_len);
    }
    // execute model
#ifdef SIMULATOR
    ret = ot_avp_npu_execute(handle, input_dataset, output_dataset);
    if (ret != 0) {
        printf("run model error\n");
        goto exec_finish;
    }
#else
    struct timespec start, end;
    double max_time = 0.0, min_time = DBL_MAX, sum_time = 0.0, avg_time = 0.0, fps = 0.0;
    ot_u32 exec_num = 0;
    arg_config* opt_arg = args->opt_arg;
    printf("repeat: %d\n", opt_arg->repeat);
    while (true) {
        exec_num++;
        if (opt_arg->repeat != (ot_u32)-1) {
            if (exec_num > opt_arg->repeat) {
                break;
            }
        }
        if (g_exit_flag == 1) {
            break;
        }
        clock_gettime(CLOCK_REALTIME, &start);
        // ot_avp_npu_set_buffer(input_dataset, 0, input_data[0].buf, 0, input_data[0].buf_len, input_data[0].stride);
        ret = ot_avp_npu_execute(handle, input_dataset, output_dataset);
        if (ret != 0) {
            printf("exec %d-th, run model fail.\n", exec_num);
            goto exec_finish;
        }
        clock_gettime(CLOCK_REALTIME, &end);
        long long exec_time_ns = ((end.tv_sec - start.tv_sec) * 1000000000) + (end.tv_nsec - start.tv_nsec);
        double time_ms = (double)exec_time_ns / 1000000.0;
        sum_time += time_ms;
        max_time = MAX(max_time, time_ms);
        min_time = MIN(min_time, time_ms);
        if (opt_arg->repeat == (ot_u32)-1) {
            if ((exec_num % 1000U) == 0U) {
                avg_time = sum_time / 1000.0;
                fps = 1000.0 / avg_time;
                printf("total times: %8d, (per 1k times) avg_time: %.3f ms, fps: %.3f", exec_num, avg_time, fps);
                sum_time = 0.0;
#ifndef SIMULATOR
                ot_avp_npu_current_state core0_state = {0};
                ot_avp_npu_current_state core1_state = {0};
                if (((opt_arg->core_ids & OT_AVP_NPU_CORE_0) != 0) || opt_arg->core_ids == 0) {
                    ret = ot_avp_npu_query_current_state_by_coreid(OT_AVP_NPU_CORE_0, &core0_state);
                    printf(", (last 1s) hw utilization: [core 0]: %3d", core0_state.hw_utilization);
                }
                if (((opt_arg->core_ids & OT_AVP_NPU_CORE_1) != 0) || opt_arg->core_ids == 0) {
                    ret = ot_avp_npu_query_current_state_by_coreid(OT_AVP_NPU_CORE_1, &core1_state);
                    printf(", (last 1s) hw utilization: [core 1]: %3d", core1_state.hw_utilization);
                }
#endif
                printf("\n");
            }
        }
    }
    if (opt_arg->repeat != (ot_u32)-1) {
        sum_time = (opt_arg->repeat > 2) ? (sum_time - max_time - min_time) : sum_time;
        avg_time = sum_time / ((opt_arg->repeat > 2) ? opt_arg->repeat - 2 : opt_arg->repeat);
        fps      = 1000.0 / avg_time;
    }
    printf("\n==== oritek execute result ====\n");
    if (opt_arg->repeat != 0U) {
        printf("max_time: %.3lf ms, min_time: %.3lf ms, avg_time: %.3f ms, fps: %.3f, repeat: %d\n\n",
            max_time, min_time, avg_time, fps, opt_arg->repeat);
    }
#endif

    // save results
    char input_file_name[FILE_NAME_MAX] = {0};
    get_file_name(input_file[0], input_file_name); // get input[0] filename as output_path
    save_model_results(handle, args, input_file_name, output_data);

exec_finish:
    for (ot_u32 i = 0; i < input_num; i++) {
        if (input_file[i] != NULL) {
            free(input_file[i]);
        }
    }
    free(input_file);
    return ret;
}

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

static bool to_shape_str(char str[], int str_size, const ot_avp_npu_shape* shape) {
    uint32_t dim_size = shape->dim_size;
    if (dim_size < 1U) {
        fprintf(stderr, "shape size is 0\n");
        return false;
    }
    for (uint32_t i = 0; i < dim_size; i++) {
        int len = snprintf(str, str_size, "%d", shape->dims[i]);
        if(len < 0){
            return false;
        }
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

static ot_s32 exec(ot_avp_handle handle, arg_config* opt_arg)
{
    ot_s32 ret = -1;
    ot_u32 input_num, output_num;
    (void)ot_avp_npu_get_input_num(handle, &input_num);
    (void)ot_avp_npu_get_output_num(handle, &output_num);

    // create input/output dataset
    ot_avp_npu_dataset* input_dataset = ot_avp_npu_create_input_dataset(handle);
    ot_avp_npu_dataset* output_dataset = ot_avp_npu_create_output_dataset(handle);

    data_buf* input_data = (data_buf*)malloc(sizeof(data_buf)* input_num);
    data_buf* output_data = (data_buf*)malloc(sizeof(data_buf) * output_num);
    (void)memset(input_data, 0, sizeof(data_buf)* input_num);
    (void)memset(output_data, 0, sizeof(data_buf) * output_num);

    printf("inputs:\n");
    for (ot_u32 i = 0; i < input_num; i++) {
        ot_avp_data_type dtype;
        (void)ot_avp_npu_get_input_dtype_by_index(handle, i, &dtype);
        ot_avp_npu_shape shape;
        (void)ot_avp_npu_get_input_shape_by_index(handle, i, &shape);
        ot_u32 default_stride = ot_avp_npu_get_input_default_stride(handle, i);
        if (default_stride == 0U) {
            printf("stride can't be 0\n");
            goto destory_dataset;
        }
        ot_u32 buf_size = ot_avp_npu_get_input_size_by_index(handle, i);
        if (buf_size == 0U) {
            printf("buffer size can't be 0\n");
            goto destory_dataset;
        }
        char shape_str[1024];
        (void)to_shape_str(shape_str, 1024, &shape);
        printf("\t%d: dtype: %s shape: %s stride: %d size: %d\n", i, DTYPE_ARRAY[dtype], shape_str, default_stride, buf_size);
        ret = sample_avp_npu_malloc(&input_data[i].buf, buf_size);
        if (ret != 0) {
            printf("malloc failed");
            goto destory_dataset;
        }
        (void)memset(input_data[i].buf, 0, buf_size);
        input_data[i].buf_len = buf_size;
        input_data[i].stride = default_stride;
        (void)ot_avp_npu_set_buffer(input_dataset, i, input_data[i].buf, 0, buf_size, default_stride);
    }
    // create output buffer
    printf("outputs:\n");
    for (uint32_t i = 0; i < output_num; i++) {
        ot_avp_data_type dtype;
        (void)ot_avp_npu_get_output_dtype_by_index(handle, i, &dtype);
        ot_avp_npu_shape shape;
        (void)ot_avp_npu_get_output_shape_by_index(handle, i, &shape);
        ot_u32 buf_size = ot_avp_npu_get_output_size_by_index(handle, i);
        if (buf_size == 0U) {
            printf("buffer size can't be 0\n");
            goto destory_dataset;
        }
        ot_u32 default_stride = ot_avp_npu_get_output_default_stride(handle, i);
        if (default_stride == 0U) {
            printf("stride can't be 0\n");
            goto destory_dataset;
        }
        char shape_str[1024];
        (void)to_shape_str(shape_str, 1024, &shape);

#ifndef SIMULATOR
        ot_avp_data_type quant_type = 0;
        ot_avp_npu_quant_param format = {0};
        ot_avp_npu_get_output_quant_by_index(handle, i, &quant_type, &format);
        if (quant_type == OT_AVP_DTYPE_INT8 || quant_type == OT_AVP_DTYPE_UINT8 || quant_type == OT_AVP_DTYPE_INT16) {
            printf("\t%d: dtype: %s shape: %s stride: %d size: %d scale: %f zero_point: %d\n",
                i, DTYPE_ARRAY[dtype], shape_str, default_stride, buf_size, format.scale, format.zero_point);
        } else {
            printf("\t%d: dtype: %s shape: %s stride: %d size: %d\n", i, DTYPE_ARRAY[dtype], shape_str, default_stride, buf_size);
        }
#else
        printf("\t%d: dtype: %s shape: %s stride: %d size: %d\n", i, DTYPE_ARRAY[dtype], shape_str, default_stride, buf_size);
#endif
        ret = sample_avp_npu_malloc(&output_data[i].buf, buf_size);
        if (ret != 0) {
            printf("malloc failed\n");
            goto destory_dataset;
        }
        (void)memset(output_data[i].buf, 0, buf_size);
        output_data[i].buf_len = buf_size;
        output_data[i].stride = default_stride;
        (void)ot_avp_npu_set_buffer(output_dataset, i, output_data[i].buf, 0, buf_size, default_stride);
    }
    if (opt_arg->output_path != NULL) {
        mkdir_dir(opt_arg->output_path);
    }
    exec_args exec_cfg;
    exec_cfg.input_dataset = input_dataset;
    exec_cfg.output_dataset = output_dataset;
    exec_cfg.input_num = (int32_t)input_num;
    exec_cfg.output_num = (int32_t)output_num;
    exec_cfg.opt_arg = opt_arg;
    exec_cfg.output_path = opt_arg->output_path;
    exec_cfg.input_buf = input_data;
    exec_cfg.output_buf = output_data;
    if (opt_arg->input_path == NULL) {
        // perf mode
        exec_cfg.input_path = NULL;
        exec_cfg.output_path = NULL;
        ret = execute_model(handle, &exec_cfg);
        if (ret != 0) {
            printf("execute model fail");
            goto destory_dataset;
        }
    } else if (opt_arg->is_folder) {
        // two-level folder
#ifdef __linux__
        struct dirent* entry;
        DIR *dir = opendir(opt_arg->input_path);
        if (dir == NULL) {
            printf("open dir fail, path:%s\n", opt_arg->input_path);
            goto destory_dataset;
        }
        // while ((entry = readdir(dir)) != NULL) {
        for (entry = readdir(dir); entry != NULL; entry = readdir(dir)) {
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
                continue;
            }
            /* LDRA_EXCLUDE 589 S */
            /* LDRA_EXCLUDE 621 S */
            char input_path[FILE_PATH_MAX] = {0};
            ret = snprintf(input_path, FILE_PATH_MAX, "%s/%s", opt_arg->input_path, entry->d_name);
            if (ret < 0) {
                printf("the length of input file path is too long, max length: [%d]\n", FILE_PATH_MAX);
                goto destory_dataset;
            }
            printf("input_path: %s\n", input_path);
            if (!is_directory(input_path)) {
                printf("input path is not directory, path: %s\n", input_path);
                goto destory_dataset;
            }
/* LDRA_EXCLUDE 71 S */
            exec_cfg.input_path = input_path;
            /* LDRA_EXCLUDE 589 S */
            /* LDRA_EXCLUDE 621 S */
            char output_path[FILE_PATH_MAX] = {0};
            ret = snprintf(output_path, FILE_PATH_MAX, "%s/%s", opt_arg->output_path, entry->d_name);
            if (ret < 0) {
                printf("the length of output file path is too long, max length: [%d]\n", FILE_PATH_MAX);
                goto destory_dataset;
            }
            mkdir_dir(output_path);
/* LDRA_EXCLUDE 71 S */
            exec_cfg.output_path = output_path;
            ret = execute_model(handle, &exec_cfg);
            if (ret != 0) {
                printf("execute model fail");
                goto destory_dataset;
            }
        }
        closedir(dir);
#else
        char tmp_path[FILE_PATH_MAX] = "";
        gen_win_path(opt_arg->input_path, tmp_path, FILE_PATH_MAX);
        printf("find the floder:%s\n", tmp_path);
        WIN32_FIND_DATA fileInfo;
        HANDLE hFind = FindFirstFile(tmp_path, &fileInfo);
        if (INVALID_HANDLE_VALUE == hFind) {
            printf("failed to find files in this directory:%s\n", tmp_path);
            goto destory_dataset;
        }
        while (FindNextFile(hFind, &fileInfo) != 0) {
            if (strcmp(fileInfo.cFileName, ".") == 0
                || strcmp(fileInfo.cFileName, "..") == 0) {
                continue;
            }
            /*---------------- input path ----------------*/
            char input_path[FILE_PATH_MAX] = {0};
            ret = snprintf(input_path, FILE_PATH_MAX, "%s%c%s", opt_arg->input_path, PATH_SEPARATOR, fileInfo.cFileName);
            if (ret < 0) {
                printf("the length of input file path is too long, max length: [%d]\n", FILE_PATH_MAX);
                goto destory_dataset;
            }
            if (!is_directory(input_path)) {
                printf("input path is not directory, path: %s\n", input_path);
                goto destory_dataset;
            }
            exec_cfg.input_path = input_path;
            /*---------------- output path ----------------*/
            char output_path[FILE_PATH_MAX] = {0};
            ret = snprintf(output_path, FILE_PATH_MAX, "%s\\%s", opt_arg->output_path, fileInfo.cFileName);
            if (ret < 0) {
                printf("the length of output file path is too long, max length: [%d]\n", FILE_PATH_MAX);
                goto destory_dataset;
            }
            mkdir_dir(output_path);
            exec_cfg.output_path = output_path;
            ret = execute_model(handle, &exec_cfg);
            if (ret != 0) {
                printf("execute model fail");
                goto destory_dataset;
            }
        }
        FindClose(hFind);
#endif
    } else {
        // one-level folder
        if (input_num > 1U) {
            // multi-input model
            exec_cfg.input_path = opt_arg->input_path;
            ret = execute_model(handle, &exec_cfg);
            if (ret != 0) {
                printf("execute model fail");
                goto destory_dataset;
            }
        } else {
            // single-input model
            struct stat st;
            if (stat(opt_arg->input_path, &st) != 0) {
                printf("input path is not exit, input path:%s\n", opt_arg->input_path);
                goto destory_dataset;
            }
            if (S_ISDIR(st.st_mode)) {
                // multiple image execute
#ifdef __linux__
                struct dirent *entry;
                DIR *dir = opendir(opt_arg->input_path);
                // while ((entry = readdir(dir)) != NULL) {
                for (entry = readdir(dir); entry != NULL; entry = readdir(dir)) {
                    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
                        continue;
                    }
                    /* LDRA_EXCLUDE 589 S */
                    /* LDRA_EXCLUDE 621 S */
                    char input_path[FILE_PATH_MAX] = {0};
                    ret = snprintf(input_path, FILE_PATH_MAX, "%s/%s", opt_arg->input_path, entry->d_name);
                    if (ret < 0) {
                        printf("the length of input file path is too long, max length: [%d]\n", FILE_PATH_MAX);
                        goto destory_dataset;
                    }
/* LDRA_EXCLUDE 71 S */
                    exec_cfg.input_path = input_path;
                    ret = execute_model(handle, &exec_cfg);
                    if (ret != 0) {
                        printf("execute model fail");
                        goto destory_dataset;
                    }
                }
                closedir(dir);
#else
                char tmp_path[FILE_PATH_MAX] = "";
                gen_win_path(opt_arg->input_path, tmp_path, FILE_PATH_MAX);
                WIN32_FIND_DATA fileInfo;
                HANDLE hFind = FindFirstFile(tmp_path, &fileInfo);
                if (INVALID_HANDLE_VALUE == hFind) {
                    printf("failed to find files in this directory:%s\n", tmp_path);
                    goto destory_dataset;
                }
                while (FindNextFile(hFind, &fileInfo) != 0) {
                    if (strcmp(fileInfo.cFileName, ".") == 0
                        || strcmp(fileInfo.cFileName, "..") == 0) {
                        continue;
                    }
                    char input_path[FILE_PATH_MAX] = "";
                    ret = snprintf(input_path, FILE_PATH_MAX, "%s%c%s", opt_arg->input_path, PATH_SEPARATOR, fileInfo.cFileName);
                    if (ret < 0) {
                        printf("the length of input file path is too long, max length: [%d]\n", FILE_PATH_MAX);
                        goto destory_dataset;
                    }
                    exec_cfg.input_path = input_path;
                    printf("exec_cfg input_path:%s\n", exec_cfg.input_path);
                    ret = execute_model(handle, &exec_cfg);
                    if (ret != 0) {
                        printf("execute model fail");
                        goto destory_dataset;
                     }
                }
                FindClose(hFind);
#endif
            } else {
                // single image execute
                exec_cfg.input_path = opt_arg->input_path;
                ret = execute_model(handle, &exec_cfg);
                if (ret != 0) {
                    printf("execute model fail");
                    goto destory_dataset;
                }
            }
        }
    }

destory_dataset:
    (void)ot_avp_npu_destroy_dataset(input_dataset);
    (void)ot_avp_npu_destroy_dataset(output_dataset);
    // free buffer
    for (uint32_t i = 0; i < input_num; i++) {
        if (input_data[i].buf != NULL) {
            (void)sample_avp_npu_free(input_data[i].buf);
        }
    }
    for (uint32_t i = 0; i < output_num; i++) {
        if (output_data[i].buf != NULL) {
            (void)sample_avp_npu_free(output_data[i].buf);
        }
    }
    free(input_data);
    free(output_data);
    return ret;
}

void print_help(const char* prog_name)
{
    printf("Usage: %s [options]\n\n", prog_name);
    printf("Options:\n");
    printf("  %-3s: %s\n", "-h", "Show help information");
    printf("  %-3s: %s\n", "-p", "Show profiling info");
    printf("  %-3s: %s\n", "-M", "Multi-core performance test(can only use -m/-r/-s)");
    printf("  %-3s: %s\n", "-m", "Specify the ortm board/sim model");
    printf("  %-3s: %s\n", "-c", "Bind NPU core(0/1/2). 0 means not bind. 1 means first core. 2 means second core.");
    printf("  %-3s: %s\n", "-r", "Repeat times");
    printf("  %-3s: %s\n", "-s", "Using SRAM size. support 0/1/2/3. Unit is MB");
    printf("  %-3s: %s\n", "-i", "Specify input file/directory");
    printf("  %-3s: %s\n", "-o", "Specify output directory");
    printf("  %-3s: %s\n", "-d", "Enable dump mode");
    printf("  %-3s: %s\n", "-t", "Target Dump with binary/txt(0/1). 0 means binary. 1 means txt");
}

int main(int argc, char** argv)
{
#if defined(__linux__)
    sample_avp_npu_register_sig_handler(sample_avp_npu_handle_sig);
#endif
    arg_config opt_arg = {NULL, NULL, NULL, false, false, false, 1U, 0U, 0U, 0U, false};
    if (-1 == parse_optarg(argc, argv, &opt_arg)) {
        fprintf(stderr, "parse arguments failed, exiting.\n");
        print_help(argv[0]);
        return -1;
    }

    extra_config extra_npu_config[2];
    const char* layer_path_key   = "DUMP_PATH";
    const char* layer_path_value = "LayerDump";
    if (layer_path_key != NULL) {
        (void)memcpy(&(extra_npu_config[0].key), (void*)layer_path_key, strlen(layer_path_key) + 1U);
    }
    if (layer_path_value != NULL) {
        (void)memcpy(&(extra_npu_config[0].value), (void*)layer_path_value, strlen(layer_path_value) + 1U);
    }
#ifdef SIMULATOR
    char* quant_path_key = "QUANT_DUMP_PATH";
    char* quant_path_value = "QuantDump";
    (void)memcpy(&(extra_npu_config[1].key), quant_path_key, strlen(quant_path_key) + 1U);
    (void)memcpy(&(extra_npu_config[1].value), quant_path_value, strlen(quant_path_value) + 1U);
    opt_arg.repeat = 1;
#else
    (void)sample_avp_npu_media_system_init();
#endif

    ot_avp_npu_config config = {
        .core_ids = 0,
        .dump_layer = opt_arg.dump_layer,   // dump layer
        .perf_mode = opt_arg.perf_mode,  // perf_mode
        .log_level = 2,   // log level info
        .extra_config_num = 2,
        .extra_configs = extra_npu_config,
    };

    if (opt_arg.model_path == NULL) {
        printf("get model path failed, please set parameter [-m]\n");
        print_help(argv[0]);
        return -1;
    }
    printf("ortm_run_model start\n");
    ot_avp_handle handle = NULL;

    int32_t ret = ot_avp_npu_init(&config);
    if (ret != 0) {
        printf("npu init failed\n");
        goto fail;
    }

    // load model
    ret = ot_avp_npu_load_model(opt_arg.model_path, &handle);
    if (ret != 0) {
        printf("open model failed\n");
        goto fail;
    }

    ot_u32 runtime_buf_size = 0;
    (void)ot_avp_npu_get_runtime_buffer_size(handle, &runtime_buf_size);
    printf("runtime buf size: %d\n", runtime_buf_size);

    void *runtime_buf = NULL;
    if (runtime_buf_size != 0U ) {
#ifdef SIMULATOR
        ret = sample_avp_npu_malloc(&runtime_buf, runtime_buf_size);
        if (ret != 0) {
            printf("malloc runtime buf fail\n");
            goto destory_buffer;
        }
        ret = ot_avp_npu_set_runtime_buffer(handle, runtime_buf, runtime_buf_size);
        if (ret != 0) {
            printf("ot_avp_npu_set_runtime_buffer fail\n");
            goto destory_buffer;
        }
#else
        ret = sample_avp_npu_set_runtime_buffer(handle,&runtime_buf,runtime_buf_size,opt_arg.buffer_type);
        if (ret != 0) {
            printf("sample_avp_npu_set_runtime_buffer fail\n");
            goto destory_buffer;
        }
#endif
    }
    if (opt_arg.core_ids != 0U) {
        ret = ot_avp_npu_bind_model_to_core(handle, opt_arg.core_ids);
        if (ret != 0) {
            printf("ot_avp_npu_bind_model_to_core fail\n");
            goto destory_buffer;
        }
    } else {
        opt_arg.core_ids = config.core_ids;
    }
    ret = exec(handle, &opt_arg);
    if (ret != 0) {
        printf("model execute fail\n");
    }
    printf("ortm_run_model end\n");
destory_buffer:
    if (runtime_buf != NULL) {
        (void)sample_avp_npu_free(runtime_buf);
    }
fail:
    (void)ot_avp_npu_unload_model(handle);
    (void)ot_avp_npu_deinit();
#ifndef SIMULATOR
    (void)sample_avp_npu_media_system_deinit();
#endif
    return ret;
}
