#ifndef SIMULATOR
#include <dirent.h>
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "file_utils.h"
#include "ot_avp_npu_rts.h"
#include "ot_smr.h"
#include "sample_avp_npu_common.h"

typedef struct cfg_param {
    char  *model_path[2];
    ot_u32 repeat_times;
    bool   use_sram;
} cfg_param;

static cfg_param g_cfg_param = {0};

static ot_u64 end_time[2];
static ot_u64 start_time[2];

#include <sys/time.h>
extern volatile sig_atomic_t g_exit_flag;

ot_u64 get_current_system_time_us()
{
    struct timeval tv;
    gettimeofday(&tv, 0);
    ot_u64 time = (tv.tv_sec * (ot_u64)1000000) + tv.tv_usec;

    return time;
}

void *execute_first_model_thread(void *param)
{
    sample_avp_model_task_info_t *model_info = (sample_avp_model_task_info_t *)param;
    start_time[0]                            = get_current_system_time_us();
    ot_u32 exec_num = 0;
    while (true) {
        exec_num++;
        if (g_cfg_param.repeat_times != (ot_u32)-1 && exec_num > g_cfg_param.repeat_times) {
            break;
        }
        if (g_exit_flag == 1) {
            break;
        }
        ot_s32 ret = ot_avp_npu_execute(model_info->handle, model_info->input_dataset, model_info->output_dataset);
        SAMPLE_AVP_CHECK_RETURN_WITH_LOG(ret != 0, NULL, "execute_first_model_thread execute fail.");
        if (g_cfg_param.repeat_times == (ot_u32)-1 && (exec_num % 5000U) == 0U) {
            printf("model: [%s], exec %d times \n",  g_cfg_param.model_path[0], exec_num);
        }
    }
    end_time[0] = get_current_system_time_us();

    return 0;
}

void *execute_second_model_thread(void *param)
{
    sample_avp_model_task_info_t *model_info = (sample_avp_model_task_info_t *)param;
    start_time[1]                            = get_current_system_time_us();
    ot_u32 exec_num = 0;
    while (true) {
        exec_num++;
        if (g_cfg_param.repeat_times != (ot_u32)-1 && exec_num > g_cfg_param.repeat_times) {
            break;
        }
        if (g_exit_flag == 1) {
            break;
        }
        ot_s32 ret = ot_avp_npu_execute(model_info->handle, model_info->input_dataset, model_info->output_dataset);
        SAMPLE_AVP_CHECK_RETURN_WITH_LOG(ret != 0, NULL, "execute_second_model_thread execute fail.");
        if (g_cfg_param.repeat_times == (ot_u32)-1 && (exec_num % 5000U) == 0U) {
            printf("model: [%s], exec %d times\n",  g_cfg_param.model_path[1], exec_num);
        }
    }
    end_time[1] = get_current_system_time_us();

    return 0;
}
void print_help(const char* prog_name)
{
    printf("Usage: %s [options]\n\n", prog_name);
    printf("Options:\n");
    printf("  %-3s: %s\n", "-h", "Show help information");
    printf("  %-3s: %s\n", "-M", "Multi-core performance test(can only use -m/-r/-s)");
    printf("  %-3s: %s\n", "-m", "Specify the ortm board/sim model");
    printf("  %-3s: %s\n", "-r", "Repeat times");
    printf("  %-3s: %s\n", "-s", "Using SRAM size.");
}
ot_s32 get_cfg_param(ot_s32 argc, char **argv)
{
    ot_s32 opt       = 0;
    char  *optstring = "hm:r:sM";
    while ((opt = getopt(argc, argv, optstring)) != -1) {
        switch (opt) {
            case 'm':
                if (optind + 1 <= argc && argv[optind][0] != '-') {
                    g_cfg_param.model_path[0] = optarg;
                    g_cfg_param.model_path[1] = argv[optind];
                    optind++;  // 跳过下一个参数
                } else {
                    g_cfg_param.model_path[0] = optarg;
                    g_cfg_param.model_path[1] = optarg;
                }
                break;
            case 'r':
                g_cfg_param.repeat_times = atoi(optarg);
                break;
            case 's':
                g_cfg_param.use_sram = true;
                break;
            case 'M':
                printf("multi core test\n");
                break;
            case 'h':
                print_help(argv[0]);
                return -1;
            default:
                printf("no support arg, please reference sample_npu -m xxx.otrm -r xxx");
                return -1;
        }
    }
    printf("model file: %s %s, repeat_times=%u\n", g_cfg_param.model_path[0], g_cfg_param.model_path[1],
           g_cfg_param.repeat_times);
    return 0;
}

ot_s32 set_runtime_buf(sample_avp_model_task_info_t *model_info, ot_u8 core)
{
    ot_u32 runtime_buf_size = 0;
    (void)ot_avp_npu_get_runtime_buffer_size(model_info->handle, &runtime_buf_size);
    printf("runtime buf size: %d\n", runtime_buf_size);

    if (runtime_buf_size != 0U) {
        ot_s32 ret = sample_avp_npu_malloc(&model_info->runtime_buf, runtime_buf_size);
        if (ret != 0) {
            printf("malloc runtime buf fail\n");
            return -1;
        }
        ot_phys_addr phys_addr = (ot_phys_addr)OT_NULL;
        ot_smr_virt2phy(model_info->runtime_buf, (ot_phys_addr *)&phys_addr);
        if (g_cfg_param.use_sram == true) {
            ot_u64 sram_base_phys_addr = 0x01800000 + core * 0x180000;  // app sram
            ot_u32 sram_size           = 0x180000;
            runtime_buf_size           = (sram_size < runtime_buf_size) ? (runtime_buf_size - sram_size) : 0;
            ret = ot_avp_npu_set_runtime_buffer_with_sram(model_info->handle, sram_base_phys_addr, sram_size,
                                                          (ot_u64)phys_addr, runtime_buf_size);
            printf("use sram addr : %#llx size: %#x\n", sram_base_phys_addr, sram_size);
        } else {
            ret =
                ot_avp_npu_set_runtime_buffer_with_sram(model_info->handle, 0, 0, (ot_u64)phys_addr, runtime_buf_size);
        }
        if (ret != 0) {
            printf("ot_avp_npu_set_runtime_buffer fail\n");
            return -1;
        }
    }
    return 0;
}

ot_s32 multi_core_exec(ot_s32 argc, char **argv)
{
    // 设置环境变量

    g_cfg_param.repeat_times = 1;
    ot_s32 ret               = get_cfg_param(argc, argv);
    SAMPLE_AVP_CHECK_RETURN_WITH_LOG(ret != 0, ret, "get_cfg_param fail");

    ot_avp_npu_config config = {
        .core_ids  = 3,
        .log_level = 2,
    };

    sample_avp_npu_media_system_init();

    ret = ot_avp_npu_init(&config);
    SAMPLE_AVP_CHECK_GOTO_WITH_LOG(ret != 0, init_fail, "npu init error.");

    sample_avp_model_task_info_t model_info[2];
    for (ot_u8 i = 0U; i < 2U; i++) {
        memset(&model_info[i], 0, sizeof(sample_avp_model_task_info_t));
        ret = ot_avp_npu_load_model(g_cfg_param.model_path[i], &model_info[i].handle);
        SAMPLE_AVP_CHECK_GOTO_WITH_LOG(ret != 0, load_model_fail, "load model fail.");
        sample_avp_create_datasets(model_info[i].handle, &model_info[i]);
        set_runtime_buf(&model_info[i], i);
    }
    ret = ot_avp_npu_bind_model_to_core(model_info[0].handle, 1);
    SAMPLE_AVP_CHECK_GOTO_WITH_LOG(ret != 0, bind_model_fail, "bind model to 1 fail.");
    ret = ot_avp_npu_bind_model_to_core(model_info[1].handle, 2);
    SAMPLE_AVP_CHECK_GOTO_WITH_LOG(ret != 0, bind_model_fail, "bind model to 2 fail.");

    pthread_t pth_id[2];
    pthread_create(&pth_id[0], NULL, execute_first_model_thread, (void *)&model_info[0]);
    pthread_create(&pth_id[1], NULL, execute_second_model_thread, (void *)&model_info[1]);

    pthread_join(pth_id[0], NULL);
    pthread_join(pth_id[1], NULL);

    double total_time = (double)(fmax(end_time[0], end_time[1]) - fmin(start_time[0], start_time[1])) / 1000.0;
    double fps        = 1000.0 / (total_time / (2 * g_cfg_param.repeat_times));

    printf("\n==== oritek execute result ====\n");
    printf("avg_time: %.3f ms, fps: %.3f, repeat: %d\n\n", total_time / (2 * g_cfg_param.repeat_times), fps,
           g_cfg_param.repeat_times);

bind_model_fail:
load_model_fail:
    for (int i = 0; i < 2; i++) {
        sample_avp_destroy_datasets(model_info[i].handle, &model_info[i]);
        ot_avp_npu_unload_model(model_info[i].handle);
        if (model_info[i].runtime_buf != NULL) {
            (void)sample_avp_npu_free(model_info[i].runtime_buf);
        }
    }
    ot_avp_npu_deinit();
init_fail:
    sample_avp_npu_media_system_deinit();

    if (ret != 0) {
        printf("%s board execute ERROR\n", g_cfg_param.model_path[0]);
        printf("%s board execute ERROR\n", g_cfg_param.model_path[1]);
        exit(1);
    }
    return ret;
}
#endif