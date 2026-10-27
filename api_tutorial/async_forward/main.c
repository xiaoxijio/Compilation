#include "ot_avp_npu_rts.h"
#include "file_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __GNUC__
#include <unistd.h>
#endif
#ifndef SIMULATOR
#include "platform_utils.h"
#else
#include "simulator_utils.h"
#endif
#include "sample_avp_npu_common.h"
#define MODEL_NUM 2
/*
 * 支持用法1：model_0 trigger + model_1 trigger + model_1 wait
 * 支持用法2：model_0 trigger + model_0 wait + model_1 trigger + model_1 wait
 */
 static ot_s32 sample_npu_async_inference_both_single_graph(int argc, char** argv)
 {
     (void)argc;
     (void)argv;
     ot_avp_npu_config config = {
         .core_ids = 0,
         .dump_layer = false,   // dump layer
         .perf_mode = false,  // perf_mode
         .log_level = 2,   // log level
         .thread_num = 2,
     };
     ot_s32 ret = ot_avp_npu_init(&config);
     SAMPLE_AVP_CHECK_RETURN_WITH_LOG(ret != 0, ret, "npu init fail.\n");
     const int model_num = MODEL_NUM;
     /* LDRA_EXCLUDE 621 S */
     sample_avp_model_task_info_t model_info[MODEL_NUM];
     const char* model_path[] =
         {"../../data/prebuild_ortm/resnet50_binary.ortm",
          "../../data/prebuild_ortm/resnet50_binary.ortm"};

     // 模型加载, 创建输入输出数据
     for (int i = 0; i < model_num; i++) {
         (void)memset(&model_info[i], 0, sizeof(sample_avp_model_task_info_t));
         if (ot_avp_npu_load_model(model_path[i], &model_info[i].handle) != 0) {
             printf("load model[%d] fail\n", i);
         }
         sample_avp_create_datasets(model_info[i].handle, &model_info[i]);
     }

     // 支持用法1：model_0 trigger + model_1 trigger + model_1 wait
     for (int i = 0; i < model_num; i++) {
         ret = ot_avp_npu_trigger(model_info[i].handle, model_info[i].input_dataset, model_info[i].output_dataset);
         if (ret != 0) {
             printf("trigger model[%d] fail.\n", i);
         }
     }
     ret = ot_avp_npu_wait(model_info[model_num - 1].handle, -1);
     if (ret != 0) {
         printf("wait model fail.\n");
     }

     // 支持用法2：model_0 trigger + model_0 wait + model_1 trigger + model_1 wait
     for (int i = 0; i < model_num; i++) {
         ret = ot_avp_npu_trigger(model_info[i].handle, model_info[i].input_dataset, model_info[i].output_dataset);
         if (ret != 0) {
             printf("trigger model[%d] fail.\n", i);
         }
         ret = ot_avp_npu_wait(model_info[i].handle, -1);
         if (ret != 0) {
             printf("wait model fail.\n");
         }
     }

     // 销毁输入输出数据, 模型卸载
     for (int i = 0; i < model_num; i++) {
         sample_avp_destroy_datasets(model_info[i].handle, &model_info[i]);
         (void)ot_avp_npu_unload_model(model_info[i].handle);
     }

     printf("%s end.\n", __FUNCTION__);

     (void)ot_avp_npu_deinit();

     return ret;
 }

 /*
  * 支持用法1：model_0 trigger + model_0 wait + model_1 trigger + model_1 wait
  */
 static ot_s32 sample_npu_async_inference_mix_single_and_multi_graph(int argc, char** argv)
 {
     (void)argc;
     (void)argv;
     ot_avp_npu_config config = {
         .core_ids = 0,
         .dump_layer = false,   // dump layer
         .perf_mode = false,  // perf_mode
         .log_level = 2,   // log level
         .thread_num = 2,
     };
     ot_s32 ret = ot_avp_npu_init(&config);
     SAMPLE_AVP_CHECK_RETURN_WITH_LOG(ret != 0, ret, "npu init fail.\n");

     const int model_num = MODEL_NUM;
     /* LDRA_EXCLUDE 621 S */
     sample_avp_model_task_info_t model_info[MODEL_NUM];
     const char* model_path[] =
         {"../../data/prebuild_ortm/resnet50_binary.ortm",
          "../../data/prebuild_ortm/resnet50_binary.ortm"};

     // 模型加载, 创建输入输出数据
     for (int i = 0; i < model_num; i++) {
         (void)memset(&model_info[i], 0, sizeof(sample_avp_model_task_info_t));
         if (ot_avp_npu_load_model(model_path[i], &model_info[i].handle) != 0) {
             printf("load model[%d] fail\n", i);
         }
         sample_avp_create_datasets(model_info[i].handle, &model_info[i]);
     }

     // 支持用法1：model_0 trigger + model_0 wait + model_1 trigger + model_1 wait
     for (int i = 0; i < model_num; i++) {
         ret = ot_avp_npu_trigger(model_info[i].handle, model_info[i].input_dataset, model_info[i].output_dataset);
         if (ret != 0) {
             printf("trigger model[%d] fail.\n", i);
         }
         ret = ot_avp_npu_wait(model_info[i].handle, -1);
         if (ret != 0) {
             printf("wait model fail.\n");
         }
     }

     // 销毁输入输出数据, 模型卸载
     for (int i = 0; i < model_num; i++) {
         sample_avp_destroy_datasets(model_info[i].handle, &model_info[i]);
         (void)ot_avp_npu_unload_model(model_info[i].handle);
     }

     printf("%s end.\n", __FUNCTION__);

     (void)ot_avp_npu_deinit();

     return ret;
 }

typedef ot_s32 (*sample_process_func)(int argc, char** argv);

typedef struct sample_execute_t{
    ot_s32 index;
    sample_process_func func;
} sample_execute_t;

/* function : show usage */
static ot_void sample_npu_usage(const ot_char *name)
{
    printf("  Usage : %s <index> \n", name);
    printf("  index :\n");
    printf("          1) async inference. [single_npu_graph_model + single_npu_graph_model]\n");
    printf("          2) async inference. [single_graph_model + multi_graph_model ]\n");
}

static ot_s32 sample_npu_run_case(int argc, char** argv)
{
    ot_s32 index = atoi(argv[1]);
    static sample_execute_t s_execute[] = {
        {0, NULL},
        {1, sample_npu_async_inference_both_single_graph},
        {2, sample_npu_async_inference_mix_single_and_multi_graph},

    };

    if (index > (ot_s32)((sizeof(s_execute) / sizeof(sample_execute_t)))) {
        printf("index[%d] is invalid", index);
        return -1;
    }

    return s_execute[index].func(argc, argv);
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        sample_npu_usage(argv[0]);
        return -1;
    }

    (void)sample_avp_npu_media_system_init();

    int ret = sample_npu_run_case(argc, argv);

    (void)sample_avp_npu_media_system_deinit();

    if (ret != 0) {
        sample_npu_usage(argv[0]);
        return -1;
    }
    return 0;
}
