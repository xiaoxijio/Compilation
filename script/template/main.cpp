#include "sample_avp_npu_common.h"
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include "opt_args.h"
#include <sys/stat.h>
#ifdef __GNUC__
#include <unistd.h>
#endif
#ifndef SIMULATOR
#include "platform_utils.h"
#else
#include "simulator_utils.h"
#endif

ot_s32 sample_{{MODEL_NAME}}_execute(sample_avp_model_task_info_t* model_info, const char* input_path)
{
    int32_t ret = -1;
    ot_u32 output_num, input_num;
    struct stat info;
    ot_avp_npu_get_input_num(model_info->handle, &input_num);
    ot_avp_npu_get_output_num(model_info->handle, &output_num);

    if (stat(input_path, &info) == 0) {
        if (S_ISREG(info.st_mode)) {
            ret = load_data_from_file(input_path, (uint8_t *)model_info->input_tensor[0].virt_addr, model_info->input_tensor[0].shape,
                model_info->input_tensor[0].stride, model_info->input_tensor[0].dtype);
        } else if (S_ISDIR(info.st_mode)) {
            char** input_file = (char**)malloc(input_num * sizeof(char*));
            for (ot_u32 i = 0U; i < input_num; i++) {
                input_file[i] = (char*)malloc(FILE_PATH_MAX);
                (void)memset(input_file[i], 0, FILE_PATH_MAX);
            }
            ret = get_input_file(input_path, input_file, input_num);
            if (ret != 0) {
                printf("get input file failed\n");
                goto EXECUTE_MODEL_FAIL;
            }
            for (ot_u32 i = 0U; i < input_num; i++) {
                ret = load_data_from_file(input_file[i], (uint8_t *)model_info->input_tensor[i].virt_addr,
                    model_info->input_tensor[i].shape, model_info->input_tensor[i].stride, model_info->input_tensor[i].dtype);
            }
        }
    }
    // 模型推理
    ret = sample_avp_execute_model(model_info, 1);  // repeat_time = 1
    SAMPLE_AVP_CHECK_GOTO_WITH_LOG(ret != 0, EXECUTE_MODEL_FAIL, "execute model fail.\n");

    // DUMP模型输出
    for (ot_u32 i = 0U; i < output_num; i++) {
        char output_file[FILE_PATH_MAX] = {0};
        ret = 0;
        ret = snprintf(output_file, FILE_PATH_MAX, "./output_%d.bin", i);
        ret = dump_data_to_binary_file(output_file, (uint8_t *)model_info->output_tensor[i].virt_addr,
            model_info->output_tensor[i].shape, model_info->output_tensor[i].stride, model_info->output_tensor[i].dtype);
        if (ret < 0) {
            printf("dump output fail, output file is too long\n");
            goto EXECUTE_MODEL_FAIL;
        }
    }
    // 可继续开发模型后处理相关代码

EXECUTE_MODEL_FAIL:

    return ret;
}

int sample_{{MODEL_NAME}}_run(const char *input_path, const char *model_path)
{
    int ret = -1;
    sample_avp_model_task_info_t {{MODEL_NAME}}_info;
    memset(&{{MODEL_NAME}}_info, 0, sizeof(sample_avp_model_task_info_t));

    // load model
    ret = ot_avp_npu_load_model(model_path, &{{MODEL_NAME}}_info.handle);
    SAMPLE_AVP_CHECK_RETURN_WITH_LOG(ret != 0, ret, "load model fail.\n");

    sample_avp_create_datasets({{MODEL_NAME}}_info.handle, &{{MODEL_NAME}}_info);

    ret = sample_{{MODEL_NAME}}_execute(&{{MODEL_NAME}}_info, input_path);
    SAMPLE_AVP_CHECK_GOTO_WITH_LOG(ret != 0, EXECUTE_MODEL_FAIL, "{{MODEL_NAME}}_execute fail.\n");

EXECUTE_MODEL_FAIL:
    sample_avp_destroy_datasets({{MODEL_NAME}}_info.handle, &{{MODEL_NAME}}_info);
    ot_avp_npu_unload_model({{MODEL_NAME}}_info.handle);
    return ret;
}

int main(int argc, char** argv)
{
    arg_config opt_arg = {NULL, NULL, NULL, false, false, false, 1U, 0U, 0U, 0U, false};
    if (-1 == parse_optarg(argc, argv, &opt_arg)) {
        fprintf(stderr, "parse arguments failed, exiting.\n");
        return -1;
    }
    int ret = -1;

    ot_avp_npu_config config = {
        .core_ids = 0U,
        .dump_layer = OT_FALSE,   // dump layer
        .perf_mode = OT_FALSE,  // perf_mode
        .log_level = 2U,   // log level
        .thread_num = 0U,
        .extra_config_num = 0U,
        .extra_configs = NULL,
    };

#ifdef SIMULATOR
    const char* model_path = "./tmp_models/{{MODEL_NAME}}.ortm";
    const char* input_path = "./tmp_res/infer_data";
#else
    const char* model_path = "../../../../data/prebuild_ortm/{{MODEL_NAME}}.ortm";
    const char* input_path = "../../../../data/ImageNet";
    ret = sample_avp_npu_media_system_init();
    if (ret == 0) {
        printf("system_init success.\n");
    }
#endif
    if (opt_arg.input_path != NULL) {
        input_path = opt_arg.input_path;
    }
    if (opt_arg.model_path != NULL) {
        model_path = opt_arg.model_path;
    }

    ret = ot_avp_npu_init(&config);
    SAMPLE_AVP_CHECK_GOTO_WITH_LOG(ret != 0, NPU_INIT_FAIL, "npu init fail.\n");

    ret = sample_{{MODEL_NAME}}_run(input_path, model_path);
    SAMPLE_AVP_CHECK_GOTO_WITH_LOG(ret != 0, NPU_RUN_FAIL, "sample {{MODEL_NAME}} run fail.\n");

NPU_RUN_FAIL:
    ot_avp_npu_deinit();

NPU_INIT_FAIL:
#ifndef SIMULATOR
    sample_avp_npu_media_system_deinit();
#endif
    return 0;
}
