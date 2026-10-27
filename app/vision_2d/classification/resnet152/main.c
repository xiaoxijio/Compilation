#include "sample_avp_npu_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "opt_args.h"
#ifdef __GNUC__
#include <unistd.h>
#endif
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#endif
#ifndef SIMULATOR
#include "platform_utils.h"
#else
#include "simulator_utils.h"
#endif
#include "classify_utils.h"

ot_s32 sample_resnet152_execute(sample_avp_model_task_info_t* model_info, const char* input_path)
{
    int32_t ret = -1;

    ret = load_data_from_file(input_path, (uint8_t *)model_info->input_tensor[0].virt_addr, model_info->input_tensor[0].shape,
        model_info->input_tensor[0].stride, model_info->input_tensor[0].dtype);

    // 模型推理
    ret = sample_avp_execute_model(model_info, 1);  // repeat_time = 1
    SAMPLE_AVP_CHECK_GOTO_WITH_LOG(ret != 0, EXECUTE_MODEL_FAIL, "execute model fail.\n");

    uint32_t num_classes = model_info->output_tensor[0].size / sizeof(float);
    ret = compute_and_display_topk((float*)model_info->output_tensor[0].virt_addr, num_classes, 5U);

EXECUTE_MODEL_FAIL:
    return ret;
}

int sample_resnet152_run(const char *input_path, const char *model_path)
{
    int ret = -1;
    sample_avp_model_task_info_t resnet152_info;
    // sample_avp_npu_rect_t npu_rect = {0};
    memset(&resnet152_info, 0, sizeof(sample_avp_model_task_info_t));

    // load model
    ret = ot_avp_npu_load_model(model_path, &resnet152_info.handle);
    SAMPLE_AVP_CHECK_RETURN_WITH_LOG(ret != 0, ret, "load model fail.\n");

    sample_avp_create_datasets(resnet152_info.handle, &resnet152_info);

    ret = sample_resnet152_execute(&resnet152_info, input_path);
    SAMPLE_AVP_CHECK_GOTO_WITH_LOG(ret != 0, EXECUTE_MODEL_FAIL, "sample_resnet152_execute fail.\n");

EXECUTE_MODEL_FAIL:
    sample_avp_destroy_datasets(resnet152_info.handle, &resnet152_info);
    ot_avp_npu_unload_model(resnet152_info.handle);
    return ret;
}

int main(int argc, char** argv)
{
    arg_config opt_arg = {NULL, NULL, NULL, OT_FALSE, OT_FALSE, OT_FALSE, 1U, 0U, 0U, 0U, false};
    if (-1 == parse_optarg(argc, argv, &opt_arg)) {
        fprintf(stderr, "parse arguments failed, exiting.\n");
        return -1;
    }
    int ret = -1;

    ot_avp_npu_config config = {
        .core_ids = 0,
        .dump_layer = OT_FALSE,   // dump layer
        .perf_mode = OT_FALSE,  // perf_mode
        .log_level = 2,   // log level info
    };
#ifdef SIMULATOR
#if defined(_WIN32) || defined(_WIN64)
    const char* model_path = "..\\resnet152.ortm";
    const char* input_path = "..\\n02009912_262.yuv420sp";
#else
    const char* model_path = "./tmp_models/resnet152.ortm";
    const char* input_path = "./tmp_res/infer_data/n02009912_262.yuv420sp";
#endif
#else

    ret = sample_avp_npu_media_system_init();
    const char* model_path = "../../../../data/prebuild_ortm/resnet152.ortm";
    const char* input_path = "../../../../data/ImageNet/yuv420sp/n02009912_262.yuv420sp";
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

    ret = sample_resnet152_run(input_path, model_path);
    SAMPLE_AVP_CHECK_GOTO_WITH_LOG(ret != 0, NPU_RUN_FAIL, "sample resnet152 run fail.\n");

NPU_RUN_FAIL:
    ot_avp_npu_deinit();

NPU_INIT_FAIL:
#ifndef SIMULATOR
    sample_avp_npu_media_system_deinit();
#endif
    return 0;
}
