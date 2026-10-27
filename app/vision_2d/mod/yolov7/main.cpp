#include "sample_avp_npu_common.h"
#include "utils.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "opt_args.h"
#ifdef __GNUC__
#include <unistd.h>
#endif
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#endif
#ifndef SIMULATOR
#include <arm_neon.h>
#include "platform_utils.h"
#else
#include "simulator_utils.h"
#endif

ot_s32 sample_yolov7_execute(sample_avp_model_task_info_t* model_info, const char* input_path)
{
    int32_t ret = -1;

    ret = load_data_from_file(input_path, (uint8_t *)model_info->input_tensor[0].virt_addr, model_info->input_tensor[0].shape,
        model_info->input_tensor[0].stride, model_info->input_tensor[0].dtype);

    // 模型推理
    ret = sample_avp_execute_model(model_info, 1);  // repeat_time = 1
    SAMPLE_AVP_CHECK_GOTO_WITH_LOG(ret != 0, EXECUTE_MODEL_FAIL, "execute model fail.\n");

    sample_npu_context_t context;
    memset(&context, 0, sizeof(sample_npu_context_t));
    context.model_info = *model_info;
    for (ot_u32 i = 0U; i < model_info->output_num; i++) {
        if (model_info->output_tensor[i].dtype == OT_AVP_DTYPE_F16) {
            sample_npu_yolov7_decode_model_float16_output_data(&context, (ot_u8 *)model_info->output_tensor[i].virt_addr, i,
                model_info->output_tensor[i].size, &model_info->output_tensor[i].shape, model_info->output_tensor[i].stride);
        } else if (model_info->output_tensor[i].dtype == OT_AVP_DTYPE_F32) {
            sample_npu_yolov7_decode_model_float32_output_data(&context, (ot_u8 *)model_info->output_tensor[i].virt_addr, i,
                model_info->output_tensor[i].size, &model_info->output_tensor[i].shape, model_info->output_tensor[i].stride);
        } else {
            printf("output only support fp32 and fp16 \n");
            return -1;
        }
    }
    sample_npu_yolov7_nms(&context);

EXECUTE_MODEL_FAIL:

    return ret;
}

int sample_yolov7_run(const char *input_path, const char *model_path)
{
    int ret = -1;
    sample_avp_model_task_info_t yolov7_info;
    memset(&yolov7_info, 0, sizeof(sample_avp_model_task_info_t));

    // load model
    ret = ot_avp_npu_load_model(model_path, &yolov7_info.handle);
    SAMPLE_AVP_CHECK_RETURN_WITH_LOG(ret != 0, ret, "load model fail.\n");

    sample_avp_create_datasets(yolov7_info.handle, &yolov7_info);

    ret = sample_yolov7_execute(&yolov7_info, input_path);
    SAMPLE_AVP_CHECK_GOTO_WITH_LOG(ret != 0, EXECUTE_MODEL_FAIL, "sample_yolov7_execute fail.\n");

EXECUTE_MODEL_FAIL:
    sample_avp_destroy_datasets(yolov7_info.handle, &yolov7_info);
    ot_avp_npu_unload_model(yolov7_info.handle);
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
#if defined(_WIN32) || defined(_WIN64)
    const char* model_path = "..\\yolov7.ortm";
    const char* input_path = "..\\000000145690.yuv420sp";
#else
    const char* model_path = "./tmp_models/yolov7.ortm";
    const char* input_path = "./tmp_res/infer_data/000000145690.yuv420sp";
#endif
#else

    ret = sample_avp_npu_media_system_init();
    const char* model_path = "../../../../data/prebuild_ortm/yolov7.ortm";
    const char* input_path = "../../../../data/ImageNet/yuv420sp/000000145690.yuv420sp";
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

    ret = sample_yolov7_run(input_path, model_path);
    SAMPLE_AVP_CHECK_GOTO_WITH_LOG(ret != 0, NPU_RUN_FAIL, "sample yolov7 run fail.\n");

NPU_RUN_FAIL:
    ot_avp_npu_deinit();

NPU_INIT_FAIL:
#ifndef SIMULATOR
    sample_avp_npu_media_system_deinit();
#endif
    return 0;
}
