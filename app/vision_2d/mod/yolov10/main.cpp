#include "sample_avp_npu_common.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "utils.h"
#include <algorithm>
#include <iostream>
#include "opt_args.h"
#include <map>
#include <vector>
#include <utility>
#include <vector>
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
#include <cfloat> // for FLT_MAX

int sample_yolov10_run(const char *input_path, const char *model_path)
{
    int ret = -1;
    sample_avp_model_task_info_t yolov10_info;
    memset(&yolov10_info, 0, sizeof(sample_avp_model_task_info_t));

    // load model
    ret = ot_avp_npu_load_model(model_path, &yolov10_info.handle);
    SAMPLE_AVP_CHECK_RETURN_WITH_LOG(ret != 0, ret, "load model fail.\n");

    sample_avp_create_datasets(yolov10_info.handle, &yolov10_info);

    ret = sample_yolo_execute(&yolov10_info, input_path);
    SAMPLE_AVP_CHECK_GOTO_WITH_LOG(ret != 0, EXECUTE_MODEL_FAIL, "sample_yolov10_execute fail.\n");

EXECUTE_MODEL_FAIL:
    sample_avp_destroy_datasets(yolov10_info.handle, &yolov10_info);
    ot_avp_npu_unload_model(yolov10_info.handle);
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
    const char* model_path = "..\\yolov10s.ortm";
    const char* input_path = "..\\000000145690.yuv420sp";
#else
    const char* model_path = "./tmp_models/yolov10s.ortm";
    const char* input_path = "./tmp_res/infer_data/000000145690.yuv420sp";
#endif
#else

    ret = sample_avp_npu_media_system_init();
    const char* model_path = "../../../../data/prebuild_ortm/yolov10s.ortm";
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

    ret = sample_yolov10_run(input_path, model_path);
    SAMPLE_AVP_CHECK_GOTO_WITH_LOG(ret != 0, NPU_RUN_FAIL, "sample yolov10 run fail.\n");

NPU_RUN_FAIL:
    ot_avp_npu_deinit();

NPU_INIT_FAIL:
#ifndef SIMULATOR
    sample_avp_npu_media_system_deinit();
#endif
    return 0;
}
