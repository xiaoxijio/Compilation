#include "sample_avp_npu_common.h"
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <iostream>
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

static void sample_yolov5_parse_bbox(char* buf, int output_boxnum, int bbox_num, sample_avp_npu_rect_t* rect)
{
    if (bbox_num > output_boxnum) {
        printf("bbox_num[%u] is invalid.\n", bbox_num);
        return;
    }
    const int bbox_attr_size = 6;    // x_min y_min x_max y_max conf classid
    for (int i = 0; i < bbox_num; i++) {
        float rect_tmp[bbox_attr_size];
        for (int j = 0; j < bbox_attr_size; j++) {
            rect_tmp[j] = *(float *)(buf + j * output_boxnum * sizeof(float) + i * sizeof(float));
        }
        rect->out_rect[bbox_num - 1 - i].x1 = rect_tmp[0];
        rect->out_rect[bbox_num - 1 - i].y1 = rect_tmp[1];
        rect->out_rect[bbox_num - 1 - i].x2 = rect_tmp[2];
        rect->out_rect[bbox_num - 1 - i].y2 = rect_tmp[3];
        rect->out_rect[bbox_num - 1 - i].conf = rect_tmp[4];
        rect->out_rect[bbox_num - 1 - i].class_id = rect_tmp[5];
    }
}

ot_s32 sample_yolov5_execute(sample_avp_model_task_info_t* model_info, const char* input_path, void* out)
{
    int32_t ret = -1;

    ret = load_data_from_file(input_path, (uint8_t *)model_info->input_tensor[0].virt_addr, model_info->input_tensor[0].shape,
        model_info->input_tensor[0].stride, model_info->input_tensor[0].dtype);

    // 模型推理
    ret = sample_avp_execute_model(model_info, 1);  // repeat_time = 1
    SAMPLE_AVP_CHECK_GOTO_WITH_LOG(ret != 0, EXECUTE_MODEL_FAIL, "execute model fail.\n");

    if (out != OT_NULL) {
        sample_avp_npu_rect_t *rect = (sample_avp_npu_rect_t *)out;
        rect->num = *(float *)model_info->output_tensor[0].virt_addr;
        int output_boxnum = model_info->output_tensor[1].shape.dims[model_info->output_tensor[1].shape.dim_size - 1U];
        sample_yolov5_parse_bbox((char *)model_info->output_tensor[1].virt_addr, output_boxnum, rect->num, rect);
    }

EXECUTE_MODEL_FAIL:

    return ret;
}

int sample_yolov5_draw_box(const char *image_path, sample_avp_npu_rect_t *npu_rect)
{
    const cv::Scalar& color = cv::Scalar(0, 0, 255);
    int thickness = 2;
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Could not load image from path: " << image_path << std::endl;
        return -1;
    }

    // 矩形的两个对角点
    for (int i = 0; i < npu_rect->num; i++) {
        if (npu_rect->out_rect[i].conf >= 0.35) {
            cv::Point top_left(npu_rect->out_rect[i].x1, npu_rect->out_rect[i].y1);                                  // 左上角点
            cv::Point bottom_right(npu_rect->out_rect[i].x2, npu_rect->out_rect[i].y2);             // 右下角点
            cv::rectangle(image, top_left, bottom_right, color, thickness);
        }
    }
    cv::imwrite("./output_image.jpg", image);

    return 0;
}

int sample_yolov5_run(const char *input_path, const char *model_path)
{
    int ret = -1;
    sample_avp_model_task_info_t yolov5_info;
    sample_avp_npu_rect_t npu_rect;
    const char *image_path = nullptr;
    memset(&yolov5_info, 0, sizeof(sample_avp_model_task_info_t));
    memset(&npu_rect, 0, sizeof(sample_avp_npu_rect_t));

    // load model
    ret = ot_avp_npu_load_model(model_path, &yolov5_info.handle);
    SAMPLE_AVP_CHECK_RETURN_WITH_LOG(ret != 0, ret, "load model fail.\n");

    sample_avp_create_datasets(yolov5_info.handle, &yolov5_info);

    ret = sample_yolov5_execute(&yolov5_info, input_path, &npu_rect);
    SAMPLE_AVP_CHECK_GOTO_WITH_LOG(ret != 0, EXECUTE_MODEL_FAIL, "yolov5_execute fail.\n");
    printf("infer result:\n");
    for (int i = 0 ; i < npu_rect.num; i++) {
        printf("class_id is: %d, conf is: %f, rect[x1,y1,x2,y2] is: [%f, %f, %f, %f]\n",
            npu_rect.out_rect[i].class_id, npu_rect.out_rect[i].conf, npu_rect.out_rect[i].x1, 
            npu_rect.out_rect[i].y1, npu_rect.out_rect[i].x2, npu_rect.out_rect[i].y2);
    }
    image_path = "../../../../data/ImageNet/pic/yolo_src_640x640.bmp";
    ret = sample_yolov5_draw_box(image_path, &npu_rect);
    
    SAMPLE_AVP_CHECK_GOTO_WITH_LOG(ret != 0, DRAW_BOX_FAIL, "yolov5_draw_box fail.\n");

DRAW_BOX_FAIL:
EXECUTE_MODEL_FAIL:
    sample_avp_destroy_datasets(yolov5_info.handle, &yolov5_info);
    ot_avp_npu_unload_model(yolov5_info.handle);
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
    const char* model_path = "..\\yolov5s.ortm";
    const char* input_path = "..\\000000145690.yuv420sp";
#else
    const char* model_path = "./tmp_models/yolov5s.ortm";
    const char* input_path = "./tmp_res/infer_data/000000145690.yuv420sp";
#endif
#else
    ret = sample_avp_npu_media_system_init();
    const char* model_path = "../../../../data/prebuild_ortm/yolov5s.ortm";
    const char* input_path = "../../../../data/ImageNet/yuv420sp/yolo_src_640x640.yuv420sp";
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

    ret = sample_yolov5_run(input_path, model_path);
    SAMPLE_AVP_CHECK_GOTO_WITH_LOG(ret != 0, NPU_RUN_FAIL, "sample yolov5 run fail.\n");

NPU_RUN_FAIL:
    ot_avp_npu_deinit();

NPU_INIT_FAIL:
#ifndef SIMULATOR
    sample_avp_npu_media_system_deinit();
#endif
    return 0;
}
