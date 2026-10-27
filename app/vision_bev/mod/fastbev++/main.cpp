#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <vector>
#include <unistd.h>
#include "sample_avp_npu_common.h"
#include "../common/data_init/bev_initializer.h"
#include "../common/center_head/sample_center_point_bbox_coder.h"
#include "../common/center_head/bev_decoder.h"
#include "../common/view/bev_visualizer.h"


typedef struct sample_bev_cfg_param
{
    char *model_path;
    char *input_path;
    char *output_path;
    uint32_t repeat_times;
    int32_t frame_cnt;
} sample_bev_cfg_param_t;

static sample_bev_cfg_param_t g_cfg_param;
#if SAMPLE_BEV_DEBUG
static int32_t sample_fastbev_dump_model_out(sample_avp_model_task_info_t *param, char *output_prefix)
{
    for (uint32_t i = 0; i < param->output_num; i++)
    {
        char output[1024] = {0};
        sprintf(output, "%s_%d.bin", output_prefix, i);
        dump_data_to_binary_file(output, (uint8_t *)param->output_tensor[i].virt_addr,
                          param->output_tensor[i].shape, param->output_tensor[i].stride, param->output_tensor[i].dtype);
    }
    return 0;
}
#endif

void BevPostProcess(const std::string &folder_path, int index, std::vector<char *> &npu_out)
{
    BevData bev_data;

    BevInitializer initializer;
    initializer.Init(bev_data, folder_path, index, false);

    for (int i = 0; i < (int)npu_out.size(); i++)
    {
        bev_data.m_prepared_data.buffer[i] = npu_out[i];
    }
    bev_data.m_npu_out_no_free = true;

    BevDecoder decoder;
    decoder.DecodeBBox(bev_data);

    BevVisualizer visualizer;
    visualizer.DrawCanvaAndImgs(bev_data);
}


static int32_t sample_fastbev_execute(sample_bev_cfg_param_t *cfg_param)
{
    sample_avp_model_task_info_t fastbev_model;
    memset(&fastbev_model, 0, sizeof(sample_avp_model_task_info_t));

    if (cfg_param->model_path == NULL || cfg_param->input_path == NULL)
    {
        printf("Error: model_path[%s] or input_path[%s] is null.\n", cfg_param->model_path, cfg_param->input_path);
        return -1;
    }

    int32_t ret = ot_avp_npu_load_model(cfg_param->model_path, &fastbev_model.handle);
    SAMPLE_AVP_CHECK_RETURN_WITH_LOG(ret != 0, ret, "load model fail.");
    printf("load model, model_path: [%s]\n", cfg_param->model_path);

    sample_avp_create_datasets(fastbev_model.handle, &fastbev_model);

    for (int index = 1; index <= cfg_param->frame_cnt; index++)
    {
        std::string root_folder_path = cfg_param->input_path;
        std::string folder_path = root_folder_path + "/" + std::to_string(index) + "/";
        // 读取输入
        ret = load_data_from_file(get_files_with_suffix(folder_path.c_str(), "_6x3x256x704.bin").c_str(), (uint8_t *)fastbev_model.input_tensor[0].virt_addr,
                                  fastbev_model.input_tensor[0].shape, fastbev_model.input_tensor[0].stride, fastbev_model.input_tensor[0].dtype);

        // 模型推理
        ret = sample_avp_execute_model(&fastbev_model, cfg_param->repeat_times);
        SAMPLE_AVP_CHECK_GOTO_WITH_LOG(ret != 0, EXECUTE_MODEL_FAIL, "execute model fail.");

        // 后处理
        std::vector<char *> npu_out = {
            (char *)fastbev_model.output_tensor[0].virt_addr,
            (char *)fastbev_model.output_tensor[1].virt_addr,
            (char *)fastbev_model.output_tensor[2].virt_addr,
            (char *)fastbev_model.output_tensor[3].virt_addr,
            (char *)fastbev_model.output_tensor[4].virt_addr,
            (char *)fastbev_model.output_tensor[5].virt_addr};
        BevPostProcess(folder_path, index, npu_out);

#if SAMPLE_BEV_DEBUG
        sample_fastbev_dump_model_out(&fastbev_model, (char *)"npu_out_");
#endif
    }

EXECUTE_MODEL_FAIL:
    sample_avp_destroy_datasets(fastbev_model.handle, &fastbev_model);
    ot_avp_npu_unload_model(fastbev_model.handle);

    return ret;
}


static int32_t sample_fastbev_run()
{
    sample_avp_npu_media_system_init();

    ot_avp_npu_config config = {
        .core_ids = 0U,
        .dump_layer = OT_FALSE, // dump layer
        .perf_mode = OT_FALSE,  // perf_mode
        .log_level = 2U,         // log level info
        .thread_num = 0U,
        .extra_config_num = 0U,
        .extra_configs = NULL,
    };
    int32_t ret = ot_avp_npu_init(&config);
    SAMPLE_AVP_CHECK_GOTO_WITH_LOG(ret != 0, NPU_INIT_FAIL, "npu init fail.");

    ret = sample_fastbev_execute(&g_cfg_param);

    ot_avp_npu_deinit();
NPU_INIT_FAIL:

    sample_avp_npu_media_system_deinit();

    return ret;
}

void sample_bev_help()
{
    printf("./sample_fastbev\n"
           "    -m : model_path.\n"
           "    -i : 输入数据集所在的文件夹\n"
           "    -n : 处理的帧数, 默认1次. \n");
}

int32_t get_cfg_param(int32_t argc, char **argv)
{
    int32_t opt = 0;
    g_cfg_param.frame_cnt = 1;
    g_cfg_param.repeat_times = 1U;
    const char *optstring = "m:hi:n:";
    while ((opt = getopt(argc, argv, optstring)) != -1)
    {
        switch (opt)
        {
        case 'h':
            sample_bev_help();
            exit(0);
            break;
        case 'm':
            g_cfg_param.model_path = optarg;
            break;
        case 'i':
            g_cfg_param.input_path = optarg;
            break;
        case 'n':
            g_cfg_param.frame_cnt = (uint32_t)atoi(optarg);
            break;
        default:
            break;
        }
    }
    printf("config:\n"
           "    model_path: %s\n"
           "    input_path: %s\n"
           "    frame_cnts: %u\n",
           g_cfg_param.model_path, g_cfg_param.input_path, g_cfg_param.frame_cnt);

    return 0;
}

#define GREEN "\x1B[32m"
#define RESET "\x1B[0m"
int main(int argc, char **argv)
{
    // 获取配置参数
    int ret = get_cfg_param(argc, argv);
    SAMPLE_AVP_CHECK_RETURN_WITH_LOG(ret != 0, ret, "get_cfg_param fail.");

    printf(GREEN "fastbev execute start. \n" RESET);

    ret = sample_fastbev_run();
    SAMPLE_AVP_CHECK_RETURN_WITH_LOG(ret != 0, ret, "execute fastbev fail.");

    printf(GREEN "fastbev execute end. \n" RESET);

    return ret;
}
