 #include <stdio.h>
 #include <dirent.h>
 #include <string.h>
#include <string>
 #include <vector>
 #include <unistd.h>
 #include "sample_avp_npu_common.h"
 #include <iostream>
 #include <filesystem>
 #include "../common/evaluate/tools.h"

typedef struct sample_LSS_cfg_param
{
    char *model_path ;
    char *input_path ;
    char *output_path;
    uint32_t repeat_times;
    int32_t frame_cnt;
    int32_t enable_pub;
} sample_LSS_cfg_param_t;

static sample_LSS_cfg_param_t cfg_param;

std::vector<char *> npu_out;


static int32_t sample_LSS_dump_model_out(sample_avp_model_task_info_t *param,
                                          sample_LSS_cfg_param_t *cfg_param,
                                          std::string output_perfix,
                                          int32_t index)
{
    std::string output_root_path = cfg_param->output_path;
    char output[1024] = {0};

    std::string output_dir = output_root_path + "/" + std::to_string(index);

    if (!std::filesystem::exists(output_dir)) {
        std::filesystem::create_directories(output_dir);
    }

    std::string output_path = output_dir + "/" + output_perfix + "_0.float.bin";
    strcpy(output, output_path.c_str());

    dump_data_to_binary_file(output, (uint8_t *)param->output_tensor[0].virt_addr,
                        param->output_tensor[0].shape, param->output_tensor[0].stride,
                        param->output_tensor[0].dtype);

    return 0;
}

// 检查字符串是否以指定后缀
bool has_suffix(const char *str, const char *suffix) {
    if (!str || !suffix) return false;
    
    size_t len_str = strlen(str);
    size_t len_suffix = strlen(suffix);
    
    // 后缀长度不能超过字符串长度
    if (len_suffix > len_str) return false;
    
    // 比较字符串末尾部分
    return strcmp(str + len_str - len_suffix, suffix) == 0;
}

// 获取当前目录下匹配指定前缀的文件名
std::string get_files_with_prefix(const char *folder_path, const char *prefix) {
    std::string full_file;
    struct dirent *entry;

    // 打开目录
    DIR *dir = opendir(folder_path);
    if (dir == NULL) {
        perror("opendir");
        return full_file;
    }

    // 遍历目录中的每个条目
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG && has_suffix(entry->d_name, prefix)) {
            // 构建完整文件路径
            full_file = folder_path;
            full_file += "/";
            full_file += entry->d_name;
            break; // 如果只需要第一个匹配的文件，保留此行
            // 如果需要所有匹配的文件，可以移除break并将结果存储到容器中
        }
    }

    // 关闭目录
    closedir(dir);
    return full_file;
}

static int32_t sample_LSS_execute(sample_LSS_cfg_param_t *cfg_param)
{
    sample_avp_model_task_info_t LSS_model;
    memset(&LSS_model, 0, sizeof(sample_avp_model_task_info_t));

    if (cfg_param->model_path == NULL || cfg_param->input_path == NULL)
    {
        printf("Error: model_path[%s] or input_path[%s] is null.\n", cfg_param->model_path, cfg_param->input_path);
        return -1;
    }

    int32_t ret = ot_avp_npu_load_model(cfg_param->model_path, &LSS_model.handle);

    SAMPLE_AVP_CHECK_RETURN_WITH_LOG(ret != 0, ret, "load model fail.");
    printf("load model, model_path: [%s]\n", cfg_param->model_path);

    sample_avp_create_datasets(LSS_model.handle, &LSS_model);

    for (int32_t index = 1; index <= cfg_param->frame_cnt; index++)
    {
        std::string root_folder_path = cfg_param->input_path;
        std::string folder_path = root_folder_path + std::to_string(index);
        std::string input_file = get_files_with_prefix(folder_path.c_str(), "_6x3x256x704.bin");
        std::cout << "enable_pub: " << cfg_param->enable_pub << std::endl;
        if (cfg_param->enable_pub == 1) {
            input_file = get_files_with_prefix(folder_path.c_str(), "_6x3x128x352.bin");
        }
        std::cout << "input file: " << input_file << std::endl;

        // 读取输入
        ret = load_data_from_file(
            input_file.c_str(),
            (uint8_t *)LSS_model.input_tensor[0].virt_addr,
            LSS_model.input_tensor[0].shape,
            LSS_model.input_tensor[0].stride,
            LSS_model.input_tensor[0].dtype
        );

        // 模型推理
        ret = sample_avp_execute_model(&LSS_model, cfg_param->repeat_times);
        SAMPLE_AVP_CHECK_GOTO_WITH_LOG(ret != 0, EXECUTE_MODEL_FAIL, "execute model fail.");

        char* safe_output = save_data_buf((const char *)LSS_model.output_tensor[0].virt_addr, LSS_model.output_tensor[0].shape, (uint32_t)LSS_model.output_tensor[0].stride,LSS_model.output_tensor[0].dtype);

        npu_out.emplace_back(safe_output);
        std::string output_perfix = input_file.substr(input_file.find_last_of("/") + 1);
        output_perfix = output_perfix.substr(0, output_perfix.find_last_of("."));
        sample_LSS_dump_model_out(&LSS_model, cfg_param, output_perfix, index);
    }


EXECUTE_MODEL_FAIL:
    sample_avp_destroy_datasets(LSS_model.handle, &LSS_model);
    ot_avp_npu_unload_model(LSS_model.handle);

    return ret;
}


static int32_t sample_LSS_run()
{
    sample_avp_npu_media_system_init();

    ot_avp_npu_config config = {
        .core_ids = 0U,
        .dump_layer = OT_FALSE, // dump layer
        .perf_mode = OT_FALSE,  // perf_mode
        .log_level = 4U,         // log level info
        .thread_num = 0U,
        .extra_config_num = 0U,
        .extra_configs = NULL,
    };
    int32_t ret = ot_avp_npu_init(&config);
    SAMPLE_AVP_CHECK_GOTO_WITH_LOG(ret != 0, NPU_INIT_FAIL, "npu init fail.");

    ret = sample_LSS_execute(&cfg_param);

    ot_avp_npu_deinit();

NPU_INIT_FAIL:

    sample_avp_npu_media_system_deinit();

    return ret;
}

void sample_LSS_help()
{
    printf("./sample_LSS\n"
           "    -m : model_path.\n"
           "    -i : 输入数据集所在的文件夹\n"
           "    -n : 处理的帧数, 默认1次. \n");
}

int32_t get_cfg_param(int32_t argc, char **argv)
{
    int32_t opt = 0;
    cfg_param.frame_cnt = 80;
    cfg_param.repeat_times = 1U;
    const char *optstring = "m:hi:n:o:p:";
    while ((opt = getopt(argc, argv, optstring)) != -1)
    {
        switch (opt)
        {
        case 'h':
            sample_LSS_help();
            exit(0);
            break;
        case 'm':
            cfg_param.model_path = optarg;
            break;
        case 'i':
            cfg_param.input_path = optarg;
            break;
        case 'n':
            cfg_param.frame_cnt = (int32_t)atoi(optarg);
            break;
        case 'p':
            cfg_param.enable_pub = (int32_t)atoi(optarg);
            break;
        case 'o':
            cfg_param.output_path = optarg;
            break;
        default:
            break;
        }
    }
    printf("config:\n"
           "    model_path: %s\n"
           "    input_path: %s\n"
           "    frame_cnts: %u\n"
           "   output_path: %s\n",
           cfg_param.model_path, cfg_param.input_path, cfg_param.frame_cnt, cfg_param.output_path);

    return 0;
}

 #define GREEN "\x1B[32m"
 #define RESET "\x1B[0m"
 int main(int argc, char **argv) {

     // 获取配置参数
    int ret = get_cfg_param(argc, argv);
    SAMPLE_AVP_CHECK_RETURN_WITH_LOG(ret != 0, ret, "get_cfg_param fail.");

    printf(GREEN "LSS execute start. \n" RESET);

    ret = sample_LSS_run();
    SAMPLE_AVP_CHECK_RETURN_WITH_LOG(ret != 0, ret, "execute LSS fail.");

    printf(GREEN "LSS execute end. \n" RESET);

    return ret;

 }

