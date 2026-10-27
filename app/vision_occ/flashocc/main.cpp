#include "sample_avp_npu_common.h"
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <vector>


struct sample_flashocc_cfg_param {
    char *model_path;
    char *input_path;
    char *output_path;
    int repeat_times;
    int frame_cnt;
};

static sample_flashocc_cfg_param g_cfg_param;

#if SAMPLE_BEV_DEBUG
static int sample_flashocc_dump_model_out(sample_avp_model_task_info_t *param, char *output_prefix) {
    for (int i = 0; i < param->output_num; i++) {
        char output[1024] = {0};
        sprintf(output, "%s_%d.bin", output_prefix, i);
        dump_data_to_binary_file(output, (uint8_t *)param->output_tensor[i].virt_addr, param->output_tensor[i].shape,
                          param->output_tensor[i].stride, param->output_tensor[i].dtype);
    }
    return 0;
}
#endif

const std::vector<std::string> GetTokens(const std::string &filename) {
    std::vector<std::string> tokens;
    tokens.clear();
    std::ifstream file(filename);

    if (file) {
        std::string token;
        while (std::getline(file, token)) {
            tokens.push_back(token);
        }
    }
    return tokens;
}

void MakeDir(const std::string folder_path) {
    std::string command;
    command = "mkdir -p " + folder_path;
    int ret = system(command.c_str());
    if (ret != 0) {
        std::cout << " Fail to build " << folder_path << " ." << std::endl;
    }
}

// 检查文件名是否以指定后缀结尾
static int has_suffix(const char *filename, const char *suffix) {
    const size_t len_filename = strlen(filename);
    const size_t len_suffix = strlen(suffix);
    if (len_filename < len_suffix) {
        return 0;
    }
    return strcmp(filename + len_filename - len_suffix, suffix) == 0;
}

// 获取当前目录下匹配指定后缀的文件名
std::string get_files_with_suffix(const char *folder_path, const char *suffix) {
    std::string full_file;
    struct dirent *entry;

    // 打开当前目录
    DIR *dir = opendir(folder_path);
    if (dir == NULL) {
        perror("opendir");
        return full_file;
    }

    // 遍历目录中的每个条目
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG && has_suffix(entry->d_name, suffix)) {
            full_file = folder_path;
            full_file += "/";
            full_file += entry->d_name;
            // printf("%s\n", full_file.c_str());
        }
    }

    // 关闭目录
    closedir(dir);
    return full_file;
}

std::vector<float> nchw_to_nwhc_fast(const std::vector<float> &input, size_t N, size_t C, size_t H, size_t W) {
    std::vector<float> output(N * W * H * C);
    const float *in_ptr = input.data();
    float *out_ptr = output.data();

    for (size_t n = 0UL; n < N; ++n) {
        for (size_t w = 0UL; w < W; ++w) {
            for (size_t h = 0UL; h < H; ++h) {
                for (size_t c = 0UL; c < C; ++c) {
                    *(out_ptr++) = in_ptr[n * C * H * W + c * H * W + h * W + w];
                }
            }
        }
    }
    return output;
}


static int sample_flashocc_execute(sample_flashocc_cfg_param *cfg_param) {
    sample_avp_model_task_info_t flashocc_model;
    memset(&flashocc_model, 0, sizeof(sample_avp_model_task_info_t));

    if (cfg_param->model_path == NULL || cfg_param->input_path == NULL) {
        printf("Error: model_path[%s] or input_path[%s] is null.\n", cfg_param->model_path, cfg_param->input_path);
        return -1;
    }

    int ret = ot_avp_npu_load_model(cfg_param->model_path, &flashocc_model.handle);
    SAMPLE_AVP_CHECK_RETURN_WITH_LOG(ret != 0, ret, "load model fail.");
    printf("load model, model_path: [%s]\n", cfg_param->model_path);

    sample_avp_create_datasets(flashocc_model.handle, &flashocc_model);

    for (int index = 1; index <= cfg_param->frame_cnt; index++) {
        std::string root_folder_path = cfg_param->input_path;
        std::string folder_path = root_folder_path + "/" + std::to_string(index) + "/";
        // 读取输入
        ret = load_data_from_file(get_files_with_suffix(folder_path.c_str(), "_6x3x256x704.bin").c_str(),
                                  (uint8_t *)flashocc_model.input_tensor[0].virt_addr,
                                  flashocc_model.input_tensor[0].shape, flashocc_model.input_tensor[0].stride,
                                  flashocc_model.input_tensor[0].dtype);

        // 模型推理
        ret = sample_avp_execute_model(&flashocc_model, cfg_param->repeat_times);
        SAMPLE_AVP_CHECK_GOTO_WITH_LOG(ret != 0, EXECUTE_MODEL_FAIL, "execute model fail.");

        // 后处理
        const auto npu_out = (char *)flashocc_model.output_tensor[0].virt_addr;
        const auto out_num = flashocc_model.output_tensor[0].size / sizeof(float);
        std::cout << "npu_out size: " << flashocc_model.output_tensor[0].size
                  << ", out_num: " << out_num << std::endl;
        std::cout << "start to process model output shape: " << std::endl;
        std::vector<float> model_out((float *)npu_out, (float *)npu_out + out_num);
        if (model_out.size() == 288 * 200 * 200) {
            model_out = nchw_to_nwhc_fast(model_out, 1UL, 288UL, 200UL, 200UL);
        } else if (model_out.size() == 288 * 128 * 128) {
            model_out = nchw_to_nwhc_fast(model_out, 1UL, 288UL, 128UL, 128UL);
        }else{
            std::cout << "Error: model output size is not supported, size: " << model_out.size() << std::endl;
            return -1;
        }
        std::cout<< "output shape: "<< flashocc_model.output_tensor[0].shape.dims[0] << "x"
                  << flashocc_model.output_tensor[0].shape.dims[3]<< "x"
                  << flashocc_model.output_tensor[0].shape.dims[2] << "x"
                  << flashocc_model.output_tensor[0].shape.dims[1] << std::endl;
        

#if SAMPLE_BEV_DEBUG
        sample_flashocc_dump_model_out(&flashocc_model, (char *)"npu_out_");
#endif
    }

EXECUTE_MODEL_FAIL:
    sample_avp_destroy_datasets(flashocc_model.handle, &flashocc_model);
    ot_avp_npu_unload_model(flashocc_model.handle);

    return ret;
}

static int sample_flashocc_run() {
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
    int ret = ot_avp_npu_init(&config);
    SAMPLE_AVP_CHECK_GOTO_WITH_LOG(ret != 0, NPU_INIT_FAIL, "npu init fail.");

    ret = sample_flashocc_execute(&g_cfg_param);

    ot_avp_npu_deinit();
NPU_INIT_FAIL:

    sample_avp_npu_media_system_deinit();

    return ret;
}

void sample_flashocc_help() {
    printf("./sample_flashocc\n"
           "    -m : model_path.\n"
           "    -i : 输入数据集所在的文件夹\n"
           "    -n : 处理的帧数, 默认1次. \n");
}

int get_cfg_param(int argc, char **argv) {
    int opt = 0;
    g_cfg_param.frame_cnt = 1;
    g_cfg_param.repeat_times = 1;
    const char *optstring = "m:hi:n:";
    while ((opt = getopt(argc, argv, optstring)) != -1) {
        switch (opt) {
            case 'h':
                sample_flashocc_help();
                exit(0);
                break;
            case 'm':
                g_cfg_param.model_path = optarg;
                break;
            case 'i':
                g_cfg_param.input_path = optarg;
                break;
            case 'n':
                g_cfg_param.frame_cnt = (int)atoi(optarg);
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
int main(int argc, char **argv) {
    // 获取配置参数
    int ret = get_cfg_param(argc, argv);
    SAMPLE_AVP_CHECK_RETURN_WITH_LOG(ret != 0, ret, "get_cfg_param fail.");

    printf(GREEN "flashocc execute start. \n" RESET);

    ret = sample_flashocc_run();
    SAMPLE_AVP_CHECK_RETURN_WITH_LOG(ret != 0, ret, "execute flashocc fail.");

    printf(GREEN "flashocc execute end. \n" RESET);

    return ret;
}
