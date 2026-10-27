
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <dirent.h>   // 用于目录遍历
#include <sys/stat.h> // 用于 stat/mkdir
#include <unistd.h>   // 用于 access
#include <vector>
#include <algorithm>
#include "sample_avp_npu_common.h"

using namespace cv;

std::vector<int> get_numeric_folders(const std::string& path) {
    std::vector<int> indices;
    DIR* dir = opendir(path.c_str());
    if (!dir) {
        std::cerr << "[Error] Failed to open directory: " << path << std::endl;
        return indices;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        // 忽略 . 和 ..
        if (entry->d_type == DT_DIR) {
            std::string name = entry->d_name;
            if (name == "." || name == "..") continue;

            try {
                int idx = std::stoi(name);
                indices.push_back(idx);
            } catch (...) {
                // 忽略非数字命名的子文件夹
            }
        }
    }
    closedir(dir);
    std::sort(indices.begin(), indices.end());
    return indices;
}


bool ensure_dir_exists(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        // Directory doesn't exist, try to create
        if (mkdir(path.c_str(), 0755) != 0) {
            std::cerr << "[Error] Failed to create directory: " << path << ", errno: " << errno << std::endl;
            return false;
        }
    } else if (!S_ISDIR(st.st_mode)) {
        std::cerr << "[Error] Path exists but is not a directory: " << path << std::endl;
        return false;
    }
    return true;
}

// 绘制彩色视差图
cv::Mat drawDisparity(const cv::Mat& disparity) {
    cv::Mat norm_disp, color_disp;
    normalize(disparity, norm_disp, 0, 255, cv::NORM_MINMAX, CV_8U);
    applyColorMap(norm_disp, color_disp, cv::COLORMAP_JET);
    return color_disp;
}

// 读取 float32 bin 文件为 cv::Mat
cv::Mat readBinDisparity(float* model_output, int height, int width) {
    cv::Mat disparity(height, width, CV_32F, model_output);
    return disparity.clone(); // 防止引用 buffer
}

cv::Mat dePreprocessImg(std::string& bin_path, int H, int W){
    // ImageNet mean and std
    const std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    const std::vector<float> stddev = {0.229f, 0.224f, 0.225f};

    const int C = 3;
    const int total = C * H * W;

    // 读取 bin 文件为 float32 向量
    std::ifstream file(bin_path, std::ios::binary);
    if (!file) {
        std::cerr << "无法打开文件: " << bin_path << std::endl;
    }

    std::vector<float> buffer(total);
    file.read(reinterpret_cast<char*>(buffer.data()), total * sizeof(float));
    file.close();

    // 创建三个通道
    std::vector<cv::Mat> channels;
    for (int c = 0; c < C; ++c) {
        cv::Mat channel(H, W, CV_32F);
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                int idx = c * H * W + i * W + j;
                float val = buffer[idx] * stddev[c] + mean[c];  // 反归一化
                channel.at<float>(i, j) = MIN(MAX(val, 0.0f), 1.0f);  // clip to [0,1]
            }
        }
        channels.push_back(channel);
    }

    // 合并通道并转换为 uchar 图像（RGB）
    cv::Mat merged;
    cv::merge(channels, merged);  // float32 [0,1]
    merged *= 255.0f;
    merged.convertTo(merged, CV_8UC3); 

    // RGB → BGR
    cv::Mat bgr_img;
    cv::cvtColor(merged, bgr_img, cv::COLOR_RGB2BGR);
    return bgr_img;
}

int getNumber(std::string& board_out_path){

    if (board_out_path.back() == '/') {
        board_out_path.pop_back();
    }

    std::stringstream ss(board_out_path);
    std::string segment;
    std::vector<std::string> segments;

    while (std::getline(ss, segment, '/')) {
        segments.push_back(segment);
    }

    std::string last_part = segments.back();
    int value = std::stoi(last_part);
    return value;
}

typedef struct sample_stereo_cfg_param
{
    const char *model_path;
    const char *input_path;
    const char *output_path;
    bool draw_stereo = true;
    const char *save_img_path = "./";
    uint32_t repeat_times;
    int32_t frame_cnt;
    uint32_t number = 1U;
} sample_stereo_cfg_param_t;

static sample_stereo_cfg_param_t g_cfg_param;
#if SAMPLE_STEREO_DEBUG
static int32_t sample_lightstereo_dump_model_out(sample_avp_model_task_info_t *param, char *output_prefix)
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

int32_t lightstereoProcess(std::vector<char *> &npu_out)
{
    int h = 544, w = 960;
    float *model_output= reinterpret_cast<float*>(npu_out[0]);
    cv::Mat disp = readBinDisparity(model_output, h, w);
    cv::Mat color_disp = drawDisparity(disp);
    std::string save_img_path = std::string(g_cfg_param.save_img_path) + "/disp_board_" + std::to_string(g_cfg_param.number) + ".jpg";
    imwrite(save_img_path, color_disp);
    if (g_cfg_param.draw_stereo){
        if (!ensure_dir_exists(std::string(g_cfg_param.input_path))) {
            std::cerr << "stereo path does not exist: " << g_cfg_param.input_path << std::endl;
            return -1;
        }
        std::string left_bin_path = std::string(g_cfg_param.input_path) + "/" + std::to_string(g_cfg_param.number) + "/0.bin";
        std::string right_bin_path = std::string(g_cfg_param.input_path) + "/" + std::to_string(g_cfg_param.number) +  "/1.bin";
        cv::Mat left_img = dePreprocessImg(left_bin_path, h, w);
        cv::Mat right_img = dePreprocessImg(right_bin_path, h, w);
        // 左右拼接并保存
        std::vector<cv::Mat> imgs = {left_img, right_img, color_disp};
        cv::Mat concatenated;
        cv::hconcat(imgs, concatenated);
        std::string stereo_concat_path = std::string(g_cfg_param.save_img_path) + "/disp_stereo_board_" + std::to_string(g_cfg_param.number) + ".jpg";
        imwrite(stereo_concat_path, concatenated);    
    }
    std::cout << "visualized results are saved at " << save_img_path << std::endl;
    return 0;
}


static int32_t sample_lightstereo_execute(sample_stereo_cfg_param_t *g_cfg_param)
{
    sample_avp_model_task_info_t lightstereo_model;
    // memset(&lightstereo_model, 0, sizeof(sample_avp_model_task_info_t));

    if (g_cfg_param->model_path == NULL || g_cfg_param->input_path == NULL)
    {
        printf("Error: model_path[%s] or input_path[%s] is null.\n", g_cfg_param->model_path, g_cfg_param->input_path);
        return -1;
    }

    int32_t ret = ot_avp_npu_load_model(g_cfg_param->model_path, &lightstereo_model.handle);
    SAMPLE_AVP_CHECK_RETURN_WITH_LOG(ret != 0, ret, "load model fail.");
    printf("load model, model_path: [%s]\n", g_cfg_param->model_path);
    sample_avp_create_datasets(lightstereo_model.handle, &lightstereo_model);
    // g_cfg_param->number = 1;
    std::cout <<"number = "<<g_cfg_param->number<<"\n";
    std::string left_path = std::string(g_cfg_param->input_path) + "/" + std::to_string(g_cfg_param->number) + "/0.bin";
    std::string right_path = std::string(g_cfg_param->input_path) + "/" + std::to_string(g_cfg_param->number) +  "/1.bin";
        // 读取输入
    {
    ret = load_data_from_file(left_path.c_str(), (uint8_t *)lightstereo_model.input_tensor[0].virt_addr,
                                lightstereo_model.input_tensor[0].shape, lightstereo_model.input_tensor[0].stride, lightstereo_model.input_tensor[0].dtype);
    ret = load_data_from_file(right_path.c_str(), (uint8_t *)lightstereo_model.input_tensor[1].virt_addr,
                                lightstereo_model.input_tensor[1].shape, lightstereo_model.input_tensor[1].stride, lightstereo_model.input_tensor[1].dtype);
        // 模型推理
    ret = sample_avp_execute_model(&lightstereo_model, g_cfg_param->repeat_times);
    SAMPLE_AVP_CHECK_GOTO_WITH_LOG(ret != 0, EXECUTE_MODEL_FAIL, "execute model fail.");

        // 后处理
    std::vector<char *> npu_out = {
            (char *)lightstereo_model.output_tensor[0].virt_addr};
    ret = lightstereoProcess(npu_out);

#if SAMPLE_STEREO_DEBUG
        sample_lightstereo_dump_model_out(&lightstereo_model, (char *)"npu_out_");
#endif
    }
EXECUTE_MODEL_FAIL:
    sample_avp_destroy_datasets(lightstereo_model.handle, &lightstereo_model);
    ot_avp_npu_unload_model(lightstereo_model.handle);
    return ret;
}


static int32_t sample_lightstereo_run()
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

    ret = sample_lightstereo_execute(&g_cfg_param);

    ot_avp_npu_deinit();
NPU_INIT_FAIL:

    sample_avp_npu_media_system_deinit();

    return ret;
}

void sample_bev_help()
{
    printf("./sample_lightstereo\n"
           "    -m : model_path.\n"
           "    -i : 输入数据集所在的文件夹\n"
           "    -c : count, 处理的帧数, 默认1次. \n"
           "    -b : Set to 1 to draw and concatenate src stereo images; 0 to disable\n"
           "    -d : Path to src stereo images (e.g., .bin), or 'none' if not used\n"
           "    -s : Path to save stereo images of model output\n"
           "    -n : Set the input number\n");
}

int32_t get_cfg_param(int32_t argc, char **argv)
{
    int32_t opt = 0;
    g_cfg_param.frame_cnt = 1;
    g_cfg_param.repeat_times = 1U;
    const char *optstring = "m:h:i:c:d:b:n:s:";
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
        case 'c':
            g_cfg_param.frame_cnt = (uint32_t)atoi(optarg);
            break;
        case 'd':
            g_cfg_param.output_path = optarg;
            break;
        case 's':
            g_cfg_param.save_img_path = optarg;
            break;
        case 'b':
            g_cfg_param.draw_stereo = optarg;
            break;
        case 'n':
            g_cfg_param.number = (uint32_t)atoi(optarg);
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

    printf(GREEN "lightstereo execute start. \n" RESET);

    ret = sample_lightstereo_run();
    SAMPLE_AVP_CHECK_RETURN_WITH_LOG(ret != 0, ret, "execute lightstereo fail.");

    printf(GREEN "lightstereo execute end. \n" RESET);

    return ret;
}