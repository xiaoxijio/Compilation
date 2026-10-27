#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "json.hpp"
#include <dirent.h>
#include "../view/bev_visualizer.h"
#include "../center_head/bev_decoder.h"
#include "bev_initializer.h"


// 从二进制文件读取数据到缓冲区
static char *read_binary_file(const char *filename)
{
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        perror("无法打开文件");
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char *buffer = (char *)malloc(file_size);
    if (buffer == NULL) {
        perror("内存分配失败");
        fclose(file);
        return NULL;
    }
    size_t bytes_read = fread(buffer, 1, file_size, file);
    if (bytes_read != file_size) {
        perror("读取文件失败");
    }
    fclose(file);

    return buffer;
}

// 读取 JSON 文件内容到 std::string 的函数
std::string readJsonFile(const std::string& filePath)
{
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filePath << std::endl;
        return "";
    }

    std::ostringstream oss;
    oss << file.rdbuf();
    file.close();

    return oss.str();
}

// 检查文件名是否以指定后缀结尾
static int has_suffix(const char *filename, const char *suffix)
{
    size_t len_filename = strlen(filename);
    size_t len_suffix = strlen(suffix);
    if (len_filename < len_suffix) {
        return 0;
    }
    return strcmp(filename + len_filename - len_suffix, suffix) == 0;
}

// 获取当前目录下匹配指定后缀的文件名
std::string get_files_with_suffix(const char *folder_path, const char *suffix)
{
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

void BevInitializer::Init(BevData& bevData, std::string folder_path, int index, bool npu_out_from_file)
{
    printf("folder_path = %s\n", folder_path.c_str());
    if (npu_out_from_file) {
        for (int i = 0; i < 6; i++) {
            std::string npu_out_folder = folder_path + "/npu_out";
            std::string npu_out_file_suffix = "_" + std::to_string(i) + ".float.bin";
            bevData.m_prepared_data.npu_out_bin[i] = get_files_with_suffix(npu_out_folder.c_str(), npu_out_file_suffix.c_str());
            bevData.m_prepared_data.buffer[i] = read_binary_file(bevData.m_prepared_data.npu_out_bin[i].c_str());
        }
    }
    bevData.m_prepared_data.bev_final_out_img = folder_path + std::to_string(index) + "_bev_final.jpg";
    bevData.m_prepared_data.ann_infos         = folder_path + "ann_infos.json";
    bevData.m_prepared_data.calibration_dicts = folder_path + "calibration_dicts.json";
    bevData.m_prepared_data.gt_boxes          = folder_path + "gt_boxes.json";
    bevData.m_prepared_data.lidar_pcd         = get_files_with_suffix(folder_path.c_str(), "_lidar_pcd.bin");
    bevData.m_prepared_data.npu_in_bin        = get_files_with_suffix(folder_path.c_str(), "_6x3x256x704.bin");
    std::string views[6] = {"CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT",
        "CAM_BACK", "CAM_BACK_RIGHT"};
    for (int i = 0; i < 6; i++) {
        bevData.m_prepared_data.img[i] = folder_path + "/" + views[i] + ".jpg";
        bevData.m_imgs[i] = cv::imread(bevData.m_prepared_data.img[i]);
        bevData.m_views[i] = views[i];
        // std::cout << bevData.m_prepared_data.img[i] << std::endl;
    }

    ParseCalibrationDictsJson(bevData);
    ParseAnnInfosJson(bevData);
    ParseGtBoxesJson(bevData);
}

void BevInitializer::ParseCalibrationDictsJson(BevData& bevData)
{
    std::string calibration_dicts_str = readJsonFile(bevData.m_prepared_data.calibration_dicts);
    nlohmann::json calibration_dicts = nlohmann::json::parse(calibration_dicts_str);
    for (int i = 0; i < 6; i++) {
        std::vector<std::vector<double>> lidar2camera(4, std::vector<double>(4, 0.0f));
        std::vector<std::vector<double>> camera2img(3, std::vector<double>(3, 0.0f));
        nlohmann::json cam = calibration_dicts[bevData.m_views[i]];
        // std::cout << m_views[i] << std::endl;
        nlohmann::json lidar2camera_json = cam["lidar2camera"];
        nlohmann::json camera2img_json   = cam["camera2img"];
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                lidar2camera[j][k] = lidar2camera_json[j][k].get<double>();
            }
        }
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                camera2img[j][k] = camera2img_json[j][k].get<double>();
            }
        }
        bevData.m_calibration_dicts_lidar2camera[i] = lidar2camera;
        bevData.m_calibration_dicts_camera2img[i]   = camera2img;
    }

    nlohmann::json ego2global_json = calibration_dicts["ego2global"];
    nlohmann::json lidar2ego_json = calibration_dicts["lidar2ego"];
    bevData.m_lidar2ego.resize(4, std::vector<double>(4, 0.0f));
    for (int j = 0; j < 4; j++) {
        for (int k = 0; k < 4; k++) {
            bevData.m_lidar2ego[j][k] = lidar2ego_json[j][k].get<double>();
        }
    }

}

void BevInitializer::ParseAnnInfosJson(BevData& bevData)
{
    std::string ann_infos_str = readJsonFile(bevData.m_prepared_data.ann_infos);
    nlohmann::json ann_infos = nlohmann::json::parse(ann_infos_str);
    // bboxes
    int bbox_num = ann_infos[0].size();
    // std::cout << bbox_num <<std::endl;
    for (int i = 0; i < bbox_num; i++) {
        std::vector<double> bbox(9);
        for (int j = 0; j < 9; j++) {
            bbox[j] = ann_infos[0][i][j].get<double>();
            // std::cout << bbox[j] <<std::endl;
        }
        bevData.m_ann_infos_bbox.emplace_back(bbox);
    }

    // cls
    std::vector<double> bbox_cls(bbox_num);
    for (int i = 0; i < bbox_num; i++) {
        bbox_cls[i] = ann_infos[1][i].get<double>();
        // std::cout << bbox_cls[i] <<std::endl;
    }
    bevData.m_ann_infos_cls = bbox_cls;
}

void BevInitializer::ParseGtBoxesJson(BevData& bevData)
{
    std::string gt_bbox_str = readJsonFile(bevData.m_prepared_data.gt_boxes);
    nlohmann::json gt_bbox = nlohmann::json::parse(gt_bbox_str);
    // bboxes
    int bbox_num = gt_bbox.size();
    // // std::cout << bbox_num <<std::endl;
    // for (int i = 0; i < bbox_num; i++) {
    //     std::vector<double> bbox(7);
    //     for (int j = 0; j < 7; j++) {
    //         bbox[j] = gt_bbox[i][j].get<double>();
    //         // std::cout << std::fixed << std::setprecision(16) << bbox[j] <<std::endl;
    //     }
    //     m_gt_boxes.emplace_back(bbox);
    // }
    for (int i = 0; i < bbox_num; i++) {
        DecoderBBox  bbox;
        bbox.x       = gt_bbox[i][0];
        bbox.y       = gt_bbox[i][1];
        bbox.z       = gt_bbox[i][2];
        bbox.w       = gt_bbox[i][3];
        bbox.l       = gt_bbox[i][4];
        bbox.h       = gt_bbox[i][5];
        bbox.feat_id = 0;
        bbox.scores  = 1.0;
        bbox.index   = 1;
        bevData.m_lidar_gt_boxes.emplace_back(bbox);
    }
}

