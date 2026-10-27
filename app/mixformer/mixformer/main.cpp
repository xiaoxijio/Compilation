#include <stdio.h>
#include <iostream>
#include <cstring>
#include <string>
#include <vector>
#include <math.h>
#include <memory.h>
#include <dirent.h>
#include <fstream>
#include <memory>
#include <tuple> 
#include <algorithm> 
#include <cmath>
#include <dirent.h>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "utils/file_utils.h"
#include "ot_avp_npu_rts.h"
#include "sample_avp_npu_common.h"

static float GOT10K_VAL_STATE[180][4] = {
    {178.0000, 220.0000, 456.0000, 315.0000}, {181.0000, 157.0000, 265.0000, 302.0000}, {750.0000, 535.0000, 157.0000, 101.0000}, 
    {1215.0000, 139.0000, 63.0000, 102.0000}, {1031.0000, 435.0000, 443.0000, 324.0000}, {328.0000, 294.0000, 581.0000, 328.0000}, 
    {467.0000, 316.0000, 456.0000, 366.0000}, {885.0000, 271.0000, 1028.0000, 719.0000}, {305.0000, 360.0000, 580.0000, 275.0000}, 
    {119.0000, 0.0000, 905.0000, 574.0000}, {1360.0000, 565.0000, 319.0000, 378.0000}, {372.0000, 173.0000, 678.0000, 478.0000},
    {365.0000, 231.0000, 77.0000, 108.0000}, {401.0000, 276.0000, 101.0000, 68.0000}, {189.0000, 212.0000, 159.0000, 129.0000},
    {831.0000, 216.0000, 103.0000, 93.0000}, {651.0000, 347.0000, 539.0000, 323.0000}, {952.0000, 383.0000, 292.0000, 242.0000}, 
    {861.0000, 390.0000, 325.0000, 235.0000}, {201.0000, 257.0000, 34.0000, 21.0000}, {57.0000, 209.0000, 66.0000, 38.0000}, 
    {454.0000, 238.0000, 36.0000, 39.0000}, {310.0000, 204.0000, 65.0000, 66.0000}, {991.0000, 180.0000, 291.0000, 210.0000},
    {320.0000, 253.0000, 158.0000, 118.0000}, {247.0000, 222.0000, 121.0000, 63.0000}, {812.0000, 208.0000, 647.0000, 275.0000}, 
    {425.0000, 144.0000, 354.0000, 294.0000}, {250.0000, 481.0000, 474.0000, 329.0000}, {788.0000, 369.0000, 374.0000, 361.0000}, 
    {274.0000, 166.0000, 113.0000, 123.0000}, {1012.0000, 320.0000, 182.0000, 315.0000}, {825.0000, 523.0000, 166.0000, 126.0000}, 
    {1023.0000, 394.0000, 412.0000, 187.0000}, {1012.0000, 468.0000, 69.0000, 32.0000}, {705.0000, 174.0000, 452.0000, 415.0000},
    {490.0000, 167.0000, 360.0000, 460.0000}, {16.0000, 369.0000, 497.0000, 250.0000}, {490.0000, 167.0000, 360.0000, 460.0000}, 
    {489.0000, 173.0000, 22.0000, 51.0000}, {279.0000, 423.0000, 1610.0000, 355.0000}, {424.0000, 434.0000, 70.0000, 48.0000}, 
    {302.0000, 166.0000, 517.0000, 335.0000}, {504.0000, 305.0000, 571.0000, 139.0000}, {549.0000, 246.0000, 164.0000, 209.0000}, 
    {1008.0000, 373.0000, 911.0000, 351.0000}, {582.0000, 271.0000, 621.0000, 339.0000}, {243.0000, 513.0000, 985.0000, 176.0000},
    {808.0000, 438.0000, 347.0000, 390.0000}, {1269.0000, 493.0000, 61.0000, 83.0000}, {799.0000, 716.0000, 292.0000, 233.0000}, 
    {602.0000, 236.0000, 449.0000, 412.0000}, {62.0000, 255.0000, 397.0000, 363.0000}, {1186.0000, 254.0000, 480.0000, 576.0000}, 
    {1054.0000, 521.0000, 45.0000, 36.0000}, {1436.0000, 302.0000, 38.0000, 372.0000}, {982.0000, 616.0000, 41.0000, 156.0000}, 
    {345.0000, 0.0000, 1574.0000, 880.0000}, {528.0000, 662.0000, 197.0000, 193.0000}, {1001.0000, 476.0000, 271.0000, 263.0000},
    {1113.0000, 455.0000, 64.0000, 131.0000}, {238.0000, 195.0000, 69.0000, 67.0000}, {469.0000, 37.0000, 696.0000, 653.0000}, 
    {311.0000, 195.0000, 163.0000, 141.0000}, {432.0000, 163.0000, 1184.0000, 738.0000}, {235.0000, 273.0000, 528.0000, 120.0000}, 
    {1045.0000, 107.0000, 830.0000, 639.0000}, {728.0000, 380.0000, 394.0000, 435.0000}, {363.0000, 347.0000, 232.0000, 147.0000}, 
    {597.0000, 594.0000, 201.0000, 259.0000}, {383.0000, 346.0000, 178.0000, 139.0000}, {591.0000, 298.0000, 310.0000, 380.0000},
    {247.0000, 350.0000, 176.0000, 107.0000}, {258.0000, 97.0000, 194.0000, 114.0000}, {590.0000, 234.0000, 158.0000, 144.0000}, 
    {632.0000, 355.0000, 81.0000, 103.0000}, {673.0000, 546.0000, 512.0000, 147.0000}, {84.0000, 178.0000, 103.0000, 75.0000}, 
    {1055.0000, 432.0000, 173.0000, 208.0000}, {960.0000, 219.0000, 188.0000, 182.0000}, {445.0000, 309.0000, 831.0000, 437.0000}, 
    {801.0000, 240.0000, 26.0000, 33.0000}, {118.0000, 211.0000, 88.0000, 142.0000}, {506.0000, 263.0000, 58.0000, 75.0000},
    {921.0000, 242.0000, 182.0000, 248.0000}, {632.0000, 46.0000, 402.0000, 601.0000}, {465.0000, 208.0000, 777.0000, 654.0000}, 
    {190.0000, 229.0000, 167.0000, 205.0000}, {1112.0000, 356.0000, 205.0000, 489.0000}, {270.0000, 38.0000, 269.0000, 384.0000}, 
    {576.0000, 240.0000, 128.0000, 156.0000}, {504.0000, 214.0000, 504.0000, 486.0000}, {640.0000, 590.0000, 720.0000, 433.0000}, 
    {709.0000, 295.0000, 831.0000, 593.0000}, {541.0000, 234.0000, 328.0000, 262.0000}, {340.0000, 266.0000, 221.0000, 173.0000},
    {549.0000, 401.0000, 157.0000, 138.0000}, {183.0000, 357.0000, 784.0000, 356.0000}, {2.0000, 0.0000, 520.0000, 189.0000}, 
    {481.0000, 541.0000, 860.0000, 325.0000}, {605.0000, 165.0000, 463.0000, 489.0000}, {289.0000, 240.0000, 385.0000, 354.0000},
    {1003.0000, 179.0000, 68.0000, 82.0000}, {966.0000, 808.0000, 83.0000, 84.0000}, {1019.0000, 334.0000, 314.0000, 209.0000}, 
    {130.0000, 49.0000, 1663.0000, 906.0000}, {416.0000, 228.0000, 45.0000, 45.0000}, {0.0000, 0.0000, 534.0000, 716.0000}, 
    {143.0000, 411.0000, 1285.0000, 411.0000}, {429.0000, 64.0000, 888.0000, 843.0000}, {576.0000, 274.0000, 122.0000, 127.0000}, 
    {781.0000, 568.0000, 495.0000, 364.0000}, {744.0000, 497.0000, 364.0000, 317.0000}, {285.0000, 166.0000, 1304.0000, 473.0000},
    {1237.0000, 408.0000, 376.0000, 150.0000}, {465.0000, 44.0000, 533.0000, 571.0000}, {39.0000, 398.0000, 727.0000, 129.0000}, 
    {257.0000, 33.0000, 782.0000, 491.0000}, {671.0000, 380.0000, 110.0000, 156.0000}, {608.0000, 469.0000, 485.0000, 291.0000}, 
    {0.0000, 201.0000, 1018.0000, 854.0000}, {733.0000, 315.0000, 771.0000, 570.0000}, {780.0000, 340.0000, 466.0000, 277.0000}, 
    {754.0000, 432.0000, 270.0000, 170.0000}, {45.0000, 80.0000, 315.0000, 159.0000}, {1270.0000, 491.0000, 222.0000, 195.0000},
    {719.0000, 547.0000, 310.0000, 424.0000}, {694.0000, 377.0000, 54.0000, 157.0000}, {458.0000, 523.0000, 321.0000, 142.0000}, 
    {1819.0000, 760.0000, 98.0000, 98.0000}, {303.0000, 262.0000, 513.0000, 210.0000}, {497.0000, 232.0000, 128.0000, 152.0000}, 
    {436.0000, 356.0000, 56.0000, 84.0000}, {280.0000, 135.0000, 231.0000, 123.0000}, {446.0000, 279.0000, 13.0000, 36.0000}, 
    {749.0000, 647.0000, 290.0000, 260.0000}, {271.0000, 171.0000, 69.0000, 77.0000}, {726.0000, 399.0000, 47.0000, 133.0000},
    {350.0000, 209.0000, 880.0000, 737.0000}, {852.0000, 156.0000, 415.0000, 521.0000}, {597.0000, 309.0000, 89.0000, 101.0000}, 
    {250.0000, 285.0000, 800.0000, 400.0000}, {240.0000, 283.0000, 976.0000, 661.0000}, {878.0000, 420.0000, 171.0000, 226.0000}, 
    {345.0000, 94.0000, 101.0000, 124.0000}, {848.0000, 585.0000, 241.0000, 292.0000}, {761.0000, 547.0000, 232.0000, 113.0000}, 
    {682.0000, 301.0000, 239.0000, 173.0000}, {663.0000, 328.0000, 279.0000, 191.0000}, {188.0000, 107.0000, 114.0000, 152.0000},
    {0.0000, 458.0000, 1867.0000, 180.0000}, {1.0000, 170.0000, 425.0000, 87.0000}, {177.0000, 377.0000, 1654.0000, 414.0000}, 
    {208.0000, 396.0000, 1710.0000, 362.0000}, {655.0000, 472.0000, 199.0000, 145.0000}, {378.0000, 74.0000, 511.0000, 344.0000}, 
    {233.0000, 86.0000, 103.0000, 162.0000}, {593.0000, 112.0000, 241.0000, 411.0000}, {528.0000, 312.0000, 513.0000, 141.0000}, 
    {1925.0000, 718.0000, 82.0000, 179.0000}, {641.0000, 470.0000, 410.0000, 257.0000}, {650.0000, 0.0000, 344.0000, 1078.0000},
    {882.0000, 324.0000, 529.0000, 517.0000}, {335.0000, 151.0000, 301.0000, 318.0000}, {942.0000, 62.0000, 449.0000, 716.0000}, 
    {450.0000, 362.0000, 413.0000, 327.0000}, {814.0000, 674.0000, 162.0000, 319.0000}, {790.0000, 817.0000, 388.0000, 198.0000}, 
    {510.0000, 273.0000, 585.0000, 446.0000}, {725.0000, 479.0000, 275.0000, 240.0000}, {382.0000, 247.0000, 99.0000, 49.0000}, 
    {846.0000, 352.0000, 185.0000, 88.0000}, {412.0000, 469.0000, 201.0000, 244.0000}, {626.0000, 775.0000, 432.0000, 251.0000},
    {614.0000, 353.0000, 10.0000, 24.0000}, {910.0000, 400.0000, 260.0000, 516.0000}, {920.0000, 980.0000, 186.0000, 236.0000}, 
    {2558.0000, 748.0000, 518.0000, 561.0000}, {434.0000, 131.0000, 363.0000, 264.0000}, {1527.0000, 741.0000, 812.0000, 806.0000}, 
};
void MakeDir(const std::string folder_path) {
    std::string command;
    command = "mkdir -p " + folder_path;  
    int ret = system(command.c_str());
    if (ret != 0) {
        std::cout << " Fail to build " << folder_path << " ." << std::endl;
    } 
}

int CountImgFiles(const std::string& directory_path) {
    int count = 0;
    DIR* dir;
    struct dirent* entry;
    dir = opendir(directory_path.c_str());
    if (dir == NULL) {
        std::cerr << "Error opening directory: " << directory_path << std::endl;
        return -1;
    }
    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        std::string file_name = entry->d_name;
        if (file_name.length() >= 4 && ((file_name.substr(file_name.length() - 4) == ".jpg") || (file_name.substr(file_name.length() - 4) == ".png"))) {
            count++;
        }
    }
    closedir(dir);
    return count;
}

std::vector<std::string> GetSortedImgs(const std::string& dir_path){
    DIR *dir = opendir(dir_path.c_str());
    if (dir == NULL) {
        std::cout << "opendir error" << std::endl;
    }
    std::vector<std::string> files_path;
    files_path.clear();
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_DIR) { 
            continue;
        } else {
            std::string file_name = entry->d_name;
            // std::cout << file_name << std::endl;
            if (file_name.length() >= 4 && ((file_name.substr(file_name.length() - 4) == ".jpg") || (file_name.substr(file_name.length() - 4) == ".png"))) {
                files_path.push_back(file_name);
            }
        }
    }
    closedir(dir);
    std::sort(files_path.begin(), files_path.end(), [](const std::string& a, const std::string& b) {
        // std::string name_a = a.substr(0, a.length() - 4);
        const int name_a = std::stoi(a.substr(0, a.length() - 4)); // Also, it can be converted to integers.
        // std::cout << name_a << std::endl;
        // std::string name_b = b.substr(0, b.length() - 4);
        const int name_b = std::stoi(b.substr(0, b.length() - 4)); // Also, it can be converted to integers.
        return name_a < name_b;
    });
    return files_path;
}

std::vector<std::string> GetSortedDirs(const std::string& dir_path){
    DIR *dir = opendir(dir_path.c_str());
    if (dir == NULL) {
        std::cout << "opendir error" << std::endl;
    }
    std::vector<std::string> files_path;
    files_path.clear();
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_DIR) { 
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
                continue;
            }
            std::string new_dir = entry->d_name;
            files_path.push_back(new_dir);
        }
    }
    closedir(dir);
    std::sort(files_path.begin(), files_path.end(), [](const std::string& a, const std::string& b) {
        return a < b;
    });
    return files_path;
}

std::vector<std::string> GetFilesList(std::string dir_path){
    DIR *dir = opendir(dir_path.c_str());
    if (dir == NULL) {
        std::cout << "opendir error" << std::endl;
    }
    std::vector<std::string> files_path;
    files_path.clear();
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_DIR) { 
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
                continue;
            std::string new_dir = dir_path + "/" + entry->d_name;
            std::vector<std::string> temp_path = GetFilesList(new_dir);
            files_path.insert(files_path.end(), temp_path.begin(), temp_path.end());
        } else {
            std::string name = entry->d_name;
            std::string data_dir = dir_path + "/" + name;
            files_path.push_back(data_dir);
        }
    }
    closedir(dir);
    return files_path;
}


sample_avp_tensor_t *GetTensorByTensorName(sample_avp_model_task_info_t* layer, bool b_input, std::string name) {
    unsigned int idx = 0U;
    int ret = -1;
    if (b_input) {
        ret = ot_avp_npu_get_input_index_by_name(layer->handle, name.c_str(), name.size(), &idx);
    } else {
        ret = ot_avp_npu_get_output_index_by_name(layer->handle, name.c_str(), name.size(), &idx);
    }
    
    if (ret != 0) {
        std::cout << "get idx by name[" << name << "]" << "fail" << std::endl;
        return nullptr;
    }
    return (b_input ? &layer->input_tensor[idx]: &layer->output_tensor[idx]);
}

int32_t FillTensorFromBuf(sample_avp_tensor_t* tensor, uint8_t* buf, uint32_t buf_size, ot_avp_data_type buf_dtype) {
        if (tensor->dtype == buf_dtype) {
            (void)memcpy((void *)tensor->virt_addr, buf, buf_size);
            return 0;
        }
        std::cout << "not support FillTensorFromBuf." << std::endl;
        return -1;
}

int32_t FillBufFromTensor(uint8_t* buf, uint32_t buf_size, ot_avp_data_type buf_dtype, sample_avp_tensor_t* in_tensor) {
    if (in_tensor->dtype == buf_dtype) {
        (void)memcpy(buf, (ot_void *)in_tensor->virt_addr, buf_size);
        return 0;
    }
    std::cout << "not support FillBufFromTensor." << std::endl;
    return -1;
}

class MixformerHead {
    public:
        MixformerHead(const float &search_sz, const std::string &para_path);
        ~MixformerHead(); 
        void ParameterInit();
        void DeleteAssignParameterMemory();
        void GetBackboneInfo(const std::string &backbone_path);
        void GetBackboneInfo(sample_avp_tensor_t* const in_tensor);
        std::vector<float> Gemm(const std::vector<float> &w, const std::vector<float> &b, const std::vector<float> &x);
        std::vector<float> GemmT(const std::vector<float> &w, const std::vector<float> &b, const std::vector<float> &x);
        std::vector<float> LayerNorm(const std::vector<float> &w, const std::vector<float> &b, const std::vector<float> &x);
        std::vector<float> Softmax(const std::vector<float> &x);
        float Sigmoid(const float &x);
        float RunnerBoxes(const std::string &coordinate_point_type);
        std::vector<float> TlbrToCxywh(const float &t, const float &l, const float &b, const float &r);
        float RunnerScore();

    private:
        float search_size;
        int dim = 768;
        int grid = 96;
        // model parameters info
        std::string parameters_path;
        // boxes-gemm-1
        char *boxes_gemm_w1_info_file = nullptr;
        float *boxes_gemm_w1_info = nullptr;
        std::vector<float> boxes_gemm_w1;
        char *boxes_gemm_b1_info_file = nullptr;
        float *boxes_gemm_b1_info = nullptr;
        std::vector<float> boxes_gemm_b1;
        // boxes-gemm-2
        char *boxes_gemm_w2_info_file = nullptr;
        float *boxes_gemm_w2_info = nullptr;
        std::vector<float> boxes_gemm_w2;
        char *boxes_gemm_b2_info_file = nullptr;
        float *boxes_gemm_b2_info = nullptr;
        std::vector<float> boxes_gemm_b2;
        // boxes-layerN-1
        char *boxes_layerN_w1_info_file = nullptr;
        float *boxes_layerN_w1_info = nullptr;
        std::vector<float> boxes_layerN_w1;
        char *boxes_layerN_b1_info_file = nullptr;
        float *boxes_layerN_b1_info = nullptr;
        std::vector<float> boxes_layerN_b1;
        // boxes-layerN-2
        char *boxes_layerN_w2_info_file = nullptr;
        float *boxes_layerN_w2_info = nullptr;
        std::vector<float> boxes_layerN_w2;
        char *boxes_layerN_b2_info_file = nullptr;
        float *boxes_layerN_b2_info = nullptr;
        std::vector<float> boxes_layerN_b2;
        // boxes-grid
        char *boxes_grid_info_file = nullptr;
        float *boxes_grid_info = nullptr;
        std::vector<float> boxes_grid;
        // score-gemm-1
        char *score_gemm_w1_info_file = nullptr;
        float *score_gemm_w1_info = nullptr;
        std::vector<float> score_gemm_w1;
        char *score_gemm_b1_info_file = nullptr;
        float *score_gemm_b1_info = nullptr;
        std::vector<float> score_gemm_b1;
        // score-gemm-2
        char *score_gemm_w2_info_file = nullptr;
        float *score_gemm_w2_info = nullptr;
        std::vector<float> score_gemm_w2;
        char *score_gemm_b2_info_file = nullptr;
        float *score_gemm_b2_info = nullptr;
        std::vector<float> score_gemm_b2;
        //  backbone-input
        float *pred_backbone_info = nullptr;
        // std::vector<float> pred_backbone;

};

MixformerHead::MixformerHead(const float &search_sz, const std::string &para_path) {
    search_size = search_sz;
    parameters_path = para_path;
}

MixformerHead::~MixformerHead() {
    DeleteAssignParameterMemory();
    if (pred_backbone_info != nullptr) {
        delete [] pred_backbone_info;
    }
}

void MixformerHead::DeleteAssignParameterMemory() {
    // boxes-gemm
    if (boxes_gemm_w1_info_file != nullptr) {
        delete [] boxes_gemm_w1_info_file;
    }
    if (boxes_gemm_w1_info != nullptr) {
        delete [] boxes_gemm_w1_info;
    }
    if (boxes_gemm_b1_info_file != nullptr) {
        delete [] boxes_gemm_b1_info_file;
    }
    if (boxes_gemm_b1_info != nullptr) {
        delete [] boxes_gemm_b1_info;
    }
    if (boxes_gemm_w2_info_file != nullptr) {
        delete [] boxes_gemm_w2_info_file;
    }
    if (boxes_gemm_w2_info != nullptr) {
        delete [] boxes_gemm_w2_info;
    }
    if (boxes_gemm_b2_info_file != nullptr) {
        delete [] boxes_gemm_b2_info_file;
    }
    if (boxes_gemm_b2_info != nullptr) {
        delete [] boxes_gemm_b2_info;
    }
    // boxes-layerN
    if (boxes_layerN_w1_info_file != nullptr) {
        delete [] boxes_layerN_w1_info_file;
    }
    if (boxes_layerN_w1_info != nullptr) {
        delete [] boxes_layerN_w1_info;
    }
    if (boxes_layerN_b1_info_file != nullptr) {
        delete [] boxes_layerN_b1_info_file;
    }
    if (boxes_layerN_b1_info != nullptr) {
        delete [] boxes_layerN_b1_info;
    }
    if (boxes_layerN_w2_info_file != nullptr) {
        delete [] boxes_layerN_w2_info_file;
    }
    if (boxes_layerN_w2_info != nullptr) {
        delete [] boxes_layerN_w2_info;
    }
    if (boxes_layerN_b2_info_file != nullptr) {
        delete [] boxes_layerN_b2_info_file;
    }
    if (boxes_layerN_b2_info != nullptr) {
        delete [] boxes_layerN_b2_info;
    }
    // grid
    if (boxes_grid_info_file != nullptr) {
        delete [] boxes_grid_info_file;
    }
    if (boxes_grid_info != nullptr) {
        delete [] boxes_grid_info;
    }
    // score-gemm
    if (score_gemm_w1_info_file != nullptr) {
        delete [] score_gemm_w1_info_file;
    }
    if (score_gemm_w1_info != nullptr) {
        delete [] score_gemm_w1_info;
    }
    if (score_gemm_b1_info_file != nullptr) {
        delete [] score_gemm_b1_info_file;
    }
    if (score_gemm_b1_info != nullptr) {
        delete [] score_gemm_b1_info;
    }
    if (score_gemm_w2_info_file != nullptr) {
        delete [] score_gemm_w2_info_file;
    }
    if (score_gemm_w2_info != nullptr) {
        delete [] score_gemm_w2_info;
    }
    if (score_gemm_b2_info_file != nullptr) {
        delete [] score_gemm_b2_info_file;
    }
    if (score_gemm_b2_info != nullptr) {
        delete [] score_gemm_b2_info;
    }
}

void MixformerHead::ParameterInit() {
    // DeleteAssignMemory();
    // boxes-gemm-w1
    std::string boxes_gemm_w1_path = parameters_path + "/" + "boxes_gemm_w1.bin";
    boxes_gemm_w1_info_file = new char [boxes_gemm_w1_path.length() + 1];
    std::strcpy(boxes_gemm_w1_info_file, boxes_gemm_w1_path.c_str());
    const auto boxes_gemm_w1_size = get_bin_file_size(boxes_gemm_w1_info_file);
    const auto boxes_gemm_w1_num = boxes_gemm_w1_size / sizeof(float);
    boxes_gemm_w1_info = new float[boxes_gemm_w1_num];
    uint8_t *boxes_gemm_w1_info_tmp = reinterpret_cast<uint8_t *>(boxes_gemm_w1_info);
    read_data_from_bin_file(boxes_gemm_w1_info_file, boxes_gemm_w1_info_tmp, boxes_gemm_w1_size);
    boxes_gemm_w1 = std::vector<float>(boxes_gemm_w1_info, boxes_gemm_w1_info + boxes_gemm_w1_num);
    // boxes-gemm-b1
    std::string boxes_gemm_b1_path = parameters_path + "/" + "boxes_gemm_b1.bin";
    boxes_gemm_b1_info_file = new char [boxes_gemm_b1_path.length() + 1];
    std::strcpy(boxes_gemm_b1_info_file, boxes_gemm_b1_path.c_str());
    const auto boxes_gemm_b1_size = get_bin_file_size(boxes_gemm_b1_info_file);
    const auto boxes_gemm_b1_num = boxes_gemm_b1_size / sizeof(float);
    boxes_gemm_b1_info = new float[boxes_gemm_b1_num];
    uint8_t *boxes_gemm_b1_info_tmp = reinterpret_cast<uint8_t *>(boxes_gemm_b1_info);
    read_data_from_bin_file(boxes_gemm_b1_info_file, boxes_gemm_b1_info_tmp, boxes_gemm_b1_size);
    boxes_gemm_b1 = std::vector<float>(boxes_gemm_b1_info, boxes_gemm_b1_info + boxes_gemm_b1_num);
    // boxes-gemm-w2
    std::string boxes_gemm_w2_path = parameters_path + "/" + "boxes_gemm_w2.bin";
    boxes_gemm_w2_info_file = new char [boxes_gemm_w2_path.length() + 1];
    std::strcpy(boxes_gemm_w2_info_file, boxes_gemm_w2_path.c_str());
    const auto boxes_gemm_w2_size = get_bin_file_size(boxes_gemm_w2_info_file);
    const auto boxes_gemm_w2_num = boxes_gemm_w2_size / sizeof(float);
    boxes_gemm_w2_info = new float[boxes_gemm_w2_num];
    uint8_t *boxes_gemm_w2_info_tmp = reinterpret_cast<uint8_t *>(boxes_gemm_w2_info);
    read_data_from_bin_file(boxes_gemm_w2_info_file, boxes_gemm_w2_info_tmp, boxes_gemm_w2_size);
    boxes_gemm_w2 = std::vector<float>(boxes_gemm_w2_info, boxes_gemm_w2_info + boxes_gemm_w2_num);
    // boxes-gemm-b2
    std::string boxes_gemm_b2_path = parameters_path + "/" + "boxes_gemm_b2.bin";
    boxes_gemm_b2_info_file = new char [boxes_gemm_b2_path.length() + 1];
    std::strcpy(boxes_gemm_b2_info_file, boxes_gemm_b2_path.c_str());
    const auto boxes_gemm_b2_size = get_bin_file_size(boxes_gemm_b2_info_file);
    const auto boxes_gemm_b2_num = boxes_gemm_b2_size / sizeof(float);
    boxes_gemm_b2_info = new float[boxes_gemm_b2_num];
    uint8_t *boxes_gemm_b2_info_tmp = reinterpret_cast<uint8_t *>(boxes_gemm_b2_info);
    read_data_from_bin_file(boxes_gemm_b2_info_file, boxes_gemm_b2_info_tmp, boxes_gemm_b2_size);
    boxes_gemm_b2 = std::vector<float>(boxes_gemm_b2_info, boxes_gemm_b2_info + boxes_gemm_b2_num);
    // boxes-layerN-w1
    std::string boxes_layerN_w1_path = parameters_path + "/" + "boxes_layerN_w1.bin";
    boxes_layerN_w1_info_file = new char [boxes_layerN_w1_path.length() + 1];
    std::strcpy(boxes_layerN_w1_info_file, boxes_layerN_w1_path.c_str());
    const auto boxes_layerN_w1_size = get_bin_file_size(boxes_layerN_w1_info_file);
    const auto boxes_layerN_w1_num = boxes_layerN_w1_size / sizeof(float);
    boxes_layerN_w1_info = new float[boxes_layerN_w1_num];
    uint8_t *boxes_layerN_w1_info_tmp = reinterpret_cast<uint8_t *>(boxes_layerN_w1_info);
    read_data_from_bin_file(boxes_layerN_w1_info_file, boxes_layerN_w1_info_tmp, boxes_layerN_w1_size);
    boxes_layerN_w1 = std::vector<float>(boxes_layerN_w1_info, boxes_layerN_w1_info + boxes_layerN_w1_num);
    // boxes-layerN-b1
    std::string boxes_layerN_b1_path = parameters_path + "/" + "boxes_layerN_b1.bin";
    boxes_layerN_b1_info_file = new char [boxes_layerN_b1_path.length() + 1];
    std::strcpy(boxes_layerN_b1_info_file, boxes_layerN_b1_path.c_str());
    const auto boxes_layerN_b1_size = get_bin_file_size(boxes_layerN_b1_info_file);
    const auto boxes_layerN_b1_num = boxes_layerN_b1_size / sizeof(float);
    boxes_layerN_b1_info = new float[boxes_layerN_b1_num];
    uint8_t *boxes_layerN_b1_info_tmp = reinterpret_cast<uint8_t *>(boxes_layerN_b1_info);
    read_data_from_bin_file(boxes_layerN_b1_info_file, boxes_layerN_b1_info_tmp, boxes_layerN_b1_size);
    boxes_layerN_b1 = std::vector<float>(boxes_layerN_b1_info, boxes_layerN_b1_info + boxes_layerN_b1_num);
    // boxes-layerN-w2
    std::string boxes_layerN_w2_path = parameters_path + "/" + "boxes_layerN_w2.bin";
    boxes_layerN_w2_info_file = new char [boxes_layerN_w2_path.length() + 1];
    std::strcpy(boxes_layerN_w2_info_file, boxes_layerN_w2_path.c_str());
    const auto boxes_layerN_w2_size = get_bin_file_size(boxes_layerN_w2_info_file);
    const auto boxes_layerN_w2_num = boxes_layerN_w2_size / sizeof(float);
    boxes_layerN_w2_info = new float[boxes_layerN_w2_num];
    uint8_t *boxes_layerN_w2_info_tmp = reinterpret_cast<uint8_t *>(boxes_layerN_w2_info);
    read_data_from_bin_file(boxes_layerN_w2_info_file, boxes_layerN_w2_info_tmp, boxes_layerN_w2_size);
    boxes_layerN_w2 = std::vector<float>(boxes_layerN_w2_info, boxes_layerN_w2_info + boxes_layerN_w2_num);
    // boxes-layerN-b2
    std::string boxes_layerN_b2_path = parameters_path + "/" + "boxes_layerN_b2.bin";
    boxes_layerN_b2_info_file = new char [boxes_layerN_b2_path.length() + 1];
    std::strcpy(boxes_layerN_b2_info_file, boxes_layerN_b2_path.c_str());
    const auto boxes_layerN_b2_size = get_bin_file_size(boxes_layerN_b2_info_file);
    const auto boxes_layerN_b2_num = boxes_layerN_b2_size / sizeof(float);
    boxes_layerN_b2_info = new float[boxes_layerN_b2_num];
    uint8_t *boxes_layerN_b2_info_tmp = reinterpret_cast<uint8_t *>(boxes_layerN_b2_info);
    read_data_from_bin_file(boxes_layerN_b2_info_file, boxes_layerN_b2_info_tmp, boxes_layerN_b2_size);
    boxes_layerN_b2 = std::vector<float>(boxes_layerN_b2_info, boxes_layerN_b2_info + boxes_layerN_b2_num);
    // boxes-grid
    std::string boxes_grid_path = parameters_path + "/" + "boxes_grid.bin";
    boxes_grid_info_file = new char [boxes_grid_path.length() + 1];
    std::strcpy(boxes_grid_info_file, boxes_grid_path.c_str());
    const auto boxes_grid_size = get_bin_file_size(boxes_grid_info_file);
    const auto boxes_grid_num = boxes_grid_size / sizeof(float);
    boxes_grid_info = new float[boxes_grid_num];
    uint8_t *boxes_grid_info_tmp = reinterpret_cast<uint8_t *>(boxes_grid_info);
    read_data_from_bin_file(boxes_grid_info_file, boxes_grid_info_tmp, boxes_grid_size);
    boxes_grid = std::vector<float>(boxes_grid_info, boxes_grid_info + boxes_grid_num);
    // score-gemm-w1
    std::string score_gemm_w1_path = parameters_path + "/" + "score_gemm_w1.bin";
    score_gemm_w1_info_file = new char [score_gemm_w1_path.length() + 1];
    std::strcpy(score_gemm_w1_info_file, score_gemm_w1_path.c_str());
    const auto score_gemm_w1_size = get_bin_file_size(score_gemm_w1_info_file);
    const auto score_gemm_w1_num = score_gemm_w1_size / sizeof(float);
    score_gemm_w1_info = new float[score_gemm_w1_num];
    uint8_t *score_gemm_w1_info_tmp = reinterpret_cast<uint8_t *>(score_gemm_w1_info);
    read_data_from_bin_file(score_gemm_w1_info_file, score_gemm_w1_info_tmp, score_gemm_w1_size);
    score_gemm_w1 = std::vector<float>(score_gemm_w1_info, score_gemm_w1_info + score_gemm_w1_num);
    // score-gemm-b1
    std::string score_gemm_b1_path = parameters_path + "/" + "score_gemm_b1.bin";
    score_gemm_b1_info_file = new char [score_gemm_b1_path.length() + 1];
    std::strcpy(score_gemm_b1_info_file, score_gemm_b1_path.c_str());
    const auto score_gemm_b1_size = get_bin_file_size(score_gemm_b1_info_file);
    const auto score_gemm_b1_num = score_gemm_b1_size / sizeof(float);
    score_gemm_b1_info = new float[score_gemm_b1_num];
    uint8_t *score_gemm_b1_info_tmp = reinterpret_cast<uint8_t *>(score_gemm_b1_info);
    read_data_from_bin_file(score_gemm_b1_info_file, score_gemm_b1_info_tmp, score_gemm_b1_size);
    score_gemm_b1 = std::vector<float>(score_gemm_b1_info, score_gemm_b1_info + score_gemm_b1_num);
    // score-gemm-w2
    std::string score_gemm_w2_path = parameters_path + "/" + "score_gemm_w2.bin";
    score_gemm_w2_info_file = new char [score_gemm_w2_path.length() + 1];
    std::strcpy(score_gemm_w2_info_file, score_gemm_w2_path.c_str());
    const auto score_gemm_w2_size = get_bin_file_size(score_gemm_w2_info_file);
    const auto score_gemm_w2_num = score_gemm_w2_size / sizeof(float);
    score_gemm_w2_info = new float[score_gemm_w2_num];
    uint8_t *score_gemm_w2_info_tmp = reinterpret_cast<uint8_t *>(score_gemm_w2_info);
    read_data_from_bin_file(score_gemm_w2_info_file, score_gemm_w2_info_tmp, score_gemm_w2_size);
    score_gemm_w2 = std::vector<float>(score_gemm_w2_info, score_gemm_w2_info + score_gemm_w2_num);
    // score-gemm-b2
    std::string score_gemm_b2_path = parameters_path + "/" + "score_gemm_b2.bin";
    score_gemm_b2_info_file = new char [score_gemm_b2_path.length() + 1];
    std::strcpy(score_gemm_b2_info_file, score_gemm_b2_path.c_str());
    const auto score_gemm_b2_size = get_bin_file_size(score_gemm_b2_info_file);
    const auto score_gemm_b2_num = score_gemm_b2_size / sizeof(float);
    score_gemm_b2_info = new float[score_gemm_b2_num];
    uint8_t *score_gemm_b2_info_tmp = reinterpret_cast<uint8_t *>(score_gemm_b2_info);
    read_data_from_bin_file(score_gemm_b2_info_file, score_gemm_b2_info_tmp, score_gemm_b2_size);
    score_gemm_b2 = std::vector<float>(score_gemm_b2_info, score_gemm_b2_info + score_gemm_b2_num);
}

void MixformerHead::GetBackboneInfo(sample_avp_tensor_t* const in_tensor) {
    if (pred_backbone_info != nullptr) {
        delete [] pred_backbone_info;
    }
    pred_backbone_info = new float[298 * dim];
    const uint32_t buf_size = 298 * dim * sizeof(float);
    uint8_t *pred_backbone_info_tmp = reinterpret_cast<uint8_t *>(pred_backbone_info);
    int32_t ret = FillBufFromTensor(pred_backbone_info_tmp, buf_size, OT_AVP_DTYPE_F32, in_tensor);
    if (ret != 0) {
        std::cout << "model out fail." << std::endl;
        exit(1);
    } 
}

std::vector<float> MixformerHead::Softmax(const std::vector<float> &x) {
    float total = 0.0;
    for (auto a : x)  {
        total += exp(a);
    }
    std::vector<float> result;
    result.clear();
    for (auto a : x)  {
        result.push_back(exp(a) / total);
    }
    return result;
}

float MixformerHead::Sigmoid(const float &x) {
    if (x > 0.0) {
        return 1 / (1 + exp(-x));
    } else {
        return exp(x) / (1 + exp(x));
    }
}

std::vector<float> MixformerHead::Gemm(const std::vector<float> &w, const std::vector<float> &b, const std::vector<float> &x) {
    std::vector<float> result;
    result.clear();
    const int w_clos = b.size();
    const int x_rows = x.size() / dim;
    const int w_rows = w.size() / w_clos;
    if (w_rows != dim) {
        std::cerr << "wrong input for gemm." << std::endl;
    }
    for (auto i = 0; i < x_rows; i++) {
        for (auto j = 0; j < w_clos ; j++) {
            float res = 0.0;
            for (auto k = 0; k < w_rows; k++) {
                res += x[i * dim + k] * w[k * w_clos + j];
            }
            result.push_back(res);
        } 
    }
    for (auto i = 0; i < x_rows; i++) {
        for (auto j = 0; j < w_clos; j++) {
            result[i * w_clos + j] += b[j];
        }
    }
    return result;
}

std::vector<float> MixformerHead::GemmT(const std::vector<float> &w, const std::vector<float> &b, const std::vector<float> &x) {
    std::vector<float> result;
    result.clear();
    const int w_clos = b.size();
    const int x_rows = x.size() / dim;
    const int w_rows = w.size() / w_clos;
    if (w_rows != dim) {
        std::cerr << "wrong input for gemmT." << std::endl;
    }
    for (auto i = 0; i < x_rows; i++) {
        for (auto j = 0; j < w_clos ; j++) {
            float res = 0.0;
            for (auto k = 0; k < w_rows; k++) {
                res += x[i * dim + k] * w[j * w_rows + k];
            }
            result.push_back(res);
        } 
    }
    for (auto i = 0; i < x_rows; i++) {
        for (auto j = 0; j < w_clos; j++) {
            result[i * w_clos + j] += b[j];
        }
    }
    return result;
}

std::vector<float> MixformerHead::LayerNorm(const std::vector<float> &w, const std::vector<float> &b, const std::vector<float> &x) {
    if (w.size() != b.size()) {
        std::cerr << "wrong weight and bias for layernorm." << std::endl;
    }
    const int clos = w.size();
    const int rows = x.size() / clos;
    std::vector<float> expects;
    expects.clear();
    std::vector<float> vars;
    vars.clear();
    // expect and variance
    for (auto i = 0; i < rows; i++) {
        float total = 0.0;
        for (auto j = 0; j < clos; j++) {
            total += x[i * clos + j];
        }
        const float mean = total / float(clos);
        for (auto j = 0; j < clos; j++) {
            const float expect = x[i * clos + j] - mean;
            expects.push_back(expect);
            const float var = expect * expect;
            vars.push_back(var);
        }
    }
    // middle results for layernorm caculation.
    std::vector<float> tmps;
    tmps.clear();
    for (auto i = 0; i < rows; i++) {
        float total = 0.0;
        for (auto j = 0; j < clos; j++) {
            total += vars[i * clos + j];
        }
        const float mean = total / float(clos) + 1e-6;
        const float tmp = std::sqrt(mean);
        for (auto j = 0; j < clos; j++) {
            const float div = expects[i * clos + j] / tmp;
            tmps.push_back(div);
        }    
    }
    // w * x + b for layernorm 
    std::vector<float> result;
    result.clear();
    for (auto i = 0; i < rows; i++) {
        float tmp = 0.0;
        for (auto j = 0; j < clos; j++) {
            tmp = tmps[i * clos + j] * w[j] + b[j];
            result.push_back(tmp);
        }
    }
    return result;
}

float MixformerHead::RunnerBoxes(const std::string &coordinate_point_type) {
    std::vector<float> data;
    if ((coordinate_point_type == "t") || (coordinate_point_type == "T")) {
        float *start_point = &pred_backbone_info[294 * dim];
        data = std::vector<float>(start_point, start_point + dim);
    } else if ((coordinate_point_type == "l") || (coordinate_point_type == "L")) {
        float *start_point = &pred_backbone_info[295 * dim];
        data = std::vector<float>(start_point, start_point + dim);
    } else if ((coordinate_point_type == "b") || (coordinate_point_type == "B")) {
        float *start_point = &pred_backbone_info[296 * dim];
        data = std::vector<float>(start_point, start_point + dim);
    }  else if ((coordinate_point_type == "r") || (coordinate_point_type == "R")) {
        float *start_point = &pred_backbone_info[297 * dim];
        data = std::vector<float>(start_point, start_point + dim);
    }
    std::vector<float> gemm_1 = GemmT(boxes_gemm_w1, boxes_gemm_b1, data);
    std::vector<float> layernorm_1 = LayerNorm(boxes_layerN_w1, boxes_layerN_b1, gemm_1);
    // relu operation
    for (auto i = 0; i < layernorm_1.size(); i++) {
        if (layernorm_1[i] < 0.0) {
            layernorm_1[i] = 0.0;
        }
    }
    std::vector<float> gemm_2 = GemmT(boxes_gemm_w2, boxes_gemm_b2, layernorm_1);
    std::vector<float> layernorm_2 = LayerNorm(boxes_layerN_w2, boxes_layerN_b2, gemm_2);
    std::vector<float> softmax = Softmax(layernorm_2);
    // mul * grid info
    if (softmax.size() != boxes_grid.size()) {
        std::cerr << "wrong input for grid caculation." << std::endl;
    }
    // reduce_sum
    float res = 0.0;
    for (auto i = 0; i < softmax.size(); i++) {
        res += softmax[i] * boxes_grid[i];
    }
    return res;
}

float MixformerHead::RunnerScore() {
    std::vector<float> data;
    float *start_point = &pred_backbone_info[294 * dim];
    data = std::vector<float>(start_point, start_point + 4 * dim);
  
    std::vector<float> gemm_1 = Gemm(score_gemm_w1, score_gemm_b1, data);
    // relu operation
    for (auto i = 0; i < gemm_1.size(); i++) {
        if (gemm_1[i] < 0.0) {
            gemm_1[i] = 0.0;
        }
    }
    std::vector<float> gemm_2 = Gemm(score_gemm_w2, score_gemm_b2, gemm_1);
    float total = 0.0;
    for (auto i = 0; i < gemm_2.size(); i++) {
        total += gemm_2[i];
    }
    const float reduce_mean = total / float(gemm_2.size());
    return Sigmoid(reduce_mean);
}

std::vector<float> MixformerHead::TlbrToCxywh(const float &t, const float &l, const float &b, const float &r) {
    std::vector<float> bbox;
    bbox.clear();
    const float x1 = t / search_size;
    const float x2 = l / search_size;
    const float y1 = b / search_size;
    const float y2 = r / search_size;
    bbox.push_back((x1 + x2) / 2.0);
    bbox.push_back((y1 + y2) / 2.0);
    bbox.push_back(x2 - x1);
    bbox.push_back(y2 - y1);
    return bbox;
}

class Tracker {
    public:
        Tracker(const std::string &para_path);
        ~Tracker();
        void TrackerInfoInit(const std::string &first_image_path, const float *state_init, const std::string &image_save_path, const std::string &file_save_path, const std::string &file_name);
        void InferModelInit(const std::string &ortm_model_path);
        void VideoCaptureByFrame(const std::string &video_frame_path);
        void Runner(const int &frame_idx, const std::string &save_path, const std::string &file_save_path);
        cv::Mat SampleTarget(const float &search_area_factor, const float &output_size, const std::string &search_type);
        std::vector<float> Preprocess(const cv::Mat &sample_target);
        void TrackerDestroy();
        void CloseStatisticfile() { out_file.close(); };
    private:
        std::shared_ptr<MixformerHead> head;
        // model preparing
        char *model_path = nullptr;
        sample_avp_model_task_info_t mixformer_model;
        // basic tracking parameters
        float max_score_decay = 1.0;
        // float search_factor = 4.55;
        // float search_factor = 4.5; //0829 template_factor = 2.0
        // float search_factor = 4.45; //0830 template_factor = 2.0
        float search_factor = 4.0; //0831 template_factor = 2.0: 67.84, 71.34 
        float search_size = 224.0;
        float template_factor = 2.0;
        float template_size = 112.0;
        int update_interval = 200;
        int online_size = 1;
        float max_pred_score = -1.0;
        // related to tracker runner
        cv::Mat image;
        float state[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        int WIDTH = 0;
        int HEIGHT = 0;
        int MARGIN = 10;
        // sample target info
        float current_resize_factor = 0.0;
        const float mean[3] = {0.485, 0.456, 0.406};
        const float std[3] = {0.229, 0.224, 0.225};
        std::vector<float> search_area;
        std::vector<float> current_template;
        std::vector<float> online_template;
        std::vector<float> online_max_template;
        // save the tracking results
        std::ofstream out_file;
        std::string current_file_name;
};

Tracker::Tracker(const std::string &para_path) {
    head = std::make_shared<MixformerHead>(search_size, para_path);
    head->ParameterInit(); 
}

Tracker::~Tracker() {
    if (model_path != nullptr) {
        delete [] model_path;
    }
}

void Tracker::VideoCaptureByFrame(const std::string &video_frame_path) {
    image = cv::imread(video_frame_path, cv::IMREAD_COLOR); // 读取彩色PNG图片
    if (image.empty()) {
        std::cerr << "Could not open or find the image" << std::endl;
    }
} 

cv::Mat Tracker::SampleTarget(const float &search_area_factor, const float &output_size, const std::string &search_type) {
    const float x = state[0];
    const float y = state[1];
    const float w = state[2];
    const float h = state[3];
    const float crop_size = std::ceil(std::sqrt(w * h) * search_area_factor);
    if (crop_size < 1.0) {
        std::cerr << "Too small bounding box." << std::endl;
    }
    // std::cout<<"**************"<<crop_size<<"\n";
    const int x1 = int(std::round(x + 0.5 * w - crop_size * 0.5));
    // int x1 = int(x + 0.5 * w - crop_size * 0.5);
    const int x2 = int(x1 + int(crop_size));
    const int y1 = int(std::round(y + 0.5 * h - crop_size * 0.5));
    // int y1 = int(y + 0.5 * h - crop_size * 0.5);
    const int y2 = int(y1 + int(crop_size));
    // std::cout<<"===============\n";
    // std::cout<<"x1 = "<<x1<<"x2 = "<<x2<<"y1 = "<<y1<<"y2 = "<<y2<<"\n";
    // std::cout<<"image.cols = "<<image.cols<<"images.rows = " <<image.rows;
    const int x1_pad = int(std::max(0, -x1));
    const int x2_pad = int(std::max(x2 - image.cols + 1, 0));
    const int y1_pad = int(std::max(0, -y1));
    const int y2_pad = int(std::max(y2 - image.rows + 1, 0));
    // std::cout<<"===============\n";
    // printf("%d,%d,%d,%d",x1_pad,x2_pad,y1_pad,y2_pad);
    const int width = (x2 - x2_pad) - (x1 + x1_pad);
    const int height = (y2 - y2_pad) - (y1 + y1_pad);
    cv::Rect roi(x1 + x1_pad, y1 + y1_pad, width, height);
    // std::cout<<"width = "<<width<<"height = "<<height<<"x_start ="<<x1 + x1_pad<<"y_start ="<<y1 + y1_pad<<"\n";
    cv::Mat sample_target_crop = image(roi).clone();
    // std::cout<< "height = "<<sample_target_crop.rows<<"width = "<<sample_target_crop.cols<<"channels = "<<sample_target_crop.channels()<<"\n";
    cv::Mat sample_target_tmp;
    cv::copyMakeBorder(sample_target_crop, sample_target_tmp, y1_pad, y2_pad, x1_pad, x2_pad, cv::BORDER_CONSTANT);
    current_resize_factor = output_size / crop_size;
    // if (search_type == "search_area") {
    //     std::cout << "current_resize_factor = " << current_resize_factor << std::endl;
    // }
    cv::Size new_size(output_size, output_size);
    cv::Mat sample_target;
    cv::resize(sample_target_tmp, sample_target, new_size);
    return sample_target;
}

std::vector<float> Tracker::Preprocess(const cv::Mat &sample_target) {
    cv::Mat rgb;
    cv::cvtColor(sample_target, rgb, cv::COLOR_BGR2RGB);
    cv::Mat float_img;
    rgb.convertTo(float_img, CV_32FC3, 1.0 / 255.0);
    cv::Mat channels[3];
    cv::split(float_img, channels);
    const int total_pixels = sample_target.cols * sample_target.rows;
    std::vector<float> normalized_NCHW; // NCHW
    normalized_NCHW.clear();
    normalized_NCHW.reserve(total_pixels * 3); // 预分配内存
    for (int c = 0; c < 3; c++) {
        // (x - mean) / std
        channels[c] = (channels[c] - mean[c]) / std[c];
        float* data = channels[c].ptr<float>();
        // copy to vector
        normalized_NCHW.insert(normalized_NCHW.end(), data, data + total_pixels);
    }
    return normalized_NCHW;
}

void Tracker::Runner(const int &frame_idx, const std::string &save_path, const std::string &file_save_path) {
    cv::Mat sample_target = SampleTarget(search_factor, search_size, "search_area");   
    std::vector<float> search_area = Preprocess(sample_target);
    /********************************************************************************/
    // model inference 
    // step 1. Set model inputs;
    sample_avp_tensor_t* img_t = GetTensorByTensorName(&mixformer_model, true, "img_t");
    int32_t ret = FillTensorFromBuf(img_t, (uint8_t *)(current_template.data()), current_template.size() * sizeof(float), OT_AVP_DTYPE_F32);
    sample_avp_tensor_t* img_ot = GetTensorByTensorName(&mixformer_model, true, "img_ot");
    ret = FillTensorFromBuf(img_ot, (uint8_t *)(online_template.data()), online_template.size() * sizeof(float), OT_AVP_DTYPE_F32);
    sample_avp_tensor_t* img_search = GetTensorByTensorName(&mixformer_model, true, "img_search");
    ret = FillTensorFromBuf(img_search, (uint8_t *)(search_area.data()), search_area.size() * sizeof(float), OT_AVP_DTYPE_F32);
    // step 2. Model execution;
    const uint32_t repeat_times = 1U;
    ret = sample_avp_execute_model(&mixformer_model, repeat_times);
    
    // step 3. Set outputs for postprocess.
    sample_avp_tensor_t *predict_info  = GetTensorByTensorName(&mixformer_model, false, "predict_info");
    head->GetBackboneInfo(predict_info);
    // /********************************************************************************************/
    const float current_pred_score = head->RunnerScore();
    std::cout << "current_pred_score = " << current_pred_score << std::endl;
    const float t = head->RunnerBoxes("T");
    // std::cout << "T=" << t << std::endl;
    const float l = head->RunnerBoxes("L");
    // std::cout << "L=" << l << std::endl;
    const float b = head->RunnerBoxes("B");
    // std::cout << "B=" << b << std::endl;
    const float r = head->RunnerBoxes("R");
    // std::cout << "R=" << r << std::endl;
    std::vector<float> bbox = head->TlbrToCxywh(t, l, b, r);
    for (auto i = 0; i < bbox.size(); i++) {
        bbox[i] = bbox[i] * float(search_size) / current_resize_factor;
    }
    //map back to the orignal image
    const float cx_prev = state[0] + 0.5 * state[2];
    const float cy_prev = state[1] + 0.5 * state[3];

    const float cx = bbox[0];
    const float cy = bbox[1];
    float w = bbox[2]; // width
    float h = bbox[3]; // height
    const float half_side = 0.5 * float(search_size) /  current_resize_factor;

    const float cx_real = cx + (cx_prev - half_side) - 0.5 * w; // top, left
    const float cy_real = cy + (cy_prev - half_side) - 0.5 * h; // top, left
    // clip the result.
    // top, left
    float x1 = cx_real;
    float y1 = cy_real;
    // botton, right;
    float x2 = cx_real + w;
    float y2 = cy_real + h;
    x1 = std::min(std::max(float(0.0), x1), float(WIDTH - MARGIN));
    x2 = std::min(std::max(float(MARGIN), x2), float(WIDTH));
    y1 = std::min(std::max(float(0.0), y1), float(HEIGHT - MARGIN));
    y2 = std::min(std::max(float(MARGIN), y2), float(HEIGHT));
    w = std::max(float(MARGIN), x2 - x1);
    h = std::max(float(MARGIN), y2 - y1);
    /********************************************************************************/
    // Update the new bbox info which is stored in state.
    state[0] = x1;
    state[1] = y1;
    state[2] = w;
    state[3] = h;
    std::cout << state[0] << ", " << state[1] << ", " << state[2] << ", " << state[3] << std::endl;
    MakeDir(file_save_path);
    // write the result to the text file
    out_file << int(std::round(state[0])) << " " << int(std::round(state[1])) << " " << int(std::round(state[2])) << " " << int(std::round(state[3])) << std::endl;
    /********************************************************************************/
    max_pred_score = max_pred_score * max_score_decay;
    if ((current_pred_score > 0.5) && (current_pred_score > max_pred_score)) {
        cv::Mat template_target = SampleTarget(template_factor, template_size, "template");
        online_max_template = Preprocess(template_target);
        online_template = online_max_template;
        max_pred_score = current_pred_score;
    }
    if (frame_idx % update_interval == 0) {
        if (online_size == 1) {
            online_template = online_max_template;
        }
        max_pred_score = -1.0;
        online_max_template = current_template;
    }
    /************************************************************************************/
    // plotting resluts
    const int thickness = 2;           // 2 pixels thick
    const int lineType = cv::LINE_8;   // 8-connected line
    // Draw the rectangle : top-left and bottom-right
    // cv::rectangle(image, cv::Point(int(x1), int(y1)), cv::Point(int(x1 + w), int(y1 + h)), cv::Scalar(0, 0, 255), thickness, lineType);
    cv::rectangle(image, cv::Point(int(x1), int(y1)), cv::Point(int(x1 + w), int(y1 + h)), cv::Scalar(0, 255, 0), thickness, lineType);
    MakeDir(save_path);
    std::string file_path = save_path + "/" + std::to_string(frame_idx) + ".png";
    cv::imwrite(file_path, image);
}

void Tracker::TrackerInfoInit(const std::string &first_image_path, const float *state_init, const std::string &image_save_path, const std::string &file_save_path, const std::string &file_name) {
    VideoCaptureByFrame(first_image_path);
    state[0] = state_init[0];
    state[1] = state_init[1];
    state[2] = state_init[2];
    state[3] = state_init[3];
    const int thickness = 2;           // 2 pixels thick
    const int lineType = cv::LINE_8;   // 8-connected line
    cv::rectangle(image, cv::Point(int(state[0]), int(state[1])), cv::Point(int(state[0] + state[2]), int(state[1] + state[3])), cv::Scalar(0, 255, 0), thickness, lineType);
    MakeDir(image_save_path);
    std::string file_path = image_save_path + "/" + "0.png";
    cv::imwrite(file_path, image);
    std::cout << state[0] << ", " << state[1] << ", " << state[2] << ", " << state[3] << std::endl;
    WIDTH = image.cols;
    HEIGHT = image.rows;
    cv::Mat sample_target = SampleTarget(template_factor, template_size, "template");
    current_template = Preprocess(sample_target);
    online_template = current_template;
    online_max_template = current_template;
    // save the statistic info
    MakeDir(file_save_path);
    current_file_name = file_name;
    std::string current_statistic_file = file_save_path + "/" + file_name;
    // out_file.open(current_statistic_file, std::ios::binary | std::ios::app | std::ios::in | std::ios::out); // add new records after the old ones
    out_file.open(current_statistic_file, std::ios::trunc); // add new records while cleaning the old ones.
    out_file << int(std::round(state[0])) << " " << int(std::round(state[1])) << " " << int(std::round(state[2])) << " " << int(std::round(state[3])) << std::endl;
}

void Tracker::InferModelInit(const std::string &ortm_model_path) {
    if (model_path != nullptr) {
        delete [] model_path;
    }
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
    if (ret != 0) {
        std::cout << "npu init fail." << std::endl;
        exit(1);
    } else {
        std::cout << "npu init success" << std::endl;
    }
    memset(&mixformer_model, 0, sizeof(sample_avp_model_task_info_t));
    if (ortm_model_path.empty()) {
        std::cout << "please checkout the model path." << std::endl;
        exit(1);
    }
    model_path = new char[ortm_model_path.length() + 1];
    std::strcpy(model_path, ortm_model_path.c_str());
    ret  = ot_avp_npu_load_model(model_path, &mixformer_model.handle);
    if (ret != 0) {
        std::cout << "load model fail." << std::endl;
        exit(1);
    } else {
        std::cout << "load model, model_path: " << ortm_model_path << std::endl;
    }
    sample_avp_create_datasets(mixformer_model.handle, &mixformer_model);
}

void Tracker::TrackerDestroy() {
    sample_avp_destroy_datasets(mixformer_model.handle, &mixformer_model);
    ot_avp_npu_unload_model(mixformer_model.handle);
    ot_avp_npu_deinit();
}

void exit_with_help()
{
	printf(
	"Usage: ./main -m model_path -v video_path -i sample_input_dir -s save_dir \n"
    "-m : the model path \n"
    "-v : the video path \n"
    "-s : save path for the final results. \n"
    "-p : parameter path mixformer head. \n"
    "-n : frame count for inference. \n"
	);
	exit(1);
}

int main(int argc, char **argv) {

    std::string video_path;
    std::string save_path;
    std::string para_path;
    std::string model_path;
    int frame_cnt;
    for (int i = 1; i < argc; i += 2) {
        if(argv[i][0] != '-') {
            std::cout << "Wrong input format." << std::endl;
            exit_with_help();
        }
        switch(argv[i][1]) {
            case 'm':
				model_path = std::string(argv[i + 1]);
				break;
            case 'v':
				video_path = std::string(argv[i + 1]);
				break;
            case 's':
				save_path = std::string(argv[i + 1]);
				break;
            case 'p':
				para_path = std::string(argv[i + 1]);
				break;
            case 'n':
                frame_cnt = atoi(argv[i + 1]);
                break;
            default:
				fprintf(stderr,"unknown inputs: -%c\n", argv[i][1]);
				exit_with_help();
				break;
        }
    }

    std::shared_ptr<Tracker> tracker = std::make_shared<Tracker>(para_path);
    tracker->InferModelInit(model_path);
    std::vector<std::string> dirs = GetSortedDirs(video_path);
    auto dirs_num = dirs.size();
    for (auto k = 0; k < dirs_num; k++) {
        auto x = dirs[k];
        // std::cout << x << std::endl;
        // continue;
        std::string frames_path = video_path + "/" + x;
        std::vector<std::string> imgs = GetSortedImgs(frames_path);
        float *state_init = &GOT10K_VAL_STATE[k][0];
        std::string first_image_path = frames_path + "/" + imgs[0];
        std::string current_img_save_path = save_path + "/" + x;
        std::string current_file_save_path = save_path;
        std::string file_name = x + ".txt";
        tracker->TrackerInfoInit(first_image_path, state_init, current_img_save_path, current_file_save_path, file_name);
        int frame_num = imgs.size();
        for (int i = 1; i < frame_num; i++) {
            std::string video_frame_path = frames_path + "/" + imgs[i];
            tracker->VideoCaptureByFrame(video_frame_path);
            tracker->Runner(i, current_img_save_path, current_file_save_path);
            if (i >= frame_cnt) break;
        }
        tracker->CloseStatisticfile();
        break;
    }
    tracker->TrackerDestroy();

    return 0;
}