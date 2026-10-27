#include <cstring>
#include <dirent.h>
#include <iostream>
#include <vector>
#include "tools.h"
#include "file_utils.h"

std::vector<std::string> getDirectoryFiles(const std::string& dirPath) {
    std::vector<std::string> filenames;
    DIR* dir = opendir(dirPath.c_str());  // 打开目录

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {  // 遍历目录项
        if (entry->d_type == DT_REG &&
            strcmp(entry->d_name, ".") != 0 &&
            strcmp(entry->d_name, "..") != 0) {
            filenames.emplace_back(entry->d_name);
        }
    }
    closedir(dir);
    return filenames;
}

std::string find_index(const std::string& file_name) {
    std::string result;
    bool found_digit = false;
    for (char c : file_name) {
        if (isdigit(c)) {
            result += c;
            found_digit = true;
        } else if (found_digit) {
            break;
        }
    }
    return result;
}

char* save_data_buf(const char* data_address, ot_avp_npu_shape dims, uint32_t stride, ot_avp_data_type dtype){
    (void)dtype;
    uint32_t loop_times = 1U;
    for (int32_t i = 0; i < ((int32_t)dims.dim_size - 1); i++) {
        loop_times *= (uint32_t)dims.dims[i];
    }
    uint32_t last_dim = (uint32_t)dims.dims[dims.dim_size - 1U];
    uint32_t out_size = loop_times*last_dim*4U;
    unsigned long line_size = last_dim*4U;

    char* safe_buffer = new char[out_size];

    for (uint32_t loop = 0U; loop < loop_times; loop++) {
        uint32_t src_offset  = loop * stride;
        uint32_t dest_offset = loop * (uint32_t)line_size;
        (void)memcpy(safe_buffer + dest_offset, data_address + src_offset, line_size);
    }
    return safe_buffer;
}
