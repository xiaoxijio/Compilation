#pragma once
#ifndef TOOLS_H
#define TOOLS_H

#include <vector>
#include <string>
#include "file_utils.h"

std::vector<std::string> getDirectoryFiles(const std::string& dirPath);
std::string find_index(const std::string& file_name);
char* save_data_buf(const char* data_address, ot_avp_npu_shape dims, uint32_t stride, ot_avp_data_type dtype);

#endif