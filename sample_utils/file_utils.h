#ifndef __SAMPLE_UTILS_FILE_UTILS_H
#define __SAMPLE_UTILS_FILE_UTILS_H
#include "ot_avp_npu_common.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

#if defined(__linux__)
#define PATH_SEPARATOR '/'
#elif defined(_WIN32) || defined(_WIN64)
#define PATH_SEPARATOR '\\'
#define S_ISDIR(m) (((m) & 0170000) == (0040000))
#endif

#define FILE_PATH_MAX (384)
#define FILE_NAME_MAX (128)

int32_t load_data_from_file(const char* file_path, uint8_t* data_buf, ot_avp_npu_shape dims, unsigned int stride, ot_avp_data_type dtype);

int32_t dump_data_to_binary_file(const char* output_file_path, uint8_t* data_buf, ot_avp_npu_shape dims, unsigned int stride, ot_avp_data_type dtype);
int32_t dump_data_to_txt_file(const char* output_file_path, uint8_t* data_buf, ot_avp_npu_shape dims, unsigned int stride, ot_avp_data_type dtype);

int32_t get_input_file(const char* input_path, char** input_file, int input_num);
int32_t get_bin_size(const char* path);

void mkdir_dir(char* dir_path);

void get_file_name(const char* file_path, char* file_name);

bool is_directory(char* file_path);
int remove_bubbles_data(uint8_t* data_buf, uint32_t* data_buf_size, ot_avp_npu_shape dims, unsigned int stride, ot_avp_data_type dtype);
void gen_win_path(const char* src, char* dst, int dst_len);

int32_t sample_avp_npu_read_bin_file(const char* file_name, void* buf, int32_t buf_len);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
