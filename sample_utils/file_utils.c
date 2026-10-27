#include "file_utils.h"
#include "ot_avp_tensor_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <math.h>
#if defined(__linux__)
#include <unistd.h>
#include <dirent.h>
#elif defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <direct.h>
#include <io.h>
#endif

static uint32_t get_size_by_dtype(uint32_t elem_size, ot_avp_data_type dtype);
static float data_convert_fp16_to_fp32(uint16_t in);
/* LDRA_EXCLUDE 78 S */
#define WRITE_TXT_FILE_DATA(TYPE)                                                                                \
    static int32_t write_txt_file_##TYPE##_data(const char* file_path, uint8_t* data_buf, ot_avp_npu_shape dims, \
                                            uint32_t stride, ot_avp_data_type dtype)                         \
    {                                                                                                        \
        if (data_buf == NULL) {                                                                              \
            printf("data_buf is null\n");                                                                    \
            return -1;                                                                                       \
        }                                                                                                    \
        FILE* file = fopen(file_path, "w");                                                                  \
        if (file == NULL) {                                                                                  \
            printf("file: %s open failed\n", file_path);                                                     \
            return -1;                                                                                       \
        }                                                                                                    \
        uint32_t loop_times = 1U;                                                                            \
        for (int32_t i = 0; i < ((int32_t)dims.dim_size - 1); i++) {                                         \
            loop_times *= (uint32_t)dims.dims[i];                                                            \
        }                                                                                                    \
        uint32_t last_dim = (uint32_t)dims.dims[dims.dim_size - 1U];                                         \
        uint32_t out_size = get_size_by_dtype(loop_times * last_dim, dtype);                                 \
        uint8_t* out_data = (uint8_t*)malloc(out_size);                                                      \
        if (out_data == NULL) {                                                                              \
            printf("malloc failed\n");                                                                       \
            return -1;                                                                                       \
        }                                                                                                    \
        unsigned long line_size = get_size_by_dtype(last_dim, dtype);                                        \
        for (uint32_t loop = 0; loop < loop_times; loop++) {                                                 \
            uint32_t src_offset  = loop * stride;                                                            \
            uint32_t dest_offset = loop * (uint32_t)line_size;                                               \
            (void)memcpy(out_data + dest_offset, data_buf + src_offset, line_size);                          \
        }                                                                                                    \
        TYPE*    type_buf  = (TYPE*)out_data;                                                                \
        uint32_t elem_size = get_size_by_dtype(1U, dtype);                                                   \
        if (elem_size != 0U) {                                                                               \
            elem_size = out_size / elem_size;                                                                \
        }                                                                                                    \
        float* float_out = (float*)malloc(elem_size * sizeof(float));                                        \
        for (uint32_t i = 0; i < elem_size; i++) {                                                           \
            if (dtype == OT_AVP_DTYPE_F16) {                                                                 \
                float_out[i] = data_convert_fp16_to_fp32(type_buf[i]);                                       \
            } else {                                                                                         \
                float_out[i] = (float)type_buf[i];                                                           \
            }                                                                                                \
            (void)fprintf(file, "%.6f\n", float_out[i]);                                                     \
        }                                                                                                    \
        free(out_data);                                                                                      \
        (void)fclose(file);                                                                                  \
        free(float_out);                                                                                     \
        return 0;                                                                                            \
    }
/* LDRA_EXCLUDE_START 140 D */
WRITE_TXT_FILE_DATA(uint8_t)
WRITE_TXT_FILE_DATA(int8_t)
WRITE_TXT_FILE_DATA(uint16_t)
WRITE_TXT_FILE_DATA(int16_t)
WRITE_TXT_FILE_DATA(uint32_t)
WRITE_TXT_FILE_DATA(int32_t)
WRITE_TXT_FILE_DATA(uint64_t)
WRITE_TXT_FILE_DATA(int64_t)
WRITE_TXT_FILE_DATA(float)
/* LDRA_EXCLUDE_END  140 D */

static int32_t write_file_data_buf(const char* file_path, uint8_t* data_buf, ot_avp_npu_shape dims, uint32_t stride, ot_avp_data_type dtype)
{
    FILE* file = fopen(file_path, "w");
    if (file == NULL) {
        printf("file: %s open failed\n", file_path);
        return -1;
    }
    uint32_t loop_times = 1U;
    for (int32_t i = 0; i < ((int32_t)dims.dim_size - 1); i++) {
        loop_times *= (uint32_t)dims.dims[i];
    }
    uint32_t last_dim = (uint32_t)dims.dims[dims.dim_size - 1U];
    uint32_t out_size = get_size_by_dtype(loop_times*last_dim, dtype);
    uint8_t* out_data = (uint8_t*)malloc(out_size);
    unsigned long line_size = get_size_by_dtype(last_dim, dtype);
    for (uint32_t loop = 0; loop < loop_times; loop++) {
        uint32_t src_offset  = loop * stride;
        uint32_t dest_offset = loop * (uint32_t)line_size;
        (void)memcpy(out_data + dest_offset, data_buf + src_offset, line_size);
    }
    (void)fwrite(out_data, sizeof(uint8_t), out_size, file);
    free(out_data);
    (void)fclose(file);
    return 0;
}

static float data_convert_fp16_to_fp32(uint16_t in)
{
    float out = 0.0f;

    uint32_t t1 = (uint32_t)in & 0x7fffU;                // Non-sign bits
    uint32_t t2 = (uint32_t)in & 0x8000U;                // Sign bit
    uint32_t t3 = (uint32_t)in & 0x7c00U;                // Exponent

    t1 <<= 13;                              // Align mantissa on MSB
    t2 <<= 16;                              // Shift sign bit into position

    t1 += 0x38000000U;                       // Adjust bias

    t1 = (t3 == 0U ? 0U : t1);              // Denormals-as-zero

    t1 |= t2;                               // Re-insert sign bit
 /* LDRA_EXCLUDE 140 D */
 /* LDRA_EXCLUDE 655 S */
    (void)memcpy(&out, &t1, sizeof(out));
    return out;
}

static uint32_t get_dtype_bits(ot_avp_data_type dtype)
{
    switch (dtype) {
        case OT_AVP_DTYPE_INT8:
            return 8U;
        case OT_AVP_DTYPE_INT16:
            return 16U;
        case OT_AVP_DTYPE_INT32:
            return 32U;
        case OT_AVP_DTYPE_INT64:
            return 64U;
        case OT_AVP_DTYPE_UINT8:
            return 8U;
        case OT_AVP_DTYPE_UINT16:
            return 16U;
        case OT_AVP_DTYPE_UINT32:
            return 32U;
        case OT_AVP_DTYPE_UINT64:
            return 64U;
        case OT_AVP_DTYPE_F32:
            return 32U;
        case OT_AVP_DTYPE_F64:
            return 64U;
        case OT_AVP_DTYPE_BOOL:
            return 8U;
        case OT_AVP_DTYPE_INT10:
            return 10U;
        case OT_AVP_DTYPE_UINT10:
            return 10U;
        case OT_AVP_DTYPE_INT12:
            return 12U;
        case OT_AVP_DTYPE_UINT12:
            return 12U;
        case OT_AVP_DTYPE_F16:
            return 16U;
        default:
            return 0U;
    }
}

static uint32_t get_size_by_dtype(uint32_t elem_size, ot_avp_data_type dtype)
{
    uint32_t data_bits = get_dtype_bits(dtype);
    return (uint32_t)ceil(1.0*(double)elem_size*(double)data_bits / 8.0);
}


static const char* filename_suffix(const char* file_path, char split)
{
    int len = (int)strlen(file_path);
    for (int i = len-1; i >= 0; i--) {
        if (file_path[i] == split) {
            return file_path + i + 1;
        }
    }
    return NULL;
}

int32_t get_bin_size(const char* path)
{
    long  size = 0;
    FILE *fp = fopen(path, "rb");
    if (fp != NULL) {
        (void)fseek(fp, 0, SEEK_END);
        size = ftell(fp);
        if (size == -1L) {
            // ftell failed
            perror("ftell failed");
        }
        (void)fclose(fp);
    }
    return (int32_t)size;
}
 /* LDRA_EXCLUDE_START 652 S */
static int32_t read_file_data_with_stride(const char* file_path, uint8_t* buf, ot_avp_npu_shape dims, uint32_t stride, ot_avp_data_type dtype)
{
    if (buf == NULL) {
        printf("buf is null\n");
        return -1;
    }
    int32_t size = get_bin_size(file_path);
    FILE *infile = fopen(file_path, "rb");
    if (infile == NULL) {
        printf("can't open path: %s", file_path);
        return -1;
    }
    (void)fseek(infile, 0, SEEK_SET);
    uint8_t* data = (uint8_t*)malloc(size);
    if (data == NULL) {
        printf("malloc failed\n");
        return -1;
    }
    (void)memset(data, 0 , size);
    size_t read_size = fread(data, 1, size, infile);
    if (read_size != (size_t)size) {
        free(data);
        (void)fclose(infile);
        printf("read size error");
        return -1;
    }
    (void)fclose(infile);
    if (data == NULL) {
        return -1;
    }
    unsigned long loop_times = 1;
    for (int32_t loop = 0; loop < ((int32_t)dims.dim_size - 1); loop++) {
        loop_times *= (uint32_t)dims.dims[loop];
    }

    unsigned long dim_val = (uint32_t)dims.dims[dims.dim_size - 1U];
    uint32_t line_size = get_size_by_dtype((uint32_t)dim_val, dtype);
    for (uint32_t loop = 0; loop < loop_times; loop++) {
        if (((loop * line_size) + line_size) > (uint32_t)size) {
            printf("file data is not enough to fill buff.\n");
            free(data);
            return -1;
        }
        uint32_t src_offset  = loop * line_size;
        uint32_t dest_offset = loop * stride;
        (void)memcpy(buf + dest_offset, data + src_offset, line_size);
    }
    free(data);
    return 0;
}
/* LDRA_EXCLUDE_END  652 S */

int32_t load_data_from_file(const char* file_path, uint8_t* data_buf, ot_avp_npu_shape dims, unsigned int stride, ot_avp_data_type dtype)
{
    const char* suffix = filename_suffix(file_path, '.');
    if (suffix == NULL) {
        printf("invalid file suffix\n");
        return -1;
    }
    int32_t ret = 0;
    uint32_t data_bits = get_dtype_bits(dtype);
    if (data_bits <= 0U) {
        printf("invalid dtype: %d", dtype);
        return -1;
    }
    if (strcmp(suffix, "txt") == 0) {
        printf("do not support xxx.txt input\n");
        return -1;
    } else {
        ret = read_file_data_with_stride(file_path, data_buf, dims, stride, dtype);
    }
    return ret;
}

int remove_bubbles_data(uint8_t* data_buf, uint32_t* data_buf_size, ot_avp_npu_shape dims, unsigned int stride, ot_avp_data_type dtype)
{
    uint32_t data_bits = get_dtype_bits(dtype);
    if (data_bits <= 0U) {
        printf("invalid dtype: %d", dtype);
        return -1;
    }

    uint32_t loop_times = 1U;
    for (int32_t i = 0; i < ((int32_t)dims.dim_size - 1); i++) {
        loop_times *= (uint32_t)dims.dims[i];
    }
    uint32_t last_dim = (uint32_t)dims.dims[dims.dim_size - 1U];
    uint32_t out_size = get_size_by_dtype(loop_times*last_dim, dtype);

    if (*data_buf_size == out_size) {
        return 0;
    }

    uint8_t* out_data = data_buf;
    unsigned long line_size = get_size_by_dtype(last_dim, dtype);
    for (uint32_t loop = 0; loop < loop_times; loop++) {
        uint32_t src_offset  = loop * stride;
        uint32_t dest_offset = loop * (uint32_t)line_size;
        (void)memcpy(out_data + dest_offset, data_buf + src_offset, line_size);
    }
    // printf("remove_bubbles_data: src_size[%u], dst_size[%u]\n", *data_buf_size, out_size);
    *data_buf_size = out_size;

    return 0;
}

int32_t dump_data_to_binary_file(const char* output_file_path, uint8_t* data_buf, ot_avp_npu_shape dims, unsigned int stride, ot_avp_data_type dtype)
{
    return write_file_data_buf(output_file_path, data_buf, dims, stride, dtype);
}

int32_t dump_data_to_txt_file(const char* output_file_path, uint8_t* data_buf, ot_avp_npu_shape dims, unsigned int stride, ot_avp_data_type dtype)
{
    uint32_t data_bits = get_dtype_bits(dtype);
    if (data_bits <= 0U) {
        printf("invalid dtype: %d", dtype);
        return -1;
    }
    switch (dtype) {
        case OT_AVP_DTYPE_F32:
            return write_txt_file_float_data(output_file_path, data_buf, dims, stride, dtype);
        case OT_AVP_DTYPE_UINT8:
            return write_txt_file_uint8_t_data(output_file_path, data_buf, dims, stride, dtype);
        case OT_AVP_DTYPE_INT8:
            return write_txt_file_int8_t_data(output_file_path, data_buf, dims, stride, dtype);
        case OT_AVP_DTYPE_UINT16:
            return write_txt_file_uint16_t_data(output_file_path, data_buf, dims, stride, dtype);
        case OT_AVP_DTYPE_INT16:
            return write_txt_file_int16_t_data(output_file_path, data_buf, dims, stride, dtype);
        case OT_AVP_DTYPE_UINT32:
            return write_txt_file_uint32_t_data(output_file_path, data_buf, dims, stride, dtype);
        case OT_AVP_DTYPE_INT32:
            return write_txt_file_int32_t_data(output_file_path, data_buf, dims, stride, dtype);
        case OT_AVP_DTYPE_UINT64:
            return write_txt_file_uint64_t_data(output_file_path, data_buf, dims, stride, dtype);
        case OT_AVP_DTYPE_INT64:
            return write_txt_file_int64_t_data(output_file_path, data_buf, dims, stride, dtype);
        case OT_AVP_DTYPE_BOOL:
            return write_txt_file_uint8_t_data(output_file_path, data_buf, dims, stride, dtype);
        case OT_AVP_DTYPE_UINT10:
            return write_file_data_buf(output_file_path, data_buf, dims, stride, dtype);
        case OT_AVP_DTYPE_UINT12:
            return write_file_data_buf(output_file_path, data_buf, dims, stride, dtype);
        case OT_AVP_DTYPE_F16:
            return write_txt_file_uint16_t_data(output_file_path, data_buf, dims, stride, dtype);
        default:
            printf("unsupport dtype %d\n", dtype);
            return -1;
    }
    return 0;
}


int32_t get_input_file(const char* input_path, char** input_file, int input_num)
{
    struct stat st;
    if (stat(input_path, &st) != 0) {
        printf("input path is not exit, input path:%s\n", input_path);
        return -1;
    }

    if (S_ISDIR(st.st_mode)) {
        // directory
        int file_num = 0;
#ifdef __linux__
        DIR *dir = opendir(input_path);
        if (dir == NULL) {
            printf("open directory failed\n");
            return -1;
        }
        struct dirent *entry;
        // while ((entry = readdir(dir)) != NULL) {
        for (entry = readdir(dir); entry != NULL; entry = readdir(dir)) {
            if (strcmp(entry->d_name, ".") == 0 ||
                    strcmp(entry->d_name, "..") == 0) {
                continue;
            }
            char file_name[FILE_NAME_MAX] = {0};
            get_file_name(entry->d_name, file_name);
            int input_index = atoi(file_name);
            if (input_index >= input_num) {
                printf("file name [%d] should < input_num[%d]\n", input_index, input_num);
                return -1;
            }
/* LDRA_EXCLUDE 589 S */
            int ret = snprintf(input_file[input_index], FILE_PATH_MAX, "%s/%s", input_path, entry->d_name);
            if (ret < 0) {
                printf("the length of input file path is too long, max length: [%d]\n", FILE_PATH_MAX);
                return -1;
            }
            file_num++;
        }
        closedir(dir);
#else
        char tmp_path[FILE_PATH_MAX] = "";
        gen_win_path(input_path, tmp_path, FILE_PATH_MAX);
        WIN32_FIND_DATA fileInfo;
        HANDLE hFind = FindFirstFile(tmp_path, &fileInfo);
        if (INVALID_HANDLE_VALUE == hFind) {
            printf("failed to find files in this directory:%s\n", tmp_path);
            return -1;
        }
        while (FindNextFile(hFind, &fileInfo) != 0) {
            if (strcmp(fileInfo.cFileName, ".") == 0 ||
                strcmp(fileInfo.cFileName, "..") == 0) {
                continue;
            }
            char file_name[FILE_NAME_MAX] = {0};
            get_file_name(fileInfo.cFileName, file_name);
            int input_index = atoi(file_name);
            if (input_index >= input_num) {
                printf("file name [%d] should < input_num[%d]\n", input_index, input_num);
                return -1;
            }
            int ret = snprintf(input_file[input_index], FILE_PATH_MAX, "%s%c%s", input_path, PATH_SEPARATOR, fileInfo.cFileName);
            if (ret < 0) {
                printf("the length of input file path is too long, max length: [%d]\n", FILE_PATH_MAX);
                return -1;
            }
            file_num++;
        }
        FindClose(hFind);
#endif
        if (file_num != input_num) {
            printf("input file num [%d] is not match model input num [%d]\n", file_num, input_num);
            return -1;
        }
    } else {
        // input file
        if (input_num != 1) {
            printf("multi-input model only support directory as -i parameter\n");
            return -1;
        }
        (void)strncpy(input_file[0], input_path, FILE_PATH_MAX);
        if (stat(input_file[0], &st) != 0) {
            printf("file %s is not exit\n", input_file[0]);
            return -1;
        }
        return 0;
    }
    return 0;
}

void mkdir_dir(char* dir_path)
{
    char command[FILE_PATH_MAX] = {0};
#ifdef _MSC_VER
    snprintf(command, FILE_PATH_MAX, "rd /s /q %s", dir_path);
    if (0 == _access(dir_path, 0)) {
        system(command);
        if (_mkdir(dir_path) != 0) {
            printf("create directory:%s failed\n", dir_path);
            return;
        }
    } else {
        if (_mkdir(dir_path) != 0) {
            printf("create directory:%s failed\n", dir_path);
            return;
        }
    }
#else
    int ret = snprintf(command, FILE_PATH_MAX, "rm -rf %s", dir_path);
    if (ret < 0) {
        printf("mkdir directory fail, dir path is too long\n");
        return;
    }
    /* LDRA_EXCLUDE 588 S */
    ret = system(command);
    if(ret == -1){
        printf("command: \'%s\' fail\n", command);
    }
    /* LDRA_EXCLUDE 83 S */
    if (mkdir(dir_path, 0755) != 0) {
        printf("create directory:%s failed", dir_path);
    }
#endif
}

void get_file_name(const char* file_path, char* file_name)
{
    // dir/dir/file_name.bin  -> file_name
    char* pos = strrchr(file_path, PATH_SEPARATOR);
    if (pos == NULL) {
        (void)memcpy(file_name, file_path, FILE_NAME_MAX - 1);
    } else {
        (void)strncpy(file_name, pos+1, FILE_NAME_MAX - 1);
    }
    file_name[FILE_NAME_MAX - 1] = '\0';
    char* suffix_pos = strrchr(file_name, '.');
    if (suffix_pos != NULL) {
        *suffix_pos = '\0';
    }
}

bool is_directory(char* file_path)
{
    struct stat st;
    if (stat(file_path, &st) != 0) {
        printf("input path is not exit, input path:%s\n", file_path);
        return false;
    }
    if (S_ISDIR(st.st_mode)) {
        return true;
    }
    return false;
}

void gen_win_path(const char* src, char* dst, int dst_len)
{
    if (src != NULL) {
        (void)strncpy(dst, src, dst_len);
        size_t len   = strlen(dst);
        dst[len]     = '\\';
        dst[len + 1U] = '*';
    }

    return;
}


int32_t sample_avp_npu_read_bin_file(const char* file_name, void* buf, int32_t buf_len)
{
    FILE* file = fopen(file_name, "rb");  // 以二进制读模式打开
    if (file == NULL) {
        printf("无法打开文件");
        return -1;
    }
    int32_t ret = -1;
    // 获取文件大小
    fseek(file, 0, SEEK_END);
    int32_t size = ftell(file);
    fseek(file, 0, SEEK_SET);

    if (size != buf_len) {
        printf("file size is smaller than buf_len.\n");
        goto fail;
    }

    // 读取文件内容
    int32_t bytes_read = fread(buf, 1, size, file);
    if (bytes_read != size) {
        printf("读取文件失败\n");
        goto fail;
    }
    ret = 0;
fail:
    fclose(file);
    return ret;
}
