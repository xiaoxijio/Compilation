#ifndef __FILE_UTILS_CLASS_H
#define __FILE_UTILS_CLASS_H
#include <string.h>
#include "ot_avp_npu_rts.h"
#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

typedef struct path_structs{
    char *data;
    struct path_structs *next;
    int size;
    /* data */
}path_structs;

typedef struct label_structs{
    int data;
    struct label_structs *next;
    int size;
    /* data */
}label_structs;

typedef struct {
    uint8_t* buf;
    uint32_t buf_len;
} data_buf;

typedef struct {
    float score;
    int32_t ind;
} score_with_ind;

ot_u32 get_dtype_size(ot_avp_data_type dtype);

ot_s32 load_data_from_file_classification(const char* file_path, ot_u8* data_buf, ot_avp_npu_shape dims, ot_ulong stride, ot_avp_data_type dtype);

int32_t dump_data_to_binary_file_classification(const char* file_path, uint8_t* data_buf, uint32_t buf_len, ot_avp_data_type dtype);

path_structs* read_file_string_data(const char* file_path);

label_structs* read_file_label_data(const char* file_path);

score_with_ind* sort_top_five(score_with_ind* scores, int class_number);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif