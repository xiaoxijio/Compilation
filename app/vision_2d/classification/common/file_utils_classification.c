#include <file_utils_classification.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef __GNUC__
#include <unistd.h>
#endif

#define FILE_PATH_MAX (1024)

static ot_u32 get_file_line(const char* path);
/* LDRA_EXCLUDE 78 S */
#define WRITE_FILE_DATA(TYPE)                                                                   \
    static int32_t write_file_##TYPE##_data(const char* file_path, uint8_t* data_buf, uint32_t buf_len)    \
    {                                                                                       \
        FILE* file = fopen(file_path, "w");                                                 \
        if (file == NULL) {                                                                 \
            printf("file: %s open failed\n", file_path);                                    \
            return -1;                                                                      \
        }                                                                                   \
        uint32_t elem_len = buf_len / sizeof(TYPE);                                         \
        TYPE* buf = (TYPE*)data_buf;                                                        \
        for (uint32_t i = 0; i < elem_len; i++) {                                           \
            fprintf(file, "%d\n", buf[i]);                                                  \
        }                                                                                   \
        (void)fclose(file);                                                                       \
        return 0;                                                                           \
    }
/* LDRA_EXCLUDE 78 S */
#define READ_FILE_DATA(TYPE)                                                                             \
    static ot_s32 read_file_##TYPE##_data(const char* file_path, ot_u8* data_buf, ot_avp_npu_shape dims, \
                                          ot_u32 stride, ot_u32 data_size)                               \
    {                                                                                                    \
        FILE* file = fopen(file_path, "r");                                                              \
        if (file == NULL) {                                                                              \
            printf("file: %s open failed\n", file_path);                                                 \
            return -1;                                                                                   \
        }                                                                                                \
        ot_u32 file_elem_size = get_file_line(file_path);                                                \
        TYPE*  file_data      = (TYPE*)malloc(sizeof(TYPE) * file_elem_size);                            \
        if (file_data == NULL) {                                                                         \
            (void)fclose(file);                                                                          \
            return -1;                                                                                   \
        }                                                                                                \
        char buffer[50] = {0};                                                                           \
        int  index      = 0;                                                                             \
        while (fgets(buffer, 50, file) != NULL) {                                                        \
            file_data[index] = (TYPE)atof(buffer);                                                       \
            index++;                                                                                     \
        }                                                                                                \
        ot_u32 loop_times = 1;                                                                           \
        for (ot_u32 i = 0U; i < (dims.dim_size - 1U); i++) {                                             \
            loop_times *= (ot_u32)dims.dims[i];                                                          \
        }                                                                                                \
        ot_u32 last_dims = (ot_u32)dims.dims[dims.dim_size - 1U];                                        \
        ot_u32 line_size = last_dims * data_size;                                                        \
        ot_u8* file_buf  = (ot_u8*)file_data;                                                            \
        for (ot_u32 i = 0U; i < loop_times; i++) {                                                       \
            ot_u32 src_offset  = i * line_size;                                                          \
            ot_u32 dest_offset = i * stride;                                                             \
            (void)memcpy(data_buf + dest_offset, file_buf + src_offset, line_size);                      \
        }                                                                                                \
        free(file_data);                                                                                 \
        (void)fclose(file);                                                                                    \
        return 0;                                                                                        \
    }

WRITE_FILE_DATA(uint8_t)
WRITE_FILE_DATA(int8_t)
WRITE_FILE_DATA(uint16_t)
WRITE_FILE_DATA(int16_t)
WRITE_FILE_DATA(uint32_t)
WRITE_FILE_DATA(int32_t)

READ_FILE_DATA(uint8_t)
READ_FILE_DATA(int8_t)
READ_FILE_DATA(int16_t)
READ_FILE_DATA(uint16_t)
READ_FILE_DATA(int32_t)
READ_FILE_DATA(uint32_t)
/* LDRA_EXCLUDE 445 S */
/* LDRA_EXCLUDE 635 S */
READ_FILE_DATA(float)

ot_u32 get_dtype_size(ot_avp_data_type dtype)
{
    switch (dtype) {
        case OT_AVP_DTYPE_INT8:
            return 1;
        case OT_AVP_DTYPE_INT16:
            return 2;
        case OT_AVP_DTYPE_INT32:
            return 4;
        case OT_AVP_DTYPE_INT64:
            return 8;
        case OT_AVP_DTYPE_UINT8:
            return 1;
        case OT_AVP_DTYPE_UINT16:
            return 2;
        case OT_AVP_DTYPE_UINT32:
            return 4;
        case OT_AVP_DTYPE_UINT64:
            return 8;
        case OT_AVP_DTYPE_F32:
            return 4;
        case OT_AVP_DTYPE_F64:
            return 8;
        case OT_AVP_DTYPE_BOOL:
            return 1;
        default:
            return 0;
    }
}

score_with_ind* sort_top_five(score_with_ind* scores, int class)
{
    for(int i = 0; i < 5; i++){
        int max_index = i;
        for(int j = i + 1; j < class; j++) {
            if(scores[max_index].score < scores[j].score) {
                max_index = j;
            }
        }
        score_with_ind change_score = scores[i];
        scores[i].score = scores[max_index].score;
        scores[i].ind = scores[max_index].ind;
        scores[max_index].score = change_score.score;
        scores[max_index].ind = change_score.ind;
    }
    return scores;
}

static const char* filename_suffix(const char* file_path, char split)
{
    int32_t len = (int32_t)strlen(file_path);
    for (int32_t i = len-1; i >= 0; i--) {
        if (file_path[i] == split) {
            return file_path + i + 1;
        }
    }
    return NULL;
}

static ot_u32 get_bin_size(const char* path)
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
    return (ot_u32)size;
}
 /* LDRA_EXCLUDE_START 652 S */

static ot_u32 read_file_binary_data_with_stride(const char* file_path, ot_u8* buf, ot_avp_npu_shape dims, ot_u32 stride, ot_u32 data_size)
{
    ot_u32 size = get_bin_size(file_path);
    FILE *infile = fopen(file_path, "rb");
    if (infile == NULL) {
        printf("can't open path: %s", file_path);
        return -1;
    }
    (void)fseek(infile, 0, SEEK_SET);
    ot_u8* data = (ot_u8*)malloc(size);
    if(data == OT_NULL){
        printf("malloc failed\n");
        (void)fclose(infile);
        return -1;
    }
    (void)memset(data, 0, size);
/* LDRA_EXCLUDE 72 D */
    ot_ulong read_size = fread(data, 1, size, infile);
    if (read_size != size) {
        (void)fclose(infile);
        if(data != NULL){
            free(data);
        }
        printf("read size error");
        return -1;
    }
    (void)fclose(infile);
    if(data == NULL){
        return -1;
    }
    ot_ulong loop_times = 1;
    for (ot_s32 loop = 0; loop < (dims.dim_size - 1); loop++) {
        loop_times *= dims.dims[loop];
    }
    ot_ulong dim_val = dims.dims[dims.dim_size - 1];
    ot_ulong line_size = dim_val * data_size;
    for (ot_ulong loop = 0; loop < loop_times; loop++) {
        ot_ulong srcoffset = loop * line_size;
        ot_ulong dstoffset = loop * stride;
        (void)memcpy(buf + dstoffset, data + srcoffset, line_size);  
    }
    free(data);
    return 0;
}
/* LDRA_EXCLUDE_END  652 S */

path_structs* read_file_string_data(const char* file_path)
{
    FILE *file = fopen(file_path, "r");
    if (file == NULL) {
        printf("file: %s open failed\n", file_path);
        return NULL;
    }
    path_structs *head = (path_structs*)malloc(sizeof(path_structs));
    path_structs *temp ;
    path_structs *cur = head;
    char buffer[256] = {0};
    int size = 0;
    while (fscanf(file,"%[^\n] ", buffer)!= EOF) {
        temp = (path_structs *)malloc(sizeof(path_structs));
        char* buffer_temp = (char *) malloc(256U * sizeof(char));
        (void)memcpy(buffer_temp, buffer, 256U*sizeof(char));
        temp->data = buffer_temp;
        cur->next = temp;
        cur = cur->next;
        size++;
    }
    (void)fclose(file);
    cur->next = NULL;
    head->next->size = size;
    return head->next;
}

label_structs* read_file_label_data(const char* file_path)
{
    FILE *file = fopen(file_path, "r");
    if (file == NULL) {
        printf("file: %s open failed\n", file_path);
        return NULL;
    }
    label_structs *head = (label_structs*)malloc(sizeof(label_structs));
    label_structs *temp ;
    label_structs *cur = head;
    int buffer[1] = {0};
    int size = 0;
    while (fscanf(file,"%d", buffer)!= EOF) {
        temp = (label_structs *)malloc(sizeof(label_structs));
        temp->data = buffer[0];
        cur->next = temp;
        cur = cur->next;
        size++;
    }
    (void)fclose(file);
    cur->next = NULL;
    head->next->size = size;
    return head->next;
}

static ot_u32 get_file_line(const char* path)
{
    ot_u32 size = 0;
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        printf("can't open path: %s", path);
        return 0;
    }
    int ch = fgetc(file);
    while (ch != EOF) {
        if (ch == (int)'\n') {
            size++;
        }
        ch = fgetc(file);
    }
    (void)fclose(file);
    return size;
}

static int32_t write_file_float32_data(const char* file_path, uint8_t* data_buf, uint32_t buf_len)
{
    FILE* file = fopen(file_path, "w");
    if (file == NULL) {
        printf("file: %s open failed\n", file_path);
        return -1;
    }
    uint32_t elem_len = buf_len / sizeof(float);
    float* buf = (float*)data_buf;
    for (uint32_t i = 0; i < elem_len; i++) {
        fprintf(file, "%.13f\n", buf[i]);
    }
    (void)fclose(file);
    return 0;
}


ot_s32 load_data_from_file_classification(const char* file_path, ot_u8* data_buf, ot_avp_npu_shape dims, ot_ulong stride, ot_avp_data_type dtype)
{
    const char* suffix = filename_suffix(file_path, '.');
    if (suffix == NULL) {
        printf("invalid file suffix\n");
        return -1;
    }
    ot_s32 ret = 0;
    ot_u32 data_size = get_dtype_size(dtype);
    if (data_size == 0) {
        printf("invalid dtype: %d", dtype);
        return -1;
    }
    if (strcmp(suffix, "bin") == 0) {
        ret = read_file_binary_data_with_stride(file_path, data_buf, dims, stride, data_size);
    } else if (strcmp(suffix, "txt") == 0) {
        switch (dtype) {
            case OT_AVP_DTYPE_F32: {
                ret = read_file_float_data(file_path, data_buf, dims, stride, data_size);
                break;
            }
            case OT_AVP_DTYPE_UINT8: {
                ret = read_file_uint8_t_data(file_path, data_buf, dims, stride, data_size);
                break;
            }
            case OT_AVP_DTYPE_INT8: {
                ret = read_file_int8_t_data(file_path, data_buf, dims, stride, data_size);
                break;
            }
            case OT_AVP_DTYPE_UINT16: {
                ret = read_file_uint16_t_data(file_path, data_buf, dims, stride, data_size);
                break;
            }
            case OT_AVP_DTYPE_INT16: {
                ret = read_file_int16_t_data(file_path, data_buf, dims, stride, data_size);
                break;
            }
            case OT_AVP_DTYPE_UINT32: {
                ret = read_file_uint32_t_data(file_path, data_buf, dims, stride, data_size);
                break;
            }
            case OT_AVP_DTYPE_INT32: {
                ret = read_file_int32_t_data(file_path, data_buf, dims, stride, data_size);
                break;
            }
            case OT_AVP_DTYPE_BOOL: {
                ret = read_file_uint8_t_data(file_path, data_buf, dims, stride, data_size);
                break;
            }
            default:
                printf("unsupport dtype %d\n", dtype);
                return -1;
        }
    } else {
        printf("invalid file suffix\n");
        return -1;
    }
    return ret;
}

int32_t dump_data_to_binary_file_classification(const char* file_path, uint8_t* data_buf, uint32_t buf_len, ot_avp_data_type dtype)
{
    switch (dtype) {
        case OT_AVP_DTYPE_F32:
            return write_file_float32_data(file_path, data_buf, buf_len);
        case OT_AVP_DTYPE_UINT8:
            return write_file_uint8_t_data(file_path, data_buf, buf_len);
        case OT_AVP_DTYPE_INT8:
            return write_file_int8_t_data(file_path, data_buf, buf_len);
        case OT_AVP_DTYPE_UINT16:
            return write_file_uint16_t_data(file_path, data_buf, buf_len);
        case OT_AVP_DTYPE_INT16:
            return write_file_int16_t_data(file_path, data_buf, buf_len);
        case OT_AVP_DTYPE_UINT32:
            return write_file_uint32_t_data(file_path, data_buf, buf_len);
        case OT_AVP_DTYPE_INT32:
            return write_file_int32_t_data(file_path, data_buf, buf_len);
        case OT_AVP_DTYPE_BOOL:
            return write_file_uint8_t_data(file_path, data_buf, buf_len);
        default:
            printf("not support dtype\n");
            return -1;
    }
    return 0;
}
