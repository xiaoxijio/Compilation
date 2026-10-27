#include "sample_npu_common.h"
#ifndef SIMULATOR
#include "ot_smr.h"
#endif
#include <stdio.h>


int ot_avp_create_tensor(ot_avp_tensor *tensor, ot_u32 size, ot_avp_npu_shape shape, ot_avp_data_type dtype, ot_u32 stride)
{
    tensor->len = size;
    tensor->shape = shape;
    tensor->dtype = dtype;
    tensor->stride.dims[0] = stride;

    ot_s32 ret = sample_avp_npu_malloc((ot_void **)&tensor->virt_addr, (ot_void **)&tensor->phys_addr, size);
    if (ret != 0) {
        return -1;
    }

    return 0;
}

int ot_avp_destroy_tensor(ot_avp_tensor *tensor)
{
    (void)sample_avp_npu_free((ot_void *)tensor->virt_addr);

    (void)memset(tensor, 0, sizeof(ot_avp_tensor));

    return 0;
}

void prepare_input_output_tensor(void *handle, model_task_info* info)
{
    (void)ot_avp_npu_get_input_num(handle, &info->input_num);
    (void)ot_avp_npu_get_output_num(handle, &info->output_num);
    info->input_dataset = ot_avp_npu_create_input_dataset(handle);
    info->output_dataset = ot_avp_npu_create_output_dataset(handle);

    for (ot_u32 i = 0; i < info->output_num; i++) {
        ot_u32 stride = ot_avp_npu_get_output_default_stride(handle, i);
        ot_u32 buffer_size = ot_avp_npu_get_output_size_by_index(handle, i);
        ot_avp_npu_shape shape;
        (void)ot_avp_npu_get_output_shape_by_index(handle, i, &shape);
        ot_avp_data_type data_type;
        (void)ot_avp_npu_get_output_dtype_by_index(handle, i, &data_type);

        (void)ot_avp_create_tensor(&info->output_tensor[i], buffer_size, shape, data_type, stride);
        // printf("out virt_addr: %llx phys_addr: %llx\n", info->output_tensor[i].virt_addr, info->output_tensor[i].phys_addr);
        (void)ot_avp_npu_set_buffer(info->output_dataset, i, (ot_u8 *)info->output_tensor[i].virt_addr,
            info->output_tensor[i].phys_addr, buffer_size, stride);
    }

    for (ot_u32 i = 0; i < info->input_num; i++) {
        ot_u32 stride = ot_avp_npu_get_input_default_stride(handle, i);
        ot_u32 buffer_size = ot_avp_npu_get_input_size_by_index(handle, i);
        ot_avp_npu_shape shape;
        (void)ot_avp_npu_get_input_shape_by_index(handle, i, &shape);
        ot_avp_data_type data_type;
        (void)ot_avp_npu_get_input_dtype_by_index(handle, i, &data_type);

        (void)ot_avp_create_tensor(&info->input_tensor[i], buffer_size, shape, data_type, stride);
        // printf("in virt_addr: %llx phys_addr: %llx\n", info->input_tensor[i].virt_addr, info->input_tensor[i].phys_addr);
        (void)ot_avp_npu_set_buffer(info->input_dataset, i, (ot_u8 *)info->input_tensor[i].virt_addr,
            info->input_tensor[i].phys_addr, buffer_size, stride);
    }
}

void destory_input_output_tensor(void *handle, model_task_info* info)
{
    (void)handle;
    (void)ot_avp_npu_destroy_dataset(info->input_dataset);
    (void)ot_avp_npu_destroy_dataset(info->output_dataset);
    for (ot_u32 i = 0; i < info->input_num; i++) {
        (void)ot_avp_destroy_tensor(&info->input_tensor[i]);
    }
    for (ot_u32 i = 0; i < info->output_num; i++) {
        (void)ot_avp_destroy_tensor(&info->output_tensor[i]);
    }
}
typedef union {
    ot_float f;    // 浮点数视图
    ot_u32   u32;  // 32位无符号整数视图
} FloatU32;

ot_float data_convert_fp16_to_fp32(const ot_u16 in)
{

    ot_u32 t1 = (ot_u32)in & 0x7fffU;                // Non-sign bits
    ot_u32 t2 = (ot_u32)in & 0x8000U;                // Sign bit
    ot_u32 t3 = (ot_u32)in & 0x7c00U;                // Exponent

    t1 <<= 13;                              // Align mantissa on MSB
    t2 <<= 16;                              // Shift sign bit into position

    t1 += 0x38000000U;                       // Adjust bias

    t1 = (t3 == 0U ? 0U : t1);                // Denormals-as-zero

    t1 |= t2;                               // Re-insert sign bit
    FloatU32 out;
    out.u32=t1;
    return out.f;
}

static ot_u32 as_uint(const float x)
{
    FloatU32 out;
    out.f=x;
    return out.u32;
}

ot_u16 data_convert_fp32_to_fp16(const ot_float in)
{
    ot_u32 b = as_uint(in) + 0x00001000U; // round-to-nearest-even: add last bit after truncated mantissa
    ot_u32 e = (b & 0x7F800000U) >> 23;  // exponent
    ot_u32 m = b & 0x007FFFFFU;          // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000 = decimal indicator flag - initial rounding
    return (ot_u16)(((b & 0x80000000U) >> 16) | ((e > 112U ? 1U : 0U) * ((((e - 112U) << 10) & 0x7C00U) | (m >> 13))) |
           (((e < 113U ? 1U : 0U) & (e > 101U ? 1U : 0U)) * ((((0x007FF000U + m) >> (125U - e)) + 1U) >> 1)) |
           ((e > 143U ? 1U : 0U) * 0x7FFFU));  // sign : normalized : denormalized : saturate
}

#ifndef SIMULATOR
int32_t sample_avp_npu_malloc(void **ptr, ot_phys_addr *phys_addr, uint64_t size)
{
    ot_smr_alloc_attr alloc_attr = {0};
    alloc_attr.len = (ot_u32)size;
    alloc_attr.align = 1;
    (void)strncpy(alloc_attr.region_name, "anony", OT_SMR_REGION_NAME_LEN_MAX - 1);
    (void)strncpy(alloc_attr.chunk_name, "npu_sample", OT_SMR_REGION_NAME_LEN_MAX - 1);

    int32_t ret = ot_smr_alloc(&alloc_attr, phys_addr);
    if (ret != 0) {
        printf("smr alloc mem_size[%lu] fail.\n", size);
        return -1;
    }
    ret = ot_smr_mmap(*phys_addr, alloc_attr.len, OT_FALSE, (ot_void **)ptr);
    if (ret != 0) {
        (void)ot_smr_free(*phys_addr);
        printf("smr mmap mem_size[%lu] fail.\n", size);
        return -1;
    }
    // printf("smr virt: %lx phy: %lx\n", (uintptr_t)*ptr, (uintptr_t)*phys_addr);
    (void)memset(*ptr, 0, size);
    return 0;
}

int32_t sample_avp_npu_free(void *ptr)
{
    ot_phys_addr phys_addr = (ot_phys_addr)OT_NULL;
    int32_t ret = ot_smr_virt2phy(ptr, &phys_addr);
    if (ret != 0) {
        printf("smr virt2phy fail.\n");
        return -1;
    }
    ret = ot_smr_munmap(ptr);
    if (ret != 0) {
        printf("smr munmap fail.\n");
        return -1;
    }
    (void)ot_smr_free(phys_addr);
    return 0;
}
#else
int32_t sample_avp_npu_malloc(void **ptr, ot_phys_addr *phys_addr, uint64_t size)
{
    *ptr = malloc(size);
    *phys_addr=*ptr;
    (void)memset(*ptr, 0, size);
    return 0;
}

int32_t sample_avp_npu_free(void *ptr)
{
    free(ptr);
    return 0;
}
#endif