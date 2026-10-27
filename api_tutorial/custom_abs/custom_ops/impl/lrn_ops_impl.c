#include <stdio.h>
#include "lrn_ops_impl.h"
#include "ot_avp_custom_utils.h"
#include "err_code.h"
#include <math.h>

#define MIN(i, j) (((i) < (j)) ? (i) : (j))
#define MAX(i, j) (((i) > (j)) ? (i) : (j))

                                                                  
#define __FILENAME__ (strrchr("/" __FILE__, '/') + 1)

#ifdef CUSTOM_DEBUG_PRT
#define prt_info(fmt, ...) do { \
    fprintf(stdout, "[I] [custom] [%s:%d] "fmt" \n", __FUNCTION__, __LINE__, ##__VA_ARGS__); \
} while(false)
#else
#define prt_info(...)
#endif
#define prt_err(fmt, ...) do { \
    fprintf(stderr, "[E] [custom] [%s:%d] "fmt" \n", __FILENAME__, __LINE__, ##__VA_ARGS__); \
} while(false)


ot_s32 lrn_infershape(ot_avp_custom_handle custom_op)
{

    ot_avp_npu_shape in0_shape;
    ot_avp_data_type in0_dtype;
    ot_avp_npu_dformat in0_format;
    // get inputs info
    (void)ot_avp_custom_get_input_shape_by_index(custom_op, 0, &in0_shape);
    (void)ot_avp_custom_get_input_dtype_by_index(custom_op, 0, &in0_dtype);
    (void)ot_avp_custom_get_input_format_by_index(custom_op, 0, &in0_format);

    // update outputs info
    (void)ot_avp_custom_set_output_shape_by_index(custom_op, 0, in0_shape);
    (void)ot_avp_custom_set_output_dtype_by_index(custom_op, 0, in0_dtype);
    (void)ot_avp_custom_set_output_format_by_index(custom_op, 0, in0_format);
    return OT_AVP_NPU_SUCCESS;
}

ot_bool lrn_check_supported(ot_avp_custom_handle custom_op)
{

    ot_avp_data_type in0_dtype;
    (void)ot_avp_custom_get_input_dtype_by_index(custom_op, 0, &in0_dtype);
    if (in0_dtype == OT_AVP_DTYPE_F32) {
        return true;
    }
    return false;
}

ot_s32 lrn_compute(ot_avp_custom_handle custom_op)
{
    // --------get attr------
    ot_float alpha = 0.0001f, beta = 0.75f, bias = 1.0f; // default value
    ot_s64 size = 0;
    ot_avp_attr *alpha_attr = ot_avp_custom_get_attr_by_name(custom_op, "alpha", 5);
    if (alpha_attr != NULL) {
        (void)ot_avp_custom_get_attr_float(alpha_attr, &alpha);
    }

    ot_avp_attr *beta_attr = ot_avp_custom_get_attr_by_name(custom_op, "beta", 4);
    if (beta_attr != NULL) {
        (void)ot_avp_custom_get_attr_float(beta_attr, &beta);
    }

    ot_avp_attr *size_attr = ot_avp_custom_get_attr_by_name(custom_op, "size", 4);
    if (size_attr != NULL) {
        (void)ot_avp_custom_get_attr_int(size_attr, &size);
    } else {
        prt_err("get [size] attr failed, [size] in lrn is required");
        return OT_AVP_NPU_INVALID_PARAM;
    }

    ot_avp_attr *bias_attr = ot_avp_custom_get_attr_by_name(custom_op, "bias", 4);
    if (bias_attr != NULL) {
        (void)ot_avp_custom_get_attr_float(bias_attr, &bias);
    }

    // prt_info("lrn attr, alpha: %f, beta: %f, bias: %f, size : %lld", alpha, beta, bias, size);
    
    // --------get tensor---------
    ot_avp_npu_shape in0_shape;
    (void)ot_avp_custom_get_input_shape_by_index(custom_op, 0, &in0_shape);
    ot_u64 in0_buf_len;
    (void)ot_avp_custom_get_input_buf_len_by_index(custom_op, 0, &in0_buf_len);
    ot_u64 in0_len = in0_buf_len / sizeof(ot_float);
    ot_float* in0_buf = (ot_float*)ot_avp_custom_get_input_buf_by_index(custom_op, 0);
    if (in0_buf == NULL) {
        return OT_AVP_NPU_INVALID_PARAM;
    }
    ot_u64 out0_buf_len;
    (void)ot_avp_custom_get_output_buf_len_by_index(custom_op, 0, &out0_buf_len);

    ot_u64 out0_len = out0_buf_len / sizeof(ot_float);
    if (in0_len != out0_len) {
        return OT_AVP_NPU_INVALID_PARAM;
    }
    ot_float* out0_buf = (ot_float*)ot_avp_custom_get_output_buf_by_index(custom_op, 0);
    if (out0_buf == NULL) {
        return OT_AVP_NPU_INVALID_PARAM;
    }

    // ---------compute------------- 
    ot_u32 batch = (ot_u32)in0_shape.dims[0];
    ot_s32 channel = in0_shape.dims[1];
    ot_u32 one_channel_size = 0U;
    ot_u32 one_batch_size = 0U;
    if (batch != 0U) {
        one_batch_size = (ot_u32)in0_len / batch;
        if (channel != 0) {
            one_channel_size = (ot_u32)in0_len / (batch * (ot_u32)channel);
        }
    }
    ot_u32 index = 0;
    for (ot_u32 b = 0; b < batch; b++) {
        for (ot_s32 c = 0; c < channel; c++) {
            for (ot_u32 j = 0; j < one_channel_size; j++) {
                int region_down = (int)floor((double)c - (((double)size - 1.0)/2.0));
                region_down = MAX(0, region_down);
                int region_up = (int)ceil((double)c + (((double)size - 1.0)/2.0));
                region_up = MIN(channel - 1, region_up);
                float squart_sum = 0.0f;
                for (int regionc = region_down; regionc <= region_up; regionc++) {
                    ot_u32 region_index = (b * one_batch_size) + ((ot_u32)regionc*one_channel_size) + j;
                    squart_sum += powf(in0_buf[region_index], 2.0f);
                }
                if (size != 0 && squart_sum != 0.0f) {
                    out0_buf[index] = powf(bias + (alpha / (float)size * squart_sum), beta);
                    if (out0_buf[index] != 0.0f) {
                        out0_buf[index] = in0_buf[index] / out0_buf[index];
                    }
                }
                index++;
            }
        }
    }
    return OT_AVP_NPU_SUCCESS;
}
