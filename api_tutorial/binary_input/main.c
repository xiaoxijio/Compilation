#include "ot_avp_npu_rts.h"
#include "file_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "opt_args.h"
#ifdef __GNUC__
#include <unistd.h>
#endif
#ifndef SIMULATOR
#include "platform_utils.h"
#else
#include "simulator_utils.h"
#endif

int main(int argc, char** argv) {
#ifdef SIMULATOR
    ot_avp_npu_config config = {
        .core_ids = 0,
        .dump_layer = false,   // dump layer
        .perf_mode = false,  // perf_mode
        .log_level = 2,   // log level info

    };
#if defined(_WIN32)
    const char* model_path = "..\\resnet50_binary.ortm";
    const char* input_path = "..\\ILSVRC2012_val_00024327.bin";
#else
    const char* model_path = "./tmp_models/resnet50_binary.ortm";
    const char* input_path = "./tmp_res/ILSVRC2012_val_00024327.bin";
#endif
#else
    ot_avp_npu_config config = {
        .core_ids = 0,
        .dump_layer = false,   // dump layer
        .perf_mode = false,  // perf_mode
        .log_level = 2,   // log level
    };
    (void)sample_avp_npu_media_system_init();
    const char* model_path = "../../data/prebuild_ortm/resnet50_binary.ortm";
    const char* input_path = "../../data/ImageNet/binary/ILSVRC2012_val_00024327.bin";
#endif
    arg_config opt_arg = {NULL, NULL, NULL, false, false, false, 1U, 0U, 0U, 0U, false};
    if (-1 == parse_optarg(argc, argv, &opt_arg)) {
        fprintf(stderr, "parse arguments failed, exiting.\n");
        return -1;
    }
    if (opt_arg.input_path != NULL) {
        input_path = opt_arg.input_path;
    }
    if (opt_arg.model_path != NULL) {
        model_path = opt_arg.model_path;
    }

    printf("binary_input sample start\n");

    ot_avp_handle handle = NULL;
    int32_t ret = ot_avp_npu_init(&config);
    if (ret != 0) {
        printf("npu init error\n");
        return ret;
    }

    // load model
    ret = ot_avp_npu_load_model(model_path, &handle);
    if (ret != 0) {
        printf("open model error\n");
        (void)ot_avp_npu_deinit();
        return ret;
    }

    // get input and output information
    uint32_t input_num = 0;
    uint32_t output_num = 0;
    (void)ot_avp_npu_get_input_num(handle, &input_num);
    (void)ot_avp_npu_get_output_num(handle, &output_num);

    ot_avp_npu_shape input_shape;
    ot_avp_npu_shape output_shape;
    (void)ot_avp_npu_get_input_shape_by_index(handle, 0, &input_shape);
    (void)ot_avp_npu_get_output_shape_by_index(handle, 0, &output_shape);

    ot_avp_data_type input_dtype, output_dtype;
    (void)ot_avp_npu_get_input_dtype_by_index(handle, 0, &input_dtype);
    (void)ot_avp_npu_get_output_dtype_by_index(handle, 0, &output_dtype);

    ot_avp_npu_dataset* input_dataset = ot_avp_npu_create_input_dataset(handle);
    ot_avp_npu_dataset* output_dataset = ot_avp_npu_create_output_dataset(handle);

    // create buffer
    uint32_t input_len = ot_avp_npu_get_input_size_by_index(handle, 0);
    void* input_data = NULL;
    (void)sample_avp_npu_malloc(&input_data, input_len);
    (void)memset(input_data, 0, input_len);
    uint32_t input_stride = ot_avp_npu_get_input_default_stride(handle, 0);
    (void)ot_avp_npu_set_buffer(input_dataset, 0, input_data, 0, input_len, input_stride);
    (void)load_data_from_file(input_path, input_data, input_shape, input_stride, input_dtype);

    uint32_t output_len = ot_avp_npu_get_output_size_by_index(handle, 0);

    void* output_data = NULL;
    (void)sample_avp_npu_malloc(&output_data, output_len);
    (void)memset(output_data, 0, output_len);
    uint32_t output_stride = ot_avp_npu_get_output_default_stride(handle, 0);
    (void)ot_avp_npu_set_buffer(output_dataset, 0, output_data, 0, output_len, output_stride);

    // execute model
    ret = ot_avp_npu_execute(handle, input_dataset, output_dataset);
    if (ret != 0) {
        printf("run model error\n");
    }

    (void)dump_data_to_binary_file("output.bin", (ot_u8*)output_data, output_shape, (ot_u32)output_stride, output_dtype);
    printf("sample end\n");

    (void)ot_avp_npu_destroy_dataset(input_dataset);
    (void)ot_avp_npu_destroy_dataset(output_dataset);
    (void)ot_avp_npu_unload_model(handle);
    (void)ot_avp_npu_deinit();
    (void)sample_avp_npu_free(input_data);
    (void)sample_avp_npu_free(output_data);
#ifndef SIMULATOR
    (void)sample_avp_npu_media_system_deinit();
#endif
    return ret;
}
