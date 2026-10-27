#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#include "file_utils.h"
#include "opt_args.h"
#include "ot_avp_npu_rts.h"
#ifdef __GNUC__
#include <unistd.h>
#endif
#include <cmath>
#include <iostream>

#ifndef SIMULATOR
#include "platform_utils.h"
#else
#include "simulator_utils.h"
#endif
using namespace std;

float obj_threshold = 0.6f;
float nms_threshold = 0.2f;
#define CATEGORY          80
#define OUTPUT_SHAPE_NUMS 2
#define OUTPUT_SHAPE_DIMS 4
#define CURRENT_DIM       416.0f
#define BATCH_SIZE        1

vector<vector<int> > g_masks = {
    {0, 1, 2},
    {3, 4, 5}
};
vector<vector<int> > g_anchors = {
    { 10,  14},
    { 23,  27},
    { 37,  58},
    { 81,  82},
    {135, 169},
    {344, 319}
};
int output_shape[OUTPUT_SHAPE_NUMS][OUTPUT_SHAPE_DIMS] = {
    {1, (3 * (4 + 1 + CATEGORY)), 26, 26},
    {1, (3 * (4 + 1 + CATEGORY)), 13, 13}
};

vector<string> labels = {
    "person",         "bicycle",    "car",           "motorbike",     "aeroplane",    "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "sofa",       "potted plant",  "bed",           "dining table", "toilet",        "TV monitor",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"};

typedef struct {
    float         prob;
    vector<float> label;
    float         x1, y1, x2, y2, score;
    float         max_label, max_id;
} DetectionRes;

static float sigmoid(float in)
{
    if ((1.f + expf(-in)) != 0.0f) {
        /* LDRA_EXCLUDE 127 D */
        return 1.f / (1.f + expf(-in));
    }
    return 0.0f;
}

static float exponential(float in) { return expf(in); }

static float arg_max(vector<float>& label_list)
{
    float ret_id   = 0.0f;
    float last_max = 0.0f;
    for (size_t i = 0UL; i < label_list.size(); i++) {
        if (label_list[i] > last_max) {
            last_max = label_list[i];
            ret_id   = (float)i;
        }
    }
    return ret_id;
}

static void DoNms(vector<DetectionRes>& detections, float nmsThresh)
{
    auto iouCompute = [](DetectionRes& lbox, DetectionRes& rbox) {
        float interBox[] = {
            max(lbox.x1, rbox.x1),  // x1
            max(lbox.y1, rbox.y1),  // y1
            min(lbox.x2, rbox.x2),  // x2
            min(lbox.y2, rbox.y2),  // y2
        };

        if (interBox[1] >= interBox[3] || interBox[0] >= interBox[2]) {
            return 0.0f;
        }

        float interBoxS = (interBox[2] - interBox[0]) * (interBox[3] - interBox[1]);
        float lbox_area = (lbox.x2 - lbox.x1) * (lbox.y2 - lbox.y1);
        float rbox_area = (rbox.x2 - rbox.x1) * (rbox.y2 - rbox.y1);
        float extra     = 0.0000000001f;
        if ((lbox_area + rbox_area - interBoxS + extra) != 0.0f) {
            /* LDRA_EXCLUDE 127 D */
            return interBoxS / (lbox_area + rbox_area - interBoxS + extra);
        } else {
            return 0.0f;
        }
    };

    vector<DetectionRes> result;
    for (unsigned int m = 0U; m < detections.size(); ++m) {
        result.push_back(detections[m]);
        for (unsigned int n = m + 1U; n < detections.size(); ++n) {
            if (iouCompute(detections[m], detections[n]) > nmsThresh) {
                detections.erase(detections.begin() + n);
                --n;
            }
        }
    }
    detections = move(result);
}

static DetectionRes rescale(DetectionRes box, int height, int width)
{
    float pad_x = 0.0f;
    float pad_y = 0.0f;
    int   hw    = max(height, width);
    if (hw != 0) {
        pad_x = (float)max(height - width, 0) * (1.0f * CURRENT_DIM / (float)hw);
        pad_y = (float)max(width - height, 0) * (1.0f * CURRENT_DIM / (float)hw);
    }

    float unpad_h = CURRENT_DIM - pad_y;
    float unpad_w = CURRENT_DIM - pad_x;
    /* LDRA_EXCLUDE 139 S */
    if (unpad_w != 0.0f && unpad_h != 0.0f) {
        box.x1 = ((box.x1 - (pad_x / 2.0f)) / unpad_w) * (float)width;
        box.y1 = ((box.y1 - (pad_y / 2.0f)) / unpad_h) * (float)height;
        box.x2 = ((box.x2 - (pad_x / 2.0f)) / unpad_w) * (float)width;
        box.y2 = ((box.y2 - (pad_y / 2.0f)) / unpad_h) * (float)height;
    }

    return box;
}

static vector<DetectionRes> postProcess(float* output)
{
    vector<vector<int> > shapes;
    for (int i = 0; i < OUTPUT_SHAPE_NUMS; i++) {
        auto        shape = output_shape[i];
        vector<int> tmp   = {shape[2], shape[3], 3, 4 + 1 + CATEGORY};
        shapes.push_back(tmp);
    }

    int                  offset = 0;
    vector<float>        label_score;
    vector<DetectionRes> all_detections;
    for (int i = 0; i < OUTPUT_SHAPE_NUMS; i++) {  // batch size
        auto                 masks = g_masks[i];
        vector<vector<int> > anchors;
        for (auto mask : masks) {
            anchors.push_back(g_anchors[mask]);
        }
        auto shape = shapes[i];

        for (int h = 0; h < shape[0]; h++) {
            int offset_h = offset + (h * shape[1] * shape[2] * shape[3]);
            for (int w = 0; w < shape[1]; w++) {
                int offset_w = offset_h + (w * shape[2] * shape[3]);
                for (int c = 0; c < shape[2]; c++) {
                    int offset_c = offset_w + (c * shape[3]);

                    float* ptr = output + offset_c;

                    ptr[4] = sigmoid(ptr[4]);  // conf

                    if (ptr[4] < obj_threshold) {
                        continue;
                    }
                    for (int j = 5; j < (5 + CATEGORY); j++) {
                        ptr[j] = sigmoid(ptr[j]);
                    }

                    ptr[0] = (sigmoid(ptr[0]) + (float)w) * CURRENT_DIM / (float)shape[0];  // center x
                    ptr[1] = (sigmoid(ptr[1]) + (float)h) * CURRENT_DIM / (float)shape[1];  // center y
                    ptr[2] = exponential(ptr[2]) * anchors[c][0];                           // cal w
                    ptr[3] = exponential(ptr[3]) * anchors[c][1];                           // cal h

                    DetectionRes det;
                    det.prob = ptr[4];
                    for (int j = 5; j < (5 + CATEGORY); j++) {
                        det.label.push_back(ptr[j]);
                    }
                    det.x1           = ptr[0] - (ptr[2] / 2.0f);
                    det.y1           = ptr[1] - (ptr[3] / 2.0f);
                    det.x2           = ptr[0] + (ptr[2] / 2.0f);
                    det.y2           = ptr[1] + (ptr[3] / 2.0f);
                    auto maxPosition = max_element(det.label.begin(), det.label.end());
                    det.score        = ptr[4] * (*maxPosition);
                    for (int j = 5; j < (5 + CATEGORY); j++) {
                        label_score.push_back(ptr[4] * ptr[j]);
                    }
                    det.max_label = *maxPosition;
                    det.max_id    = arg_max(label_score);
                    label_score.clear();

                    all_detections.push_back(det);
                }
            }
        }
        offset += shape[0] * shape[1] * shape[2] * shape[3];
    }

    sort(all_detections.begin(), all_detections.end(),
         [=](const DetectionRes& left, const DetectionRes& right) { return left.score > right.score; });
    DoNms(all_detections, nms_threshold);

    return all_detections;
}

int main(int argc, char** argv)
{
    ot_avp_npu_config config = {};

    ot_avp_npu_shape input_shape;
    ot_avp_npu_shape output_shape0, output_shape1;
    ot_avp_data_type input_dtype;
    uint32_t         dims0 = 1U, dims1 = 1U;

    uint32_t input_num  = 0U;
    uint32_t output_num = 0U;

    uint8_t* input_data      = nullptr;
    uint8_t* out0_buf        = nullptr;
    uint8_t* out0_align_buf  = nullptr;
    uint8_t* out1_buf        = nullptr;
    uint8_t* out1_align_buf  = nullptr;
    uint32_t input_len       = 0U;
    uint32_t input_stride    = 0U;
    uint32_t out0_len        = 0U;
    uint32_t out0_stride_len = 0U;
    uint32_t out0_w_len      = 0U;
    uint32_t out1_len        = 0U;
    uint32_t out1_stride_len = 0U;
    uint32_t out1_w_len      = 0U;
    size_t   out_size        = 0UL;
    uint8_t* p               = nullptr;
    uint8_t* pOut            = nullptr;
    void*    out             = nullptr;
    int      height          = 325;
    int      width           = 500;

    std::vector<float>        results;
    std::vector<DetectionRes> all_detections;
    std::ofstream             outFile("results.bin", std::ios::binary);

    ot_avp_npu_dataset* input_dataset  = nullptr;
    ot_avp_npu_dataset* output_dataset = nullptr;

    config.core_ids          = 0U;
    config.dump_layer        = OT_FALSE;  // dump layer
    config.perf_mode         = OT_FALSE;  // perf_mode
    config.log_level         = 2U;         // log level info
#if defined(_WIN32)
    const char* model_path = "..\\tiny-yolov3_yuv420sp.ortm";
    const char* input_path = "..\\COCO_val2014_000000568213.yuv420sp";
#else
#ifdef SIMULATOR
    const char* model_path = "./tmp_models/tiny-yolov3_yuv420sp.ortm";
    const char* input_path = "./tmp_res/COCO_val2014_000000568213.yuv420sp";
#else
    const char* model_path = "../../data/prebuild_ortm/tiny-yolov3_yuv420sp.ortm";
    const char* input_path = "../../data/COCO2014/yuv420sp/COCO_val2014_000000568213.yuv420sp";
    (void)sample_avp_npu_media_system_init();
#endif
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
    printf("tiny-yolov3 isa simulator start\n");
    ot_avp_handle handle = NULL;

    int ret = ot_avp_npu_init(&config);
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
    ot_u32 runtime_buf_size = 0U;
    (void)ot_avp_npu_get_runtime_buffer_size(handle, &runtime_buf_size);
    printf("runtime buf size: %d\n", runtime_buf_size);

    void *runtime_buf = NULL;
    if (runtime_buf_size != 0U ) {
#ifdef SIMULATOR
        ret = sample_avp_npu_malloc(&runtime_buf, runtime_buf_size);
        if (ret != 0) {
            printf("malloc runtime buf fail\n");
            goto FAIL;
        }
        ret = ot_avp_npu_set_runtime_buffer(handle, (ot_u8*)runtime_buf, runtime_buf_size);
        if (ret != 0) {
            printf("ot_avp_npu_set_runtime_buffer fail\n");
            goto FAIL;
        }
#else
        ret = sample_avp_npu_set_runtime_buffer(handle,&runtime_buf,runtime_buf_size,0);
        if (ret != 0) {
            printf("sample_avp_npu_set_runtime_buffer fail\n");
            goto FAIL;
        }
#endif
    }
    // get input and output information

    (void)ot_avp_npu_get_input_num(handle, &input_num);
    (void)ot_avp_npu_get_output_num(handle, &output_num);
    printf("model input num: %d, model output num: %d\n", input_num, output_num);


    (void)ot_avp_npu_get_input_shape_by_index(handle, 0, &input_shape);
    (void)ot_avp_npu_get_output_shape_by_index(handle, 0, &output_shape0);
    (void)ot_avp_npu_get_output_shape_by_index(handle, 1, &output_shape1);
    (void)ot_avp_npu_get_input_dtype_by_index(handle, 0, &input_dtype);
    input_dataset  = ot_avp_npu_create_input_dataset(handle);
    output_dataset = ot_avp_npu_create_output_dataset(handle);

    printf("input shape: ");
    for (uint32_t i = 0U; i < input_shape.dim_size; i++) {
        printf("%d ", input_shape.dims[i]);
    }
    input_len    = ot_avp_npu_get_input_size_by_index(handle, 0);  // 416*416*3/2
    input_stride = ot_avp_npu_get_input_default_stride(handle, 0);
    printf("input_len:%d, input_stride:%d\n", input_len, input_stride);
    ret = sample_avp_npu_malloc((void**)&input_data, input_len);
    if (ret != 0) {
        printf("malloc failed");
        goto FAIL;
    }
    (void)load_data_from_file(input_path, input_data, input_shape, input_stride, input_dtype);
    (void)ot_avp_npu_set_buffer(input_dataset, 0, input_data, 0, input_len, input_stride);

    printf("output shape0: ");
    for (uint32_t i = 0U; i < output_shape0.dim_size; i++) {
        printf("%d ", output_shape0.dims[i]);
        dims0 *= (uint32_t)output_shape0.dims[i];
    }
    ret = sample_avp_npu_malloc((void**)&out0_buf, dims0 * sizeof(float));
    if (ret != 0) {
        printf("malloc failed");
        goto FAIL;
    }
    (void)memset(out0_buf, 0, (dims0 * sizeof(float)));
    out0_len        = ot_avp_npu_get_output_size_by_index(handle, 0);
    out0_stride_len = ot_avp_npu_get_output_default_stride(handle, 0);
    out0_w_len      = ((uint32_t)output_shape0.dims[output_shape0.dim_size - 1U]) * sizeof(float);
    printf("shape0 size:%d, out0_len:%d, out0_w_len:%d, out0_stride_len:%d\n", dims0, out0_len, out0_w_len,
           out0_stride_len);
    ret = sample_avp_npu_malloc((void**)&out0_align_buf, out0_len);
    if (ret != 0) {
        printf("malloc failed");
        goto FAIL;
    }
    (void)memset(out0_align_buf, 0, out0_len);
    (void)ot_avp_npu_set_buffer(output_dataset, 0, out0_align_buf, 0, out0_len, out0_stride_len);

    printf("output shape1: ");
    for (uint32_t i = 0U; i < output_shape1.dim_size; i++) {
        printf("%d ", output_shape1.dims[i]);
        dims1 *= (uint32_t)output_shape1.dims[i];
    }
    ret = sample_avp_npu_malloc((void**)&out1_buf, dims1 * sizeof(float));
    if (ret != 0) {
        printf("malloc failed");
        goto FAIL;
    }
    (void)memset(out1_buf, 0, (dims1 * sizeof(float)));
    out1_len        = ot_avp_npu_get_output_size_by_index(handle, 1);
    out1_stride_len = ot_avp_npu_get_output_default_stride(handle, 1);
    out1_w_len      = ((uint32_t)output_shape1.dims[output_shape1.dim_size - 1U]) * sizeof(float);
    printf("shape1 size:%d, out1_len:%d, out1_w_len:%d, out1_stride_len:%d\n", dims1, out1_len, out1_w_len,
           out1_stride_len);
    ret = sample_avp_npu_malloc((void**)&out1_align_buf, out1_len);
    if (ret != 0) {
        printf("malloc failed");
        goto FAIL;
    }
    (void)memset(out1_align_buf, 0, out1_len);
    (void)ot_avp_npu_set_buffer(output_dataset, 1, out1_align_buf, 0, out1_len, out1_stride_len);

    // trigger npu execute model
    ret = ot_avp_npu_execute(handle, input_dataset, output_dataset);
    if (ret != 0) {
        printf("run model error\n");
        return -1;
    }


    // user extract valid data from aligned output memory when parsing and post-processing
    p    = out0_align_buf;
    pOut = out0_buf;
    for (uint32_t i = 0U; i < (1U * 26U * 26U); i++) {
        for (uint32_t w = 0U; w < out0_w_len; w++) {
            *(pOut + (i * out0_w_len) + w) = *(p + (i * out0_stride_len) + w);
        }
    }
    p    = out1_align_buf;
    pOut = out1_buf;
    for (uint32_t i = 0U; i < (1U * 13U * 13U); i++) {
        for (uint32_t w = 0U; w < out1_w_len; w++) {
            *(pOut + (i * out1_w_len) + w) = *(p + (i * out1_stride_len) + w);
        }
    }

    out_size = (dims0 + dims1) * sizeof(float);
    out      = malloc(out_size);
    if (out == nullptr) {
        ret = -1;
        goto FAIL;
    }
    (void)memset(out, 0, out_size);
    (void)memcpy(out, (void*)out0_buf, dims0 * sizeof(float));
    (void)memcpy((uint8_t*)out + (dims0 * sizeof(float)), out1_buf, dims1 * sizeof(float));
    all_detections = postProcess((float*)out);
    // print boxes
    for (size_t j = 0UL; j < all_detections.size(); j++) {
        printf("---------obj:%ld---------\n", j);
        DetectionRes* detRes = &all_detections[j];
        printf("x1:%.3f, y1:%.3f, x2:%.3f, y2:%.3f\n", detRes->x1, detRes->y1, detRes->x2, detRes->y2);
        printf("%.5f, %.5f, %s\n\n", detRes->prob, detRes->max_label, labels[detRes->max_id].c_str());
        all_detections[j] = rescale(all_detections[j], height, width);
        printf("detect info rescale:\n");
        printf("x1:%.3f, y1:%.3f, x2:%.3f, y2:%.3f\n", detRes->x1, detRes->y1, detRes->x2, detRes->y2);
        printf("%.5f, %.5f, %s\n\n", detRes->prob, detRes->max_label, labels[detRes->max_id].c_str());
        results.push_back(detRes->max_id);
        results.push_back(detRes->x1);
        results.push_back(detRes->y1);
        results.push_back(detRes->x2);
        results.push_back(detRes->y2);
    }
    outFile.write((char*)(results.data()), sizeof(float) * results.size());

    printf("sample_yuv420sp_input end\n");

    free(out);
FAIL:
    if (out1_align_buf != OT_NULL) {
        sample_avp_npu_free(out1_align_buf);
    }
    if (out1_buf != OT_NULL) {
        sample_avp_npu_free(out1_buf);
    }
    if (out0_align_buf != OT_NULL) {
        sample_avp_npu_free(out0_align_buf);
    }
    if (out0_buf != OT_NULL) {
        sample_avp_npu_free(out0_buf);
    }
    if (input_data != OT_NULL) {
        sample_avp_npu_free(input_data);
    }
    if (input_dataset != OT_NULL) {
        ot_avp_npu_destroy_dataset(input_dataset);
    }
    if (output_dataset != OT_NULL) {
        ot_avp_npu_destroy_dataset(output_dataset);
    }
    if (runtime_buf != OT_NULL) {
        sample_avp_npu_free(runtime_buf);
    }
    (void)ot_avp_npu_unload_model(handle);
    (void)ot_avp_npu_deinit();
    outFile.close();
#ifndef SIMULATOR
    (void)sample_avp_npu_media_system_deinit();
#endif
    return ret;
}
