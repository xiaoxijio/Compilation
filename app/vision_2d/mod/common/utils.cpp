#include "utils.h"
#ifndef SIMULATOR
#include <arm_neon.h>
#endif
#include "sample_avp_npu_common.h"
#include <math.h>
#include <cmath>  // for fabs
#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <cfloat> // for FLT_MAX

float sigmoid(float x)
{
    return 1 / (1 + exp(-x));
}
 

// 快速指数函数近似
inline float fast_exp(float x) {
    x = 1.0f + x / 1024.0f;
    x *= x; x *= x; x *= x; x *= x;
    x *= x; x *= x; x *= x; x *= x;
    x *= x; x *= x;
    return x;
}

void Yolo_process::insert_object(float conf, int position, int class_id, int index) {
    if (conf < obj_threshold) {
        return;
    }
    if (obj_count == MAX_OBJECTS && conf < obj_infos[obj_count -1].conf) {
        return;
    }
    if (obj_count == MAX_OBJECTS) {
        obj_count--;
    }
    int i = obj_count;
    for (; i > 0 && obj_infos[i-1].conf < conf; --i) {
        obj_infos[i] = obj_infos[i - 1];
    }
    obj_infos[i].conf = conf;
    obj_infos[i].position = position;
    obj_infos[i].class_id = class_id;
    obj_infos[i].ouptut_index = index;
    obj_count++;
}
void Yolo_process::get_valid_object(){
    get_valid_object(0);
    get_valid_object(1);
    get_valid_object(2);
}
void Yolo_process::get_valid_object(int index){
    uint16_t* data = conf_data_[index];
    int fature_size = w[index]*w[index];
    std::vector<std::pair<float, int>> conf(fature_size);
    for (int c = 0, offset = 0; c < 80; ++c) {
        for (int i = 0; i < fature_size; ++i, ++offset) {
            float cur_conf = data_convert_fp16_to_fp32(data[offset]);
            cur_conf = sigmoid(cur_conf); // 快速查表sigmoid
            if (cur_conf >= conf[i].first) {
                conf[i].first = cur_conf;
                conf[i].second = c;
            }
        }
    }
    for (int i = 0; i < fature_size; ++i) {
        if (conf[i].first > obj_threshold) {
            insert_object(conf[i].first, i, conf[i].second, index);
        }
    }
}

void Yolo_process::update_valid_object() {
    for (int i = 0; i < obj_count; ++i) {
        int cur_output_index = obj_infos[i].ouptut_index;
        int fature_size = w[cur_output_index]*w[cur_output_index];
        std::vector<float> after_conv16(4, 0);
        for (int j = 0; j < 4; ++j) {
            float cur_sum = 0;
            float cur_max = std::numeric_limits<float>::lowest();
            std::vector<float> cur16data(16,0);
            for (int k = 0; k < 16; ++k) {
                uint16_t* data_posation = cor_data_[cur_output_index] + j * 16 * fature_size + obj_infos[i].position + k * fature_size;
                cur16data[k] = data_convert_fp16_to_fp32(data_posation[0]);
                cur_max = cur16data[k] > cur_max ? cur16data[k] : cur_max;
            }
            for (int k = 0; k < 16; ++k) {
                cur16data[k] = std::exp(cur16data[k] - cur_max) + 1e-5;
                cur_sum += cur16data[k];
            }
            for (int k = 0; k < 16; ++k) {
                after_conv16[j] += k * (cur16data[k] / cur_sum); // conv加权
            }
        }
        int hi = obj_infos[i].position / w[cur_output_index];
        int wi = obj_infos[i].position - hi * w[cur_output_index];
        obj_infos[i].x1 = ((wi + 0.5) - after_conv16[0]) * 8 * (cur_output_index + 1);
        obj_infos[i].y1 = ((hi + 0.5) - after_conv16[1]) * 8 * (cur_output_index + 1);
        obj_infos[i].x2 = ((wi + 0.5) + after_conv16[2]) * 8 * (cur_output_index + 1);
        obj_infos[i].y2 = ((hi + 0.5) + after_conv16[3]) * 8 * (cur_output_index + 1);
        class_buckets[obj_infos[i].class_id].push_back(i);
        obj_infos[i].area = (obj_infos[i].x1 - obj_infos[i].x2) * (obj_infos[i].y1 - obj_infos[i].y2);
    }
}
inline float fast_iou(const object_info* a, const object_info* b) {
    float xx1 = (a->x1 > b->x1) ? a->x1 : b->x1;
    float yy1 = (a->y1 > b->y1) ? a->y1 : b->y1;
    float xx2 = (a->x2 < b->x2) ? a->x2 : b->x2;
    float yy2 = (a->y2 < b->y2) ? a->y2 : b->y2;
    
    float w = (xx2 > xx1) ? (xx2 - xx1) : 0.0f;
    float h = (yy2 > yy1) ? (yy2 - yy1) : 0.0f;
    float inter = w * h;
    return inter / (a->area + b->area - inter);
}
void Yolo_process::NMS() {
    for (const auto& class_bucket : class_buckets) {
        std::vector<int> current_class = class_bucket.second;
        std::sort(current_class.begin(), current_class.end(), 
            [this](int a, int b) { 
                return obj_infos[a].conf > obj_infos[b].conf; 
            });
        
        // 执行NMS
        for (size_t i = 0UL; i < current_class.size(); ++i) {
            object_info det_i = obj_infos[current_class[i]];
            if (det_i.conf < obj_threshold) continue;  // 已被抑制

            for (size_t j = i + 1UL; j < current_class.size(); ++j) {
                object_info &det_j = obj_infos[current_class[j]];
                if (det_j.conf < obj_threshold) continue;
                // 快速IoU计算
                if (fast_iou(&det_i, &det_j) > iou_threshold) {
                    det_j.conf = 0;  // 标记为抑制
                }
            }
        }
    }
}
void Yolo_process::prinf_box() {
    printf("iou  %f, conf %f \n", iou_threshold, obj_threshold);
    for (int i = 0; i < obj_count; i++) {
        if (obj_infos[i].conf < obj_threshold) {
            continue;
        }
        printf("box: %d, x1 %f, y1 %f, x2 %f, y2 %f, conf %f \n", obj_infos[i].class_id, obj_infos[i].x1 ,obj_infos[i].y1, obj_infos[i].x2, obj_infos[i].y2, obj_infos[i].conf);
    }
}

//execute for yolov8 and yolov10
ot_s32 sample_yolo_execute(sample_avp_model_task_info_t* model_info, const char* input_path)
{
    int32_t ret = -1;

    ret = load_data_from_file(input_path, (uint8_t *)model_info->input_tensor[0].virt_addr, model_info->input_tensor[0].shape,
        model_info->input_tensor[0].stride, model_info->input_tensor[0].dtype);

    // 模型推理
    ret = sample_avp_execute_model(model_info, 1);

    // [1*64*80*80],[1*80*80*80] [1*64*40*40],[1*80*40*40] [1*64*20*20],[1*80*20*20]
    Yolo_process yolo_process = Yolo_process(
        std::vector<uint16_t*>{(uint16_t *)(model_info->output_tensor[1].virt_addr), (uint16_t *)(model_info->output_tensor[3].virt_addr), (uint16_t *)(model_info->output_tensor[5].virt_addr)},
        std::vector<uint16_t*>{(uint16_t *)(model_info->output_tensor[0].virt_addr), (uint16_t *)(model_info->output_tensor[2].virt_addr), (uint16_t *)(model_info->output_tensor[4].virt_addr)});
    
    SAMPLE_AVP_CHECK_GOTO_WITH_LOG(ret != 0, EXECUTE_MODEL_FAIL, "execute model fail.\n");
    yolo_process.pre_job();
    yolo_process.get_valid_object();
    yolo_process.update_valid_object();
    yolo_process.NMS();
    yolo_process.prinf_box();
    return ret;

EXECUTE_MODEL_FAIL:
    sample_avp_destroy_datasets(model_info->handle, model_info);
    ot_avp_npu_unload_model(model_info->handle);
    return ret;
}

//   筛选好的框 打印画到图上
ot_s32 bbox_compare(const void *aa, const void *bb)
{
    sample_avp_npu_rect *a = (sample_avp_npu_rect *)aa;
    sample_avp_npu_rect *b = (sample_avp_npu_rect *)bb;
    // 按照confidence从高到底排序
    if (a->conf < b->conf) {
        return OT_TRUE;
    }
    return OT_FALSE;
}

#define VGETQ_LANE_F32(INDEX) \
    val = vgetq_lane_f32(final_max_vec, (INDEX)); \
    if (val > max_value) { \
        max_value = val; \
        max_index = vgetq_lane_s32(final_max_index_vec, (INDEX)); \
    }


static int get_max_index_float32(const float* arr, int size) {
    if (size <= 0) {
        return -1;
    }
    int max_index = 0;
#ifdef SIMULATOR
    for (int i = 0; i < size; ++i) {
        if (arr[i] > arr[max_index]) {
            max_index = i;
        }
    }
#else
    // 每次处理8个数
    int i;
    float32x4_t max_vec1 = vdupq_n_f32(arr[0]);
    float32x4_t max_vec2 = vdupq_n_f32(arr[0]);
    int32x4_t index_vec1 = {0, 1, 2, 3};
    int32x4_t index_vec2 = {4, 5, 6, 7};
    int32x4_t max_index_vec1 = vdupq_n_s32(0);
    int32x4_t max_index_vec2 = vdupq_n_s32(0);

    for (i = 0; i <= (size - 8); ) {
        float32x4_t current1 = vld1q_f32(&arr[i]);
        float32x4_t current2 = vld1q_f32(&arr[i+4]);

        uint32x4_t mask1 = vcgtq_f32(current1, max_vec1);
        uint32x4_t mask2 = vcgtq_f32(current2, max_vec2);

        max_vec1 = vbslq_f32(mask1, current1, max_vec1);
        max_vec2 = vbslq_f32(mask2, current2, max_vec2);

        max_index_vec1 = vbslq_s32(mask1, index_vec1, max_index_vec1);
        max_index_vec2 = vbslq_s32(mask2, index_vec2, max_index_vec2);

        index_vec1 = vaddq_s32(index_vec1, vdupq_n_s32(8));
        index_vec2 = vaddq_s32(index_vec2, vdupq_n_s32(8));
        i += 8;
    }

    // 比较两个向量找出全局最大值
    uint32x4_t mask = vcgtq_f32(max_vec2, max_vec1);
    float32x4_t final_max_vec = vbslq_f32(mask, max_vec2, max_vec1);
    int32x4_t final_max_index_vec = vbslq_s32(mask, max_index_vec2, max_index_vec1);

    // 找出向量中的最大值
    float max_value = -FLT_MAX, val = -FLT_MAX;
    VGETQ_LANE_F32(0)
    VGETQ_LANE_F32(1)
    VGETQ_LANE_F32(2)
    VGETQ_LANE_F32(3)

    // 处理剩余元素
    for (; i < size; i++) {
        if (arr[i] > max_value) {
            max_value = arr[i];
            max_index = i;
        }
    }
#endif
    return max_index;
}

/*
    1 * 8400 * 84
    topk 1 * topk  xywhcn 6 小于config的排除
*/
void yolov10_postProcess_float32(void* output, float* selectBox, float obj_threshold,
        int batch_size, int box_num, int category, void* boxBuf, int topk, bool is_xywh) {
    float* real_type_output = (float *)output;
    transpose_nobubble_float32(real_type_output, 84, 8400); // 顺序要求 {
    float* curOutputPtr = real_type_output + 4;
    int step = category + 4;
    int curIndex=0;
    int topkBoxsCount = 0;
    struct box* topkBoxs = (struct box *)boxBuf;
    for (int i = 0; i < batch_size; i++, topkBoxs+=topk) { // batch size
        /* LDRA_EXCLUDE 271 S */
        /* LDRA_EXCLUDE_START 9 S */
        for (int box_count = 0; box_count < box_num; box_count++, curOutputPtr += step, curIndex += step) {
            int maxPosition = get_max_index_float32(curOutputPtr, category);
            float curConfidence = curOutputPtr[maxPosition];
            // 排除置信度小于阈值的目标
            if (curConfidence <= obj_threshold) {
                continue;
            }
            // 300个目标已满 并且最小的置信度 大于当前置信度 则抛弃
            /* LDRA_EXCLUDE 692 S */
            if (topkBoxsCount == topk && curConfidence < topkBoxs[topkBoxsCount - 1].confidence) {
                continue;
            }
            struct box curBox = {curConfidence, curIndex + maxPosition + 4};
            // 从最后一个位置开始 或者299开始
            for (int j = topkBoxsCount; j >= 0; --j) {
                // 限制在299
                if (j > (topk - 1)) {
                    continue;
                }
                // 将298 移动到299
                if (j == 0) {
                    topkBoxs[j] = curBox;
                    break;
                }
                topkBoxs[j] = topkBoxs[j-1];
                // 如果299 大于 当前置信度 则将当前的目标插入299 并退出
                if (topkBoxs[j].confidence > curConfidence) {
                    topkBoxs[j] = curBox;
                    break;
                }
            }
            // 如果目标数小于300 则记录目标数
            if (topkBoxsCount < topk) {
                topkBoxsCount++;
            }
        }
        /* LDRA_EXCLUDE_END  9 S */
        int selectBoxIndex = i * topk * 6;
        for (int j = 0; j < topkBoxsCount; ++j) {
            if(step != 0){
                int exectLocat = topkBoxs[j].index;
                int locat = (int)(exectLocat / step) * step;
                int n = (exectLocat % step) - 4;
                if (is_xywh) {
                    selectBox[selectBoxIndex] = real_type_output[locat] -  real_type_output[locat + 2] * 0.5;
                    selectBox[selectBoxIndex + 1] = real_type_output[locat + 1] -  real_type_output[locat + 3] * 0.5;
                    selectBox[selectBoxIndex + 2] = real_type_output[locat] +  real_type_output[locat + 2] * 0.5;
                    selectBox[selectBoxIndex + 3] = real_type_output[locat + 1] +  real_type_output[locat + 3] * 0.5;
                } else {
                    selectBox[selectBoxIndex] = real_type_output[locat];          // x1
                    selectBox[selectBoxIndex + 1] = real_type_output[locat + 1];  // y1
                    selectBox[selectBoxIndex + 2] = real_type_output[locat + 2];  // x2
                    selectBox[selectBoxIndex + 3] = real_type_output[locat + 3];  // y2
                }
                selectBox[selectBoxIndex + 4] = real_type_output[exectLocat]; // confidence
                selectBox[selectBoxIndex + 5] = (float)n;                  // class
                selectBoxIndex += 6;
            }
        }
    }
}

#define VGETQ_LANE_F16(INDEX) \
    val = vgetq_lane_f16(final_max_vec, (INDEX)); \
    if (val > max_value) { \
        max_value = val; \
        max_index = vgetq_lane_u16(final_max_index_vec, (INDEX)); \
    }

static int get_max_index(const uint16_t* ori_ptr, int size) {
    if (size <= 0) {
        return -1;
    }
    int max_index = 0;
#ifdef SIMULATOR
    for (int i = 0; i < size; ++i) {
        if (data_convert_fp16_to_fp32(ori_ptr[i]) > data_convert_fp16_to_fp32(ori_ptr[max_index])) {
            max_index = i;
        }
    }
#else
    float16_t* arr = (float16_t *)ori_ptr;
    // 每次处理8个数
    int i;
    float16x8_t max_vec1 = vdupq_n_f16(arr[0]);
    float16x8_t max_vec2 = vdupq_n_f16(arr[0]);
    uint16x8_t index_vec1 = {0, 1, 2, 3, 4, 5, 6, 7};
    uint16x8_t index_vec2 = {8,9,10,11,12,13,14,15};
    uint16x8_t max_index_vec1 = vdupq_n_u16(0);
    uint16x8_t max_index_vec2 = vdupq_n_u16(0);

    for (i = 0; i <= (size - 16); ) {
        float16x8_t current1 = vld1q_f16(&arr[i]);
        float16x8_t current2 = vld1q_f16(&arr[i+8]);

        uint16x8_t mask1 = vcgtq_f16(current1, max_vec1);
        uint16x8_t mask2 = vcgtq_f16(current2, max_vec2);

        max_vec1 = vbslq_f16(mask1, current1, max_vec1);
        max_vec2 = vbslq_f16(mask2, current2, max_vec2);

        max_index_vec1 = vbslq_u16(mask1, index_vec1, max_index_vec1);
        max_index_vec2 = vbslq_u16(mask2, index_vec2, max_index_vec2);

        index_vec1 = vaddq_u16(index_vec1, vdupq_n_u16(16));
        index_vec2 = vaddq_u16(index_vec2, vdupq_n_u16(16));
        i += 16;
    }

    // 比较两个向量找出全局最大值
    uint16x8_t mask = vcgtq_f16(max_vec2, max_vec1);
    float16x8_t final_max_vec = vbslq_f16(mask, max_vec2, max_vec1);
    uint16x8_t final_max_index_vec = vbslq_u16(mask, max_index_vec2, max_index_vec1);

    // 找出向量中的最大值
    float max_value = -FLT_MAX, val = -FLT_MAX;
    VGETQ_LANE_F16(0)
    VGETQ_LANE_F16(1)
    VGETQ_LANE_F16(2)
    VGETQ_LANE_F16(3)
    VGETQ_LANE_F16(4)
    VGETQ_LANE_F16(5)
    VGETQ_LANE_F16(6)
    VGETQ_LANE_F16(7)

    // 处理剩余元素
    for (; i < size; i++) {
        if (arr[i] > max_value) {
            max_value = arr[i];
            max_index = i;
        }
    }
#endif
    return max_index;
}

/*
    1 * 8400 * 84
    topk 1 * topk  xywhcn 6 小于config的排除
    84*8400 -> 8400*84
*/
void yolov10_postProcess_float16(void* output, float* selectBox, float obj_threshold,
        int batch_size, int box_num, int category, void* boxBuf, int topk, bool is_xywh) {
    uint16_t* real_type_output = (uint16_t *)output;
    transpose_nobubble_float16(real_type_output, 84, 8400); // 顺序要求
    #ifndef SIMULATOR
        float16_t* curOutputPtr = (float16_t*)real_type_output + 4;
        float16_t* output_f16 = (float16_t*)real_type_output;
    #else
        uint16_t* curOutputPtr = real_type_output + 4;
    #endif
    int step = category + 4;
    int curIndex=0;
    int topkBoxsCount = 0;
    struct box* topkBoxs = (struct box *)boxBuf;
    for (int i = 0; i < batch_size; i++, topkBoxs+=topk) { // batch size
        /* LDRA_EXCLUDE 271 S */
        /* LDRA_EXCLUDE_START 9 S */
        for (int box_count = 0; box_count < box_num; box_count++, curOutputPtr += step, curIndex += step) {
            int maxPosition = get_max_index((uint16_t*)curOutputPtr, category);

            #ifndef SIMULATOR
                float curConfidence = curOutputPtr[maxPosition];
            #else
                float curConfidence = data_convert_fp16_to_fp32(curOutputPtr[maxPosition]);
            #endif
            // 排除置信度小于阈值的目标
            if (curConfidence <= obj_threshold) {
                continue;
            }
            // 300个目标已满 并且最小的置信度 大于当前置信度 则抛弃
            /* LDRA_EXCLUDE 692 S */
            if (topkBoxsCount == topk && curConfidence < topkBoxs[topkBoxsCount - 1].confidence) {
                continue;
            }
            struct box curBox = {curConfidence, curIndex + maxPosition + 4};
            // 从最后一个位置开始 或者299开始
            for (int j = topkBoxsCount; j >= 0; --j) {
                // 限制在299
                if (j > (topk - 1)) {
                    continue;
                }
                // 将298 移动到299
                if (j == 0) {
                    topkBoxs[j] = curBox;
                    break;
                }
                topkBoxs[j] = topkBoxs[j-1];
                // 如果299 大于 当前置信度 则将当前的目标插入299 并退出
                if (topkBoxs[j].confidence > curConfidence) {
                    topkBoxs[j] = curBox;
                    break;
                }
            }
            // 如果目标数小于300 则记录目标数
            if (topkBoxsCount < topk) {
                topkBoxsCount++;
            }
        }
        /* LDRA_EXCLUDE_END  9 S */
        int selectBoxIndex = i * topk * 6;
        for (int j = 0; j < topkBoxsCount; ++j) {
            if(step != 0){
                int exectLocat = topkBoxs[j].index;
                int locat = (int)(exectLocat / step) * step;
                int n = (exectLocat % step) - 4;
                #ifndef SIMULATOR
                    float x0 = output_f16[locat];
                    float x1 = output_f16[locat + 1];
                    float x2 = output_f16[locat + 2];
                    float x3 = output_f16[locat + 3];
                    float x4 = output_f16[exectLocat];
                #else
                    float x0 = data_convert_fp16_to_fp32(real_type_output[locat]);
                    float x1 = data_convert_fp16_to_fp32(real_type_output[locat + 1]);
                    float x2 = data_convert_fp16_to_fp32(real_type_output[locat + 2]);
                    float x3 = data_convert_fp16_to_fp32(real_type_output[locat + 3]);
                    float x4 = data_convert_fp16_to_fp32(real_type_output[exectLocat]);
                #endif
                if (is_xywh) {
                    selectBox[selectBoxIndex] = x0 -  x2 * 0.5;
                    selectBox[selectBoxIndex + 1] = x1 -  x3 * 0.5;
                    selectBox[selectBoxIndex + 2] = x0 +  x2 * 0.5;
                    selectBox[selectBoxIndex + 3] = x1 +  x3 * 0.5;
                } else {
                    selectBox[selectBoxIndex] = x0;      // x1
                    selectBox[selectBoxIndex + 1] = x1;  // y1
                    selectBox[selectBoxIndex + 2] = x2;  // x2
                    selectBox[selectBoxIndex + 3] = x3;  // y2
                }
                selectBox[selectBoxIndex + 4] = x4;       // confidence
                selectBox[selectBoxIndex + 5] = (float)n;                  // class
                selectBoxIndex += 6;
            }
        }
    }
}


ot_void sample_npu_yolov7_nms(sample_npu_context_t *context)
{
    ot_u32 *valid_flag;
    if (context->out_bbox_num <= 0U) {
        printf("out_bbox_num: %d\n", context->out_bbox_num);
        return;
    }
    valid_flag = (ot_u32 *)malloc(sizeof(ot_u32) * context->out_bbox_num);
    if (valid_flag == NULL) {
        printf("malloc fail.\n");
        return;
    }
    memset(valid_flag, 0, sizeof(ot_u32) * context->out_bbox_num);
    for (uint32_t i = 0U ; i < context->out_bbox_num; i++) {
        valid_flag[i] = 1U;
    }
    // 按照confidence从高到底排序
    qsort(&context->out_bbox, context->out_bbox_num, sizeof(sample_avp_npu_rect), bbox_compare);

    float iou_thresh = 0.45;
    for (uint32_t i = 0U; i < context->out_bbox_num; ++i) {
        if (!valid_flag[i]) {
            continue;
        }
        sample_avp_npu_rect *bbox1 = &context->out_bbox[i];

        for (uint32_t j = i + 1U; j < context->out_bbox_num; ++j) {
            sample_avp_npu_rect *bbox2 = &context->out_bbox[j];
            if (!valid_flag[j]) {
                continue;
            }
            if (iou_compute(bbox1, bbox2) > iou_thresh) {
                valid_flag[j] = 0U;
            }
        }
    }
    for (uint32_t i = 0U; i < context->out_bbox_num; i++) {
        if (valid_flag[i] == 0U) {
            continue;
        }
        sample_avp_npu_rect *bbox = &context->out_bbox[i];

        bbox->x1 = MAX(bbox->x1, 0);
        bbox->y1 = MAX(bbox->y1, 0);
        bbox->x2 = MAX(bbox->x2, 0);
        bbox->y2 = MAX(bbox->y2, 0);
        bbox->x1 = MIN(bbox->x1, 640);
        bbox->y1 = MIN(bbox->y1, 640);
        bbox->x2 = MIN(bbox->x2, 640);
        bbox->y2 = MIN(bbox->y2, 640);

        printf("class_id is: %d, conf is: %f, rect[x1,y1,x2,y2] is: [%f, %f, %f, %f]\n",
            bbox->class_id, bbox->conf, bbox->x1, bbox->y1, bbox->x2, bbox->y2);
        context->out_valid_bbox_num++;
    }
    free(valid_flag);
}
// 输入为按照conf排序的 xywhcn数据
uint32_t sample_npu_yolo_nms(uint32_t out_bbox_num, float* inferData, float iou_thresh, float conf)
{
    uint32_t out_valid_bbox_num = 0U;
    sample_avp_npu_rect *out_bbox = (sample_avp_npu_rect *)inferData;
    for (uint32_t i = 0U; i < out_bbox_num; ++i) {
        if (out_bbox[i].conf < conf) {
            continue;
        }

        for (uint32_t j = i + 1U; j < out_bbox_num; ++j) {
            if (out_bbox[j].conf < conf) {
                continue;
            }
            if (iou_compute(&out_bbox[i], &out_bbox[j]) > iou_thresh) {
                out_bbox[j].conf = 0;
            }
        }
    }
    for (uint32_t i = 0U; i < out_bbox_num; i++) {
        if (out_bbox[i].conf < conf) {
            continue;
        }
        sample_avp_npu_rect *bbox = &out_bbox[i];

        bbox->x1 = MAX(bbox->x1, 0);
        bbox->y1 = MAX(bbox->y1, 0);
        bbox->x2 = MAX(bbox->x2, 0);
        bbox->y2 = MAX(bbox->y2, 0);
        bbox->x1 = MIN(bbox->x1, 640);
        bbox->y1 = MIN(bbox->y1, 640);
        bbox->x2 = MIN(bbox->x2, 640);
        bbox->y2 = MIN(bbox->y2, 640);
        float *class_ptr = (float *)bbox;

        printf("class_id is: %f, conf is: %f, rect[x1,y1,x2,y2] is: [%f, %f, %f, %f]\n",
            class_ptr[5], bbox->conf, bbox->x1, bbox->y1, bbox->x2, bbox->y2);
        out_valid_bbox_num++;
    }
    printf("detect object num: %d\n", out_valid_bbox_num);
    return out_valid_bbox_num;
}

// 反向 sigmoid（logit）函数
float_t inverse_sigmoid(float_t y) {
    if (y <= 0.0 || y >= 1.0) {
        printf("logit input err: %f", y);
    }
    return log(y / (1.0 - y));
}
//  [1, 255, w, h] -> [1, 3, w, h, 85]
ot_void sample_npu_yolov7_decode_model_float16_output_data(sample_npu_context_t *context, ot_u8* buf, int node_index,
    ot_ulong buf_size, ot_avp_npu_shape *dims, ot_ulong stride)
{
    SAMPLE_AVP_CHECK_RETURN_WITH_LOG(buf == OT_NULL, RETURN_NONE, "buf is null.\n");

    printf("{%d %d %d %d} %ld\n", dims->dims[0], dims->dims[1], dims->dims[2], dims->dims[3], buf_size);

    float anchors[3][6] = {
        {10,  13, 16,  30,  33, 23},
        {30,  61, 62,  45,  59, 119},
        {116, 90, 156, 198, 373, 326}
    };

    int channel = 3;
    int height = dims->dims[2];
    int width = dims->dims[3];
    float obj_threshold = 0.1;
    int n_offset = height * (int)stride / 2;
    for (int c = 0; c < channel; c++) {
        int offset_c = c * 85 * n_offset;
        for (int h = 0; h < height; h++) {
            int offset_h = h * (int)stride / 2;
            for (int w = 0; w < width; w++) {
                #ifndef SIMULATOR
                float16_t *ptr = (float16_t *)buf + offset_c + w + offset_h;
                float_t t_x = ptr[0 * n_offset];
                float_t t_y = ptr[1 * n_offset];
                float_t t_w = ptr[2 * n_offset];
                float_t t_h = ptr[3 * n_offset];
                float_t t_c = ptr[4 * n_offset];
                #else
                int16_t *ptr = (int16_t *)buf + offset_c + w + offset_h;
                float_t t_x = data_convert_fp16_to_fp32(ptr[0 * n_offset]);
                float_t t_y = data_convert_fp16_to_fp32(ptr[1 * n_offset]);
                float_t t_w = data_convert_fp16_to_fp32(ptr[2 * n_offset]);
                float_t t_h = data_convert_fp16_to_fp32(ptr[3 * n_offset]);
                float_t t_c = data_convert_fp16_to_fp32(ptr[4 * n_offset]);
                #endif
                if (t_c < obj_threshold) {
                    continue;
                }
                t_x = (t_x * 2.0f - 0.5 + w) * 640 / width;   // center x
                t_y = (t_y * 2.0f - 0.5 + h) * 640 / height;  // center y
                t_w = pow(t_w * 2.f, 2) * anchors[node_index][c * 2];     // cal w
                t_h = pow(t_h * 2.f, 2) * anchors[node_index][c * 2 + 1]; // cal h

                float tmp_score = 0.0;
                float tmp_id = 0;
                for (int i = 5; i < 85; i++) {
                    #ifndef SIMULATOR
                    float ptrF32data = ptr[i * n_offset];
                    #else
                    float ptrF32data = data_convert_fp16_to_fp32(ptr[i * n_offset]);
                    #endif
                    if (tmp_score < ptrF32data) {
                        tmp_score = ptrF32data;
                        tmp_id = i - 5;
                    }
                }
                tmp_score = tmp_score * t_c;
                if (tmp_score > 0.4) {
                    // 过滤出符合的锚框
                    sample_avp_npu_rect *rect = &context->out_bbox[context->out_bbox_num];
                    rect->x1 = t_x - t_w / 2;
                    rect->y1 = t_y - t_h / 2;
                    rect->x2 = t_x + t_w / 2;
                    rect->y2 = t_y + t_h / 2;
                    rect->conf = tmp_score;
                    rect->class_id = tmp_id;

                    context->out_bbox_num++;
                }
            }
        }
    }
}


//  [1, 255, w, h] -> [1, 3, w, h, 85]
ot_void sample_npu_yolov7_decode_model_float32_output_data(sample_npu_context_t *context, ot_u8* buf, int node_index,
    ot_ulong buf_size, ot_avp_npu_shape *dims, ot_ulong stride)
{
    SAMPLE_AVP_CHECK_RETURN_WITH_LOG(buf == OT_NULL, RETURN_NONE, "buf is null.\n");

    printf("{%d %d %d %d} %ld\n", dims->dims[0], dims->dims[1], dims->dims[2], dims->dims[3], buf_size);

    float anchors[3][6] = {
        {10,  13, 16,  30,  33, 23},
        {30,  61, 62,  45,  59, 119},
        {116, 90, 156, 198, 373, 326}
    };

    int channel = 3;
    int height = dims->dims[2];
    int width = dims->dims[3];
    float obj_threshold = 0.1;
    int n_offset = height * (int)stride / 4;
    for (int c = 0; c < channel; c++) {
        int offset_c = c * 85 * n_offset;
        for (int h = 0; h < height; h++) {
            int offset_h = h * (int)stride / 4;
            for (int w = 0; w < width; w++) {
                float *ptr = (float *)buf + offset_c + w + offset_h;
                float t_x = ptr[0 * n_offset];
                float t_y = ptr[1 * n_offset];
                float t_w = ptr[2 * n_offset];
                float t_h = ptr[3 * n_offset];
                float t_c = ptr[4 * n_offset];

                if (t_c < obj_threshold) {
                    continue;
                }
                t_x = (t_x * 2.0f - 0.5 + w) * 640 / width;   // center x
                t_y = (t_y * 2.0f - 0.5 + h) * 640 / height;  // center y
                t_w = pow(t_w * 2.f, 2) * anchors[node_index][c * 2];     // cal w
                t_h = pow(t_h * 2.f, 2) * anchors[node_index][c * 2 + 1]; // cal h

                float tmp_score = ptr[0];
                float tmp_id = 0;
                for (int i = 5; i < 85; i++) {
                    if (tmp_score < ptr[i * n_offset]) {
                        tmp_score = ptr[i * n_offset];
                        tmp_id = i - 5;
                    }
                }
                tmp_score = tmp_score * t_c;
                if (tmp_score > 0.4) {
                    // 过滤出符合的锚框
                    sample_avp_npu_rect *rect = &context->out_bbox[context->out_bbox_num];
                    rect->x1 = t_x - t_w / 2;
                    rect->y1 = t_y - t_h / 2;
                    rect->x2 = t_x + t_w / 2;
                    rect->y2 = t_y + t_h / 2;
                    rect->conf = tmp_score;
                    rect->class_id = tmp_id;

                    context->out_bbox_num++;
                }
            }
        }
    }
}