

#include "sample_avp_npu_common.h"
#include <vector>
#include <map>
#include <math.h>
#include <cmath>  // for fabs
#include <limits>

constexpr float obj_threshold = 0.25f;
constexpr float iou_threshold = 0.5f;
constexpr int MAX_OBJECTS = 100;

struct object_info {
    float x1, y1, x2, y2;
    float conf = std::numeric_limits<float>::lowest();
    int class_id;      // 目标的类别
    int position;      // 在特征图上的位置
    int ouptut_index;  // 使用的步长索引(0,1,2)
    float area;        // 计算面积用于nms
};

//process for yolov8 and yolov10
class Yolo_process{
public:
    Yolo_process(std::vector<uint16_t*> conf_data, std::vector<uint16_t*> cor_data) : conf_data_(conf_data), cor_data_(cor_data){}
    void get_valid_object();
    void get_valid_object(int index);
    void update_valid_object();
    void insert_object(float conf, int position, int class_id, int index);
    void NMS();
    void prinf_box();
    void pre_job() {
        obj_infos = std::vector<object_info>(MAX_OBJECTS);
        softMax_max = std::vector<float>(16, std::numeric_limits<float>::lowest());
        softMax_sum = std::vector<float>(16, 0);
        class_buckets.clear();
    }

private:
    std::vector<uint16_t*> conf_data_;
    std::vector<uint16_t*> cor_data_;
    std::vector<float> softMax_max;
    std::vector<float> softMax_sum;
    std::vector<int> w = {80, 40, 20};
    std::vector<object_info> obj_infos;
    std::map<int, std::vector<int>> class_buckets;
    int obj_count = 0;
};
ot_s32 sample_yolo_execute(sample_avp_model_task_info_t* model_info, const char* input_path);
void yolov10_postProcess_float32(void* output, float* selectBox, float obj_threshold,
        int batch_size, int box_num, int category, void* boxBuf, int topk, bool is_xywh);
void yolov10_postProcess_float16(void* output, float* selectBox, float obj_threshold,
        int batch_size, int box_num, int category, void* boxBuf, int topk, bool is_xywh);
uint32_t sample_npu_yolo_nms(uint32_t out_bbox_num, float* inferData, float iou_thresh, float conf);
ot_void sample_npu_yolov7_nms(sample_npu_context_t *context);

ot_void sample_npu_yolov7_decode_model_float16_output_data(sample_npu_context_t *context, ot_u8* buf, int node_index,
    ot_ulong buf_size, ot_avp_npu_shape *dims, ot_ulong stride);
ot_void sample_npu_yolov7_decode_model_float32_output_data(sample_npu_context_t *context, ot_u8* buf, int node_index,
    ot_ulong buf_size, ot_avp_npu_shape *dims, ot_ulong stride);