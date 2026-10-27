#ifndef CENTER_POINT_BBOX_CODER_H
#define CENTER_POINT_BBOX_CODER_H

#include <ratio>
#include <vector>
#include <queue>
#include <cmath>

#include "../bev_data.h"

#define SAMPLE_BEV_DEBUG  0
#define SAMPLE_BEV_OPENCV 0

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

// 定义一个结构体来存储值和索引
struct DecoderValueIndex {
    int feat_id = 0;
    float score = 0.0;
    int index = 0;
    int x = 0;
    int y = 0;
    // 重载小于运算符，用于优先队列的比较
    bool operator<(const DecoderValueIndex& other) const {
        return score <= other.score;
    }
};
std::vector<DecoderBBox> center_point_bbox_decode_topk_scores(float *heatmap);
void center_point_bbox_decode_fill_data(std::vector<DecoderBBox>& bboxs,
    float* reg, float* height, float* dims, float* rot, float* vel);
void center_point_bbox_decode_scores_thresh(std::vector<DecoderBBox>& bboxs);
void center_point_bbox_decode_expand(std::vector<DecoderBBox>& bboxs);
void center_point_bbox_decode_contract(std::vector<DecoderBBox>& bboxs);

void center_point_bbox_decode_nms(std::vector<DecoderBBox>& bboxes, std::vector<DecoderBBox>& output_bboxs);
void center_point_bbox_decode_nms_by_opencv(std::vector<DecoderBBox>& bboxs, std::vector<DecoderBBox>& output_bboxs);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif