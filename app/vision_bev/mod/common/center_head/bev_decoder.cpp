#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "json.hpp"
#include "bev_decoder.h"
#include "../data_init/bev_initializer.h"
#include "../view/bev_visualizer.h"
#include "sample_center_point_bbox_coder.h"

void BevDecoder::DecodeBBox(BevData& bev_data)
{
    float* reg       = (float *)bev_data.m_prepared_data.buffer[0];
    float* height    = (float *)bev_data.m_prepared_data.buffer[1];
    float* dims      = (float *)bev_data.m_prepared_data.buffer[2];
    float* rot       = (float *)bev_data.m_prepared_data.buffer[3];
    float* vel       = (float *)bev_data.m_prepared_data.buffer[4];
    float* featmap   = (float *)bev_data.m_prepared_data.buffer[5];

    // decode heatmap {1, 10, 128, 128};
    std::vector<DecoderBBox> topk_scores = center_point_bbox_decode_topk_scores(featmap);

    // decode reg, height, dims, rot, vel
    center_point_bbox_decode_fill_data(topk_scores, reg, height, dims, rot, vel);

    // score_thresh过滤, post_center_range过滤
    center_point_bbox_decode_scores_thresh(topk_scores);
    // expand
    center_point_bbox_decode_expand(topk_scores);

    // nms
    if (bev_data.m_enable_opencv_nms) {
        center_point_bbox_decode_nms_by_opencv(topk_scores, bev_data.m_nms_bboxs);
    } else {
        center_point_bbox_decode_nms(topk_scores, bev_data.m_nms_bboxs);
    }
    // contract
    center_point_bbox_decode_contract(bev_data.m_nms_bboxs);

    if (bev_data.m_debug_flag) {
        int num_bboxs = bev_data.m_nms_bboxs.size();
        printf("bbox_num: %d\n", num_bboxs);
        for (int i = 0; i < num_bboxs && i < 10; i++) {
            printf("x[%.3f],y[%.3f],z[%.3f],l[%.3f],w[%.3f],h[%.3f],rot[%.3f]\n",
                bev_data.m_nms_bboxs[i].x, bev_data.m_nms_bboxs[i].y, bev_data.m_nms_bboxs[i].z, bev_data.m_nms_bboxs[i].l, bev_data.m_nms_bboxs[i].w, bev_data.m_nms_bboxs[i].h, bev_data.m_nms_bboxs[i].rot);
        }
    }
}