#include "sample_center_point_bbox_coder.h"
#include "sample_bev_nms.h"
#include <unordered_set>
#include <numeric>
#include <stdio.h>
#include <dirent.h>
#include <string.h>

#define SAMPLE_BBOX_TOPK_NUM 500

static float sigmoid(float x) {
    return 1.0 / (1.0 + std::exp(-x));
}

std::vector<DecoderBBox> center_point_bbox_decode_topk_scores(float *heatmap)
{
    int dim[4] = {1, 10, 128, 128};

    std::vector<DecoderBBox> topk_final_scores;
#if SAMPLE_BEV_DEBUG
    std::vector<std::vector<DecoderValueIndex>> cls_topk_scores(10);
#endif
    std::priority_queue<DecoderValueIndex> topk_scores_pri_queue;
    // 1x10x128x128 -> 1x10x500
    for (int i = 0; i < dim[0]; i++) {
        int offset_dim1 = i * dim[1] * dim[2] * dim[3];
        for (int j = 0; j < dim[1]; j++) {
            int offset_dim2 = offset_dim1 + j * dim[2] * dim[3];
            std::priority_queue<DecoderValueIndex> pq;
            for (int k = 0; k < dim[2] * dim[3]; k++) {
                int offset_dim3 = offset_dim2 + k;
                DecoderValueIndex val;
                val.feat_id = j;
                val.score = sigmoid(heatmap[offset_dim3]);
                val.index = k;
                val.x = k % 128;
                val.y = k / 128;
                pq.push(val);
            }
            for (int k = 0; k < SAMPLE_BBOX_TOPK_NUM; k++) {
                DecoderValueIndex top = pq.top();
                pq.pop();
                topk_scores_pri_queue.push(top);
#if SAMPLE_BEV_DEBUG
                cls_topk_scores[j].push_back(top);
#endif
            }
 #if SAMPLE_BEV_DEBUG
            for (int ii = 0; ii < 10; ii++) {
                printf("cls[%d],x[%d],y[%d],score[%.4f],index[%d],\n",
                    cls_topk_scores[j][ii].feat_id,
                    cls_topk_scores[j][ii].x, cls_topk_scores[j][ii].y,
                    cls_topk_scores[j][ii].score, cls_topk_scores[j][ii].index);
            }
#endif
        }
    }

    // 1x10x500 -> 1x500
    for (int i = 0; i < SAMPLE_BBOX_TOPK_NUM; i++) {
        DecoderValueIndex top = topk_scores_pri_queue.top();
        topk_scores_pri_queue.pop();
        DecoderBBox bbox;
        bbox.x       = top.x;
        bbox.y       = top.y;
        bbox.feat_id = top.feat_id;
        bbox.scores  = top.score;
        bbox.index   = top.index;
        topk_final_scores.push_back(bbox);
    }
#if SAMPLE_BEV_DEBUG
    printf("decode_topk_scores : %lld\n", topk_final_scores.size());
    for (int i = 0; i < 10; i++) {
        printf("x[%.4f],y[%.4f],z[%.4f],score[%.4f],index[%d],cls[%d]\n",
        topk_final_scores[i].x, topk_final_scores[i].y, topk_final_scores[i].z,
        topk_final_scores[i].scores, topk_final_scores[i].index, topk_final_scores[i].feat_id);
    }
#endif
    return topk_final_scores;
}

void center_point_bbox_decode_fill_data(std::vector<DecoderBBox>& bboxs,
    float* reg, float* height, float* dims, float* rot, float* vel)
{
    float voxel_size[3]        = {0.1, 0.1, 0.2};
    float point_cloud_range[6] = {-51.2, -51.2, -5.0, 51.2, 51.2, 3.0}; // pc_range
    float out_size_factor = 8;
    for (auto& bbox: bboxs) {
        // decode reg {1, 2, 128, 128};
        int x_offset = 128 * 128 * 0 + 128 * bbox.y + bbox.x;
        int y_offset = 128 * 128 * 1 + 128 * bbox.y + bbox.x;
        // decode heihgt {1, 1, 128, 128};
        int z_offset = 128 * bbox.y + bbox.x;
        // decode dims {1, 3, 128, 128}
        int l_offset = 128 * 128 * 0 + 128 * bbox.y + bbox.x;
        int w_offset = 128 * 128 * 1 + 128 * bbox.y + bbox.x;
        int h_offset = 128 * 128 * 2 + 128 * bbox.y + bbox.x;
        // decode rot {1, 2, 128, 128}
        int rots_offset = 128 * 128 * 0 + 128 * bbox.y + bbox.x;
        int rotc_offset = 128 * 128 * 1 + 128 * bbox.y + bbox.x;
        // decode vel {1, 2, 128, 128}
        int vx_offset = 128 * 128 * 0 + 128 * bbox.y + bbox.x;
        int vy_offset = 128 * 128 * 1 + 128 * bbox.y + bbox.x;

        bbox.x = (bbox.x + reg[x_offset]) * out_size_factor * voxel_size[0] + point_cloud_range[0];
        bbox.y = (bbox.y + reg[y_offset]) * out_size_factor * voxel_size[1] + point_cloud_range[1];
        bbox.z = height[z_offset];
        bbox.l = std::exp(dims[l_offset]);
        bbox.w = std::exp(dims[w_offset]);
        bbox.h = std::exp(dims[h_offset]);
        bbox.rot = std::atan2(rot[rots_offset], rot[rotc_offset]);
        bbox.vx = vel[vx_offset];
        bbox.vy = vel[vy_offset];
    }
#if SAMPLE_BEV_DEBUG
    printf("decode_fill_data : %ld \n", bboxs.size());
    for (int i = 0; i < 10; i++) {
        printf("x[%.4f],y[%.4f],z[%.4f],l[%.4f],w[%.4f],h[%.4f],rot[%.4f],vx[%.4f],vy[%.4f],score[%.4f],index[%d],cls[%d]\n",
        bboxs[i].x, bboxs[i].y, bboxs[i].z, bboxs[i].l, bboxs[i].w, bboxs[i].h,bboxs[i].rot, bboxs[i].vx, bboxs[i].vy,
        bboxs[i].scores, bboxs[i].index, bboxs[i].feat_id);
    }
#endif
}

void center_point_bbox_decode_scores_thresh(std::vector<DecoderBBox>& bboxs)
{
    float post_center_range[6] = {-61.2, -61.2, -10.0, 61.2, 61.2, 10.0};
    float score_threshold = 0.1;

    float x_min = post_center_range[0];
    float y_min = post_center_range[1];
    float z_min = post_center_range[2];
    float x_max = post_center_range[3];
    float y_max = post_center_range[4];
    float z_max = post_center_range[5];

    for (auto it = bboxs.begin(); it != bboxs.end(); ) {
        if (it->scores < score_threshold ||
            it->x < x_min || it->y < y_min || it->z < z_min ||
            it->x > x_max || it->y > y_max || it->z > z_max) {
            it = bboxs.erase(it);
        } else {
            ++it;
        }
    }
#if SAMPLE_BEV_DEBUG
    printf("decode_scores_thresh : %ld \n", bboxs.size());
    for (int i = 0; i < 10; i++) {
        printf("x[%.4f],y[%.4f],z[%.4f],l[%.4f],w[%.4f],h[%.4f],rot[%.4f],vx[%.4f],vy[%.4f],score[%.4f],index[%d],cls[%d]\n",
        bboxs[i].x, bboxs[i].y, bboxs[i].z, bboxs[i].l, bboxs[i].w, bboxs[i].h,bboxs[i].rot, bboxs[i].vx, bboxs[i].vy,
        bboxs[i].scores, bboxs[i].index, bboxs[i].feat_id);
    }
#endif
}

void center_point_bbox_decode_expand(std::vector<DecoderBBox>& bboxs)
{
    float nms_rescale_factor[10] = {
        1.0, 0.7, 0.7, 0.4, 0.55,
        1.1, 1.0, 1.0, 1.5, 3.5
    };
    for (auto& bbox: bboxs) {
        float factor = nms_rescale_factor[bbox.feat_id];
        bbox.l = bbox.l * factor;
        bbox.w = bbox.w * factor;
        bbox.h = bbox.h * factor;
    }
#if SAMPLE_BEV_DEBUG
    printf("decode_expand : %ld \n", bboxs.size());
    for (int i = 0; i < (int)bboxs.size(); i++) {
        printf("x[%.4f],y[%.4f],z[%.4f],l[%.4f],w[%.4f],h[%.4f],rot[%.4f],vx[%.4f],vy[%.4f],score[%.4f],index[%d],cls[%d]\n",
        bboxs[i].x, bboxs[i].y, bboxs[i].z, bboxs[i].l, bboxs[i].w, bboxs[i].h,bboxs[i].rot, bboxs[i].vx, bboxs[i].vy,
        bboxs[i].scores, bboxs[i].index, bboxs[i].feat_id);
    }
#endif
}

void center_point_bbox_decode_contract(std::vector<DecoderBBox>& bboxs)
{
    float nms_rescale_factor[10] = {
        1.0, 0.7, 0.7, 0.4, 0.55,
        1.1, 1.0, 1.0, 1.5, 3.5
    };
    for (auto& bbox: bboxs) {
        float factor = nms_rescale_factor[bbox.feat_id];
        bbox.l = bbox.l / factor;
        bbox.w = bbox.w / factor;
        bbox.h = bbox.h / factor;
    }
#if SAMPLE_BEV_DEBUG
    printf("decode_contract : %ld \n", bboxs.size());
    for (int i = 0; i < 10; i++) {
        printf("x[%.4f],y[%.4f],z[%.4f],l[%.4f],w[%.4f],h[%.4f],rot[%.4f],vx[%.4f],vy[%.4f],score[%.4f],index[%d],cls[%d]\n",
        bboxs[i].x, bboxs[i].y, bboxs[i].z, bboxs[i].l, bboxs[i].w, bboxs[i].h,bboxs[i].rot, bboxs[i].vx, bboxs[i].vy,
        bboxs[i].scores, bboxs[i].index, bboxs[i].feat_id);
    }
#endif
}

double calc_iou_by_opencv(cv::RotatedRect rect1, cv::RotatedRect rect2)
{
    double areaRect1 = (double)rect1.size.width * (double)rect1.size.height;
    double areaRect2 = (double)rect2.size.width * (double)rect2.size.height;
    std::vector<cv::Point2f> vertices;
    (void)cv::rotatedRectangleIntersection(rect1, rect2, vertices);
    if (vertices.size()==0) {
        return 0.0;
    }

    std::vector<cv::Point2f> order_pts;
    cv::convexHull(cv::Mat(vertices), order_pts, true);
    double area = cv::contourArea(order_pts);
    double inner = (double)(area / (areaRect1 + areaRect2 - area) + 1e-6f); //0.0001);
    return inner;
}

void center_point_bbox_decode_nms_by_opencv(std::vector<DecoderBBox>& bboxs, std::vector<DecoderBBox>& output_bboxs)
{
    // fill cv::RotatedRect
    for (auto & bbox : bboxs) {
        cv::Point2f center(bbox.x, bbox.y);
        cv::Size2f size(bbox.w, bbox.h);
        // float angle = bbox.rot;
        float angle = bbox.rot * 180/CV_PI;
        bbox.rbox = cv::RotatedRect(center, size, angle);
    }

    double iou_threshold = 0.2;
    // bboxs 已经按照scores降序排序
    while (bboxs.size() > 0) {
        int bbox_num = bboxs.size();
        output_bboxs.push_back(bboxs[0]);
        for (int i = 0; i < bbox_num - 1; i++) {
            double iou_val = calc_iou_by_opencv(bboxs[0].rbox, bboxs[i + 1].rbox);
            if (iou_val > iou_threshold) {
                bboxs.erase(bboxs.begin() + i + 1);
            }
        }
        bboxs.erase(bboxs.begin());
    }

    printf("decode_nms : %ld \n", output_bboxs.size());
    // FIXME: nms结果和python代码有出入
#if SAMPLE_BEV_DEBUG
    for (int i = 0; i < (int)output_bboxs.size(); i++) {
        printf("x[%.4f],y[%.4f],z[%.4f],l[%.4f],w[%.4f],h[%.4f],rot[%.4f],vx[%.4f],vy[%.4f],score[%.4f],index[%d],cls[%d]\n",
        output_bboxs[i].x, output_bboxs[i].y, output_bboxs[i].z, output_bboxs[i].l, output_bboxs[i].w, output_bboxs[i].h,output_bboxs[i].rot, output_bboxs[i].vx, output_bboxs[i].vy,
        output_bboxs[i].scores, output_bboxs[i].index, output_bboxs[i].feat_id);
    }
#endif
}

void center_point_bbox_decode_nms(std::vector<DecoderBBox>& bboxes, std::vector<DecoderBBox>& output_bboxs)
{
    std::unordered_set<int> suppressed;
    double iou_threshold = 0.2;
    // bboxs 已经按照scores降序排序
    for (size_t i = 0UL; i < bboxes.size(); ++i) {
        if (suppressed.count(i)) continue;
        output_bboxs.push_back(bboxes[i]);

        for (size_t j = i + 1UL; j < bboxes.size(); ++j) {
            if (iou_bev(bboxes[i], bboxes[j]) > iou_threshold) {
                suppressed.insert(j); // IoU 过高，抑制该框
            }
        }
    }

    printf("decode_nms : %ld \n", output_bboxs.size());
#if SAMPLE_BEV_DEBUG
    for (int i = 0; i < (int)output_bboxs.size(); i++) {
        printf("x[%.4f],y[%.4f],z[%.4f],l[%.4f],w[%.4f],h[%.4f],rot[%.4f],vx[%.4f],vy[%.4f],score[%.4f],index[%d],cls[%d]\n",
        output_bboxs[i].x, output_bboxs[i].y, output_bboxs[i].z, output_bboxs[i].l, output_bboxs[i].w, output_bboxs[i].h,output_bboxs[i].rot, output_bboxs[i].vx, output_bboxs[i].vy,
        output_bboxs[i].scores, output_bboxs[i].index, output_bboxs[i].feat_id);
    }
#endif
}