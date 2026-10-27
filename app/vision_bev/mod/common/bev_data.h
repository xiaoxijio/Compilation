#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>


#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

typedef struct _DecoderBBox {
    float x = 0.0;
    float y = 0.0;
    float z = 0.0;
    float l = 0.0;
    float w = 0.0;
    float h = 0.0;
    float rot = 0.0;
    float vx = 0.0;
    float vy = 0.0;

    int feat_id = 0;
    float scores = 0.0;
    int index = 0;

    cv::RotatedRect rbox;     // 用于旋转矩形的NMS
    cv::Point2f vertices[4];  // 存储旋转矩形的4个顶点

    std::vector<cv::Point3f> corners;   // 存储立体框的八个定点
} DecoderBBox;

typedef struct _SampleBevPreparedData {
    std::string folder_path;
    std::string img[6];
    std::string ann_infos;
    std::string calibration_dicts;
    std::string gt_boxes;
    std::string lidar_pcd;
    std::string npu_in_bin;
    std::string npu_out_bin[6];
    std::string bev_final_out_img;
    char* buffer[6];
} SampleBevPreparedData;

struct BevData
{
    SampleBevPreparedData m_prepared_data;
    std::vector<DecoderBBox> m_nms_bboxs;
    std::vector<DecoderBBox> m_lidar_gt_boxes;
    bool m_npu_out_no_free = false;
    bool m_debug_flag = false;
    bool m_enable_opencv_nms = false;
    bool m_draw_gt = true;

    cv::Mat m_imgs[6];
    cv::Mat m_canvas;

    std::string m_views[6];
    std::vector<std::vector<double>> m_calibration_dicts_lidar2camera[6]; // {4, 4}
    std::vector<std::vector<double>> m_calibration_dicts_camera2img[6];   // {4, 4}
    std::vector<std::vector<double>> m_ego2global;
    std::vector<std::vector<double>> m_lidar2ego;
    std::vector<std::vector<double>> global2lidar;
    std::vector<std::vector<double>> m_ann_infos_bbox; // {n, 9}
    std::vector<double> m_ann_infos_cls;               // {n}
    // std::vector<std::vector<double>>    m_gt_boxes;       // {n, 7}
};


#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif