#include "bev_visualizer.h"
#include "../center_head/bev_decoder.h"
#include "../data_init/bev_initializer.h"
#include <opencv2/opencv.hpp>
#include "../center_head/sample_center_point_bbox_coder.h"
#include "json.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>


int g_canva_size = 1000;
int g_show_range = 50;
float g_class_thred[10] = {
    0.5,    // car
    0.5,    // truck
    0.5,    // bus
    0.5,    // pedestrian
    0.5,    // motorcycle
    0.5,    // traffic_cone
    0.5,    // barrier
    0.5,    // trailer
    0.5,    // construction_vehicle
    0.5,    // bicycle
};
std::vector<cv::Scalar> colors = {
    cv::Scalar(0, 255, 0),    // 绿色 (index 0)
    cv::Scalar(0, 255, 255)   // 黄色 (index 1) （OpenCV是BGR格式）
};

cv::Point trans_to_lidar_coordinates_to_canva(float x, float y)
{
    int canva_size = g_canva_size;
    int show_range = g_show_range;

    int v = static_cast<int>((-y + show_range) / show_range / 2.0 * canva_size);
    int u = static_cast<int>((x + show_range) / show_range / 2.0 * canva_size);

    return cv::Point(u, v);  // 返回图像坐标 (u, v)
}

// 计算 3D 旋转框的 8 个角点坐标(ego -> Lidar坐标系)。
std::vector<cv::Point3f> sample_bev_get_lidar_corners(
    const DecoderBBox& bbox,
    std::vector<std::vector<double>>& m_lidar2ego,
    const bool gt_flag){

    std::vector<cv::Point3f> corners(8);  // 存储 8 个角点

    // 计算半尺寸
    float half_l = bbox.l / 2.0f;
    float half_w = bbox.w / 2.0f;
    float half_h = bbox.h / 2.0f;

    // 局部坐标 (绕中心点)
    std::vector<cv::Point3f> local_corners = {
        {-half_l, -half_w, -half_h},  // 左后下 (x0, y0, z0)
        {-half_l,  half_w, -half_h},  // 左前下 (x0, y1, z0)
        {half_l,  half_w, -half_h},   // 右前下 (x1, y1, z0)
        {half_l, -half_w, -half_h},   // 右后下 (x1, y0, z0)

        {-half_l, -half_w,  half_h},  // 左后上 (x0, y0, z1)
        {-half_l,  half_w,  half_h},  // 左前上 (x0, y1, z1)
        {half_l,  half_w,  half_h},   // 右前上 (x1, y1, z1)
        {half_l, -half_w,  half_h},   // 右后上 (x1, y0, z1)
    };

    // 计算绕 z 轴的旋转矩阵 (yaw 旋转)
    float cos_yaw = std::cos(bbox.rot);
    float sin_yaw = std::sin(bbox.rot);

    cv::Mat rotation = (cv::Mat_<float>(3, 3) <<
        cos_yaw, -sin_yaw, 0,
        sin_yaw,  cos_yaw, 0,
        0,        0,       1);


    // 获得 ego2lidar
    cv::Mat lidar2ego_mat = (cv::Mat_<double>(4, 4) <<
        m_lidar2ego[0][0], m_lidar2ego[0][1], m_lidar2ego[0][2], m_lidar2ego[0][3],
        m_lidar2ego[1][0], m_lidar2ego[1][1], m_lidar2ego[1][2], m_lidar2ego[1][3],
        m_lidar2ego[2][0], m_lidar2ego[2][1], m_lidar2ego[2][2], m_lidar2ego[2][3],
        m_lidar2ego[3][0], m_lidar2ego[3][1], m_lidar2ego[3][2], m_lidar2ego[3][3]);

    cv::Mat ego2lidar_mat;
    cv::invert(lidar2ego_mat, ego2lidar_mat, cv::DECOMP_LU); // LU 分解求逆

    // 旋转 + 平移， @ego2lidar 从 ego 转换到 lidar 坐标
    for (int i = 0; i < 8; i++) {
        cv::Mat point = (cv::Mat_<float>(3, 1) <<
            local_corners[i].x,
            local_corners[i].y,
            local_corners[i].z);

        cv::Mat rotated_point = rotation * point;

        if (gt_flag) {
            corners[i] = cv::Point3f(
                rotated_point.at<float>(0, 0) + bbox.x,
                rotated_point.at<float>(1, 0) + bbox.y,
                rotated_point.at<float>(2, 0) + bbox.z
            );
        }else{

            cv::Mat corners_mat = ego2lidar_mat * (cv::Mat_<double>(4, 1) <<
                rotated_point.at<float>(0, 0) + bbox.x,
                rotated_point.at<float>(1, 0) + bbox.y,
                rotated_point.at<float>(2, 0) + bbox.z,
                1.0);
            corners[i] = cv::Point3f(
                corners_mat.at<double>(0, 0),
                corners_mat.at<double>(1, 0),
                corners_mat.at<double>(2, 0));
        }
    }

    return corners;
}

std::vector<bool> checkPointInImg(const std::vector<cv::Point3f>& points, int height, int width, const std::vector<bool>& valid_init) {
    std::vector<bool> valid_c;

    bool is_valid = false;
    for (int j = 0; j < 8; ++j) {
        const cv::Point3f& pt = points[j];  // 访问 8 个点中的每一个
        bool valid = (pt.x >= 0 && pt.y >= 0 && pt.x < width && pt.y < height);
        if (valid) {
            is_valid = true;
        }
    }
    // 结合 valid_init 进行按位与操作
    for (int j = 0; j < 8; ++j) {
        bool final_valid = is_valid;
        if (!valid_init.empty() && j < (int)valid_init.size()) {
            final_valid = final_valid && valid_init[j];  // 按位与操作
        }
        valid_c.push_back(final_valid);
    }
    return valid_c;
}

// C++ 实现 lidar 点到图像坐标的投影，仿照 Python lidar2img 函数
std::vector<cv::Point3f> lidar2img(
    const DecoderBBox& bbox,
    std::vector<std::vector<double>>& lidar2camera,
    std::vector<std::vector<double>>& camera2img,
    std::vector<bool>& valid_mask_out) {

    int N = 8;

    // 1. 构建 lidar 点的齐次坐标
    cv::Mat points_lidar_homogeneous(N, 4, CV_64F);
    for (int i = 0; i < N; ++i) {
        points_lidar_homogeneous.at<double>(i, 0) = bbox.corners[i].x;
        points_lidar_homogeneous.at<double>(i, 1) = bbox.corners[i].y;
        points_lidar_homogeneous.at<double>(i, 2) = bbox.corners[i].z;
        points_lidar_homogeneous.at<double>(i, 3) = 1.0;
    }

    // 2. 构建 camera2lidar（4x4）并计算其逆矩阵 lidar2camera
    cv::Mat lidar2camera_mat = (cv::Mat_<double>(4, 4) <<
        lidar2camera[0][0], lidar2camera[0][1], lidar2camera[0][2], lidar2camera[0][3],
        lidar2camera[1][0], lidar2camera[1][1], lidar2camera[1][2], lidar2camera[1][3],
        lidar2camera[2][0], lidar2camera[2][1], lidar2camera[2][2], lidar2camera[2][3],
        lidar2camera[3][0], lidar2camera[3][1], lidar2camera[3][2], lidar2camera[3][3]);

    // 3. 点从 lidar 转换到 camera 坐标系
    cv::Mat points_camera_homogeneous = points_lidar_homogeneous * lidar2camera_mat.t();

    // 4. 保留 Z > 0 的点作为有效点，并归一化坐标
    std::vector<cv::Point3f> points_camera(N);
    for (int i = 0; i < N; ++i) {
        double X = points_camera_homogeneous.at<double>(i, 0);
        double Y = points_camera_homogeneous.at<double>(i, 1);
        double Z = points_camera_homogeneous.at<double>(i, 2);

        if (Z > 0.5) {
            valid_mask_out[i] = true;
            points_camera[i] = cv::Point3f(X / Z, Y / Z, 1.0);
        } else {
            points_camera[i] = cv::Point3f(0, 0, 0); // 默认填 0
        }
    }

    // 5. 构建相机内参矩阵 camera2img
    cv::Mat camera2img_mat = (cv::Mat_<double>(3, 3) <<
        camera2img[0][0], camera2img[0][1], camera2img[0][2],
        camera2img[1][0], camera2img[1][1], camera2img[1][2],
        camera2img[2][0], camera2img[2][1], camera2img[2][2]);
    // 6. 投影到图像平面
    std::vector<cv::Point3f> points_img(N);
    for (int i = 0; i < N; ++i) {
        if (valid_mask_out[i]) {
            double x = points_camera[i].x;
            double y = points_camera[i].y;
            double z = 1.0;
            cv::Mat uv = camera2img_mat * (cv::Mat_<double>(3, 1) << x, y, z);
            points_img[i] = cv::Point3f(uv.at<double>(0, 0), uv.at<double>(1, 0), uv.at<double>(2, 0));
        } else {
            points_img[i] = cv::Point3f(-1, -1, -1); // 无效点可设为 -1
        }
    }

    return points_img;
}

void DrawImgs(BevData& bev_data, std::vector<DecoderBBox>& vis_bboxes, bool gt_flag)
{
    float width  = 1600;
    float height = 900;
    std::vector<std::pair<int, int>> draw_boxes_indexes_img_view = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5},
        {5, 6}, {6, 7}, {7, 4}, {0, 4}, {1, 5},
        {2, 6}, {3, 7}
    };
    int cnt = 0;
    for (int i = 0; i < 6; i++) {
        for (auto& bbox: vis_bboxes) {
            if (bbox.scores < g_class_thred[bbox.feat_id]) { continue; }

            std::vector<bool> valid_mask_out;
            valid_mask_out.resize(8, false);
            // std::vector<cv::Point3f> corners = sample_bev_get_img_corners(bbox, m_calibration_dicts_lidar2camera[i], m_calibration_dicts_camera2img[i]);
            std::vector<cv::Point3f> corners = lidar2img(bbox, bev_data.m_calibration_dicts_lidar2camera[i], bev_data.m_calibration_dicts_camera2img[i], valid_mask_out);
            // printf("## x[%0.4f], y[%0.4f], z[%0.4f] \n", corners[0].x, corners[0].y, corners[0].z);
            cv::Point pts[8];
            for (int i = 0; i < 8; ++i) {
                pts[i].x = corners[i].x;
                pts[i].y = corners[i].y;
            }

            std::vector<bool> valid = checkPointInImg(corners, height, width, valid_mask_out);
            for(int v = 0; v < (int)draw_boxes_indexes_img_view.size(); ++v){
                std::pair<int, int> index = draw_boxes_indexes_img_view[v];
                if(valid[index.first] && valid[index.second]){
                    cv::line(bev_data.m_imgs[i], pts[index.first], pts[index.second], colors[gt_flag], 3);
                }
            }
            cnt += 1;
        }
    }
    printf("imgs cnt: %d \n", cnt);
}

void DrawCanva(BevData& bev_data, std::vector<DecoderBBox>& vis_bboxes, bool gt_flag)
{
    int thickness = 2;
    int cnt = 0;
    for (auto& bbox: vis_bboxes) {
        if (bbox.scores < g_class_thred[bbox.feat_id]) {
            continue;
        }
        // 获取旋转矩形的8个顶点
        std::vector<cv::Point3f> corners = bbox.corners;
        cnt += 1;

        // 绘制旋转矩形
        cv::Point pts[4];
        for (int i = 0; i < 4; ++i) {
            pts[i] = trans_to_lidar_coordinates_to_canva(corners[i].x, corners[i].y);
        }
        cv::line(bev_data.m_canvas, pts[0], pts[1], colors[gt_flag], thickness);
        cv::line(bev_data.m_canvas, pts[1], pts[2], colors[gt_flag], thickness);
        cv::line(bev_data.m_canvas, pts[2], pts[3], colors[gt_flag], thickness);
        cv::line(bev_data.m_canvas, pts[3], pts[0], colors[gt_flag], thickness);

        // 绘制方向线
        // cv::Point center_pts = trans_to_lidar_coordinates_to_canva((corners[2].x + corners[3].x) / 2, (corners[2].y + corners[3].y) / 2);
        // cv::Point head_center_pts = trans_to_lidar_coordinates_to_canva(bbox.x, bbox.y);
        cv::Point head_center_pts = trans_to_lidar_coordinates_to_canva((corners[2].x + corners[3].x) / 2, (corners[2].y + corners[3].y) / 2);
        cv::Point center_pts = trans_to_lidar_coordinates_to_canva((corners[0].x + corners[2].x) / 2, (corners[0].y + corners[2].y) / 2);
        cv::line(bev_data.m_canvas, center_pts, head_center_pts, colors[gt_flag], thickness);
    }
    printf("cavas cnt: %d \n", cnt);
}

void sample_bev_read_binary_file(const char *filename, char **out_buffer, int *out_size)
{
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        perror("无法打开文件");
        return;
    }
    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char *buffer = (char *)malloc(file_size);
    if (buffer == NULL) {
        perror("内存分配失败");
        fclose(file);
        return;
    }
    size_t bytes_read = fread(buffer, 1, file_size, file);
    if (bytes_read != file_size) {
        perror("读取文件失败");
    }
    fclose(file);

    *out_buffer = buffer;
    *out_size   = file_size;
}

void ConcatCanvaAndImgs(BevData& bev_data)
{
    cv::Mat hConcatResult[2], vConcatResult;

    std::vector<cv::Mat> mats = {bev_data.m_imgs[0], bev_data.m_imgs[1], bev_data.m_imgs[2]};
    cv::hconcat(mats, hConcatResult[0]);

    // mats = {m_imgs[3], m_imgs[4], m_imgs[5]};
    // 左右翻转（水平翻转，flipCode = 1）
    cv::Mat flippedImg3, flippedImg5;
    flip(bev_data.m_imgs[3], flippedImg3, 1);
    flip(bev_data.m_imgs[5], flippedImg5, 1);
    mats = {flippedImg3, bev_data.m_imgs[4], flippedImg5};
    cv::hconcat(mats, hConcatResult[1]);

    mats = {hConcatResult[0], hConcatResult[1]};
    cv::vconcat(mats, vConcatResult);

    cv::Mat resizeCanva;
    cv::resize(bev_data.m_canvas, resizeCanva, cv::Size(1800, 1800), 0, 0, cv::INTER_LINEAR);

    cv::Mat finalCanva;
    mats = {vConcatResult, resizeCanva};
    cv::hconcat(mats, finalCanva);

    cv::imwrite(bev_data.m_prepared_data.bev_final_out_img, finalCanva);
}

cv::Mat sample_bev_create_lidar_points(std::string &lidar_top)
{
    // 创建一个的bev画布，背景为黑色
    int canva_size = g_canva_size;
    cv::Mat canvas = cv::Mat::zeros(canva_size, canva_size, CV_8UC3);

    // 画lidar points
    float *lidar_points    = NULL;
    int lidar_points_size = 0;
    sample_bev_read_binary_file(lidar_top.c_str(), (char **)&lidar_points, &lidar_points_size);
    lidar_points_size = lidar_points_size / sizeof(float);
    // lidar point {n, 5};
    int lidar_points_num = lidar_points_size / 5;
    int show_range = g_show_range;
    for (int i = 0; i < lidar_points_num; i++) {
        // {x, y, z, ?, ?}
        int offset_x = i * 5 + 0;
        int offset_y = i * 5 + 1;
        lidar_points[offset_y] = -lidar_points[offset_y];
        lidar_points[offset_x] = (lidar_points[offset_x] + show_range) / show_range / 2.0 * canva_size;
        lidar_points[offset_y] = (lidar_points[offset_y] + show_range) / show_range / 2.0 * canva_size;
    }

    for (int i = 0; i < lidar_points_num; i++) {
        int offset_x = i * 5 + 0;
        int offset_y = i * 5 + 1;
        cv::Point point(lidar_points[offset_x], lidar_points[offset_y]);
        int radius = 0;
        cv::Scalar color(255, 255, 255);
        int thickness = 1;
        cv::circle(canvas, point, radius, color, thickness);
    }

    free(lidar_points);

    return canvas;
}


void BevVisualizer::DrawCanvaAndImgs(BevData& bev_data)
{
    bev_data.m_canvas = sample_bev_create_lidar_points(bev_data.m_prepared_data.lidar_pcd);

    printf("final bbox_num:[%ld]\n", bev_data.m_nms_bboxs.size());

    // 计算旋转矩形的8个顶点
    for (auto& bbox: bev_data.m_nms_bboxs) {
        if (bbox.scores < g_class_thred[bbox.feat_id]) {
            continue;
        }
        bbox.corners = sample_bev_get_lidar_corners(bbox, bev_data.m_lidar2ego, false);
    }

    DrawCanva(bev_data, bev_data.m_nms_bboxs, false);
    DrawImgs(bev_data, bev_data.m_nms_bboxs, false);

    if(bev_data.m_draw_gt){
        // 获取GT的lidar corners
        for (auto& bbox: bev_data.m_lidar_gt_boxes) {
            bbox.corners = sample_bev_get_lidar_corners(bbox, bev_data.m_lidar2ego, true);
        }
        printf("gt bbox_num: %ld \n", bev_data.m_lidar_gt_boxes.size());
        DrawCanva(bev_data, bev_data.m_lidar_gt_boxes, true);
        DrawImgs(bev_data, bev_data.m_lidar_gt_boxes, true);
    }
    ConcatCanvaAndImgs(bev_data);
}

cv::Point trans_to_img_coordinates_to_canva(float x, float y)
{
    int v = static_cast<int>((-x));
    int u = static_cast<int>((-y));

    return cv::Point(u, v);  // 返回图像坐标 (u, v)
}


void Print2DMat(const cv::Mat& mat)
{
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            std::cout << static_cast<double>(mat.at<double>(i, j)) << " ";
        }
        std::cout << std::endl;
    }
}
void PrintPoint3f(const std::vector<cv::Point3f>& points)
{
    for (auto& p: points) {
        std::cout << p.x << " " << p.y << " " << p.z << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void printMatInfo(const cv::Mat& mat, const std::string& name) {
    std::cout << "=== " << name << " ===" << std::endl;
    std::cout << "Size: " << mat.rows << " x " << mat.cols << std::endl;
    std::cout << "Type: " << mat.type() << std::endl;
    std::cout << "Values:\n" << mat << std::endl << std::endl;
}


// 计算 3D 旋转框的 8 个角点坐标(图像坐标系)。
std::vector<cv::Point3f> sample_bev_get_img_corners(const DecoderBBox& bbox,
    std::vector<std::vector<double>>& lidar2camera, std::vector<std::vector<double>>& camera2img)
{
    std::vector<cv::Point3f> corners(8);  // 存储 8 个角点

    cv::Mat lidar_mat = (cv::Mat_<double>(8, 4) <<
        bbox.corners[0].x, bbox.corners[0].y, bbox.corners[0].z, 1,
        bbox.corners[1].x, bbox.corners[1].y, bbox.corners[1].z, 1,
        bbox.corners[2].x, bbox.corners[2].y, bbox.corners[2].z, 1,
        bbox.corners[3].x, bbox.corners[3].y, bbox.corners[3].z, 1,
        bbox.corners[4].x, bbox.corners[4].y, bbox.corners[4].z, 1,
        bbox.corners[5].x, bbox.corners[5].y, bbox.corners[5].z, 1,
        bbox.corners[6].x, bbox.corners[6].y, bbox.corners[6].z, 1,
        bbox.corners[7].x, bbox.corners[7].y, bbox.corners[7].z, 1);
    cv::Mat lidar2camera_mat = (cv::Mat_<double>(4, 4) <<
        lidar2camera[0][0], lidar2camera[0][1], lidar2camera[0][2], lidar2camera[0][3],
        lidar2camera[1][0], lidar2camera[1][1], lidar2camera[1][2], lidar2camera[1][3],
        lidar2camera[2][0], lidar2camera[2][1], lidar2camera[2][2], lidar2camera[2][3],
        lidar2camera[3][0], lidar2camera[3][1], lidar2camera[3][2], lidar2camera[3][3]);
    cv::Mat camera2img_mat = (cv::Mat_<double>(3, 3) <<
        camera2img[0][0], camera2img[0][1], camera2img[0][2],
        camera2img[1][0], camera2img[1][1], camera2img[1][2],
        camera2img[2][0], camera2img[2][1], camera2img[2][2]);
    printMatInfo(lidar_mat, "lidar_mat");
    printMatInfo(lidar2camera_mat, "lidar2camera_mat");
    printMatInfo(camera2img_mat, "camera2img_mat");
    cv::Mat final_mat = lidar_mat * lidar2camera_mat.t() * camera2img_mat.t();

    for (int i = 0; i < 8; i++) {
        // v = x / z;
        // u = y / z;
        corners[i] = cv::Point3f(
            static_cast<double>(final_mat.at<double>(i, 1)) / static_cast<double>(final_mat.at<double>(i, 2)),
            static_cast<double>(final_mat.at<double>(i, 0)) / static_cast<double>(final_mat.at<double>(i, 2)),
            static_cast<double>(final_mat.at<double>(i, 2)));
        // std::cout << "## " <<
        //     static_cast<double>(final_mat.at<double>(i, 1)) << " " <<
        //     static_cast<double>(final_mat.at<double>(i, 0)) << " " <<
        //     static_cast<double>(final_mat.at<double>(i, 2)) << " " << std::endl;
        // std::cout << "### " << corners[i].x << " " << corners[i].y << " " << corners[i].z << " " << std::endl;
    }
    Print2DMat(lidar_mat);
    std::cout << std::endl;
    Print2DMat(lidar2camera_mat);
    std::cout << std::endl;
    Print2DMat(camera2img_mat);
    std::cout << std::endl;
    Print2DMat(final_mat);
    std::cout << std::endl;
    PrintPoint3f(corners);

    return corners;
}

void sample_bev_draw_bev_with_opencv_nms_result(std::vector<DecoderBBox>& bboxs, cv::Mat& canvas)
{
    int thickness = 2;
    for (auto& bbox: bboxs) {
        if (bbox.scores < g_class_thred[bbox.feat_id]) {
            continue;
        }
        // 计算旋转矩形的四个顶点
        bbox.rbox.points(bbox.vertices);
        // 将顶点转换为整数类型的点
        cv::Point pts[4];
        for (int i = 0; i < 4; ++i) {
            pts[i] = trans_to_lidar_coordinates_to_canva(bbox.vertices[i].x, bbox.vertices[i].y);
        }
        // 绘制旋转矩形
        cv::line(canvas, pts[0], pts[1], cv::Scalar(0, 255, 0), thickness);
        cv::line(canvas, pts[1], pts[2], cv::Scalar(0, 255, 0), thickness);
        cv::line(canvas, pts[2], pts[3], cv::Scalar(0, 255, 0), thickness);
        cv::line(canvas, pts[3], pts[0], cv::Scalar(0, 255, 0), thickness);
    }
}
