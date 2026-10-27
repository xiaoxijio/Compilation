
#include "sample_bev_nms.h"
#include "sample_center_point_bbox_coder.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <numeric>

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

// **计算矩形的 4 个角点**
void get_rotated_vertices(const DecoderBBox &box, std::vector<std::pair<float, float>> &vertices) {
    float cos_theta = std::cos(box.rot);
    float sin_theta = std::sin(box.rot);
    float half_l = box.l / 2;
    float half_w = box.w / 2;

    // 计算 4 个角点
    vertices.push_back({box.x + half_l * cos_theta - half_w * sin_theta, box.y + half_l * sin_theta + half_w * cos_theta});
    vertices.push_back({box.x - half_l * cos_theta - half_w * sin_theta, box.y - half_l * sin_theta + half_w * cos_theta});
    vertices.push_back({box.x - half_l * cos_theta + half_w * sin_theta, box.y - half_l * sin_theta - half_w * cos_theta});
    vertices.push_back({box.x + half_l * cos_theta + half_w * sin_theta, box.y + half_l * sin_theta - half_w * cos_theta});
}

// **计算凸包面积（Graham 扫描法）**
float polygon_area(std::vector<std::pair<float, float>> &points) {
    if (points.size() < 3) return 0.0f; // 至少需要 3 个点
    std::sort(points.begin(), points.end());

    std::vector<std::pair<float, float>> hull;
    for (const auto &p : points) {
        while (hull.size() > 1 && (hull[hull.size() - 1].first - hull[hull.size() - 2].first) *
                                      (p.second - hull[hull.size() - 2].second) -
                                      (hull[hull.size() - 1].second - hull[hull.size() - 2].second) *
                                      (p.first - hull[hull.size() - 2].first) <= 0) {
            hull.pop_back();
        }
        hull.push_back(p);
    }

    long unsigned int lower_hull_size = hull.size();
    for (int i = points.size() - 2; i >= 0; --i) {
        while (hull.size() > lower_hull_size && (hull[hull.size() - 1].first - hull[hull.size() - 2].first) *
                                                   (points[i].second - hull[hull.size() - 2].second) -
                                                   (hull[hull.size() - 1].second - hull[hull.size() - 2].second) *
                                                   (points[i].first - hull[hull.size() - 2].first) <= 0) {
            hull.pop_back();
        }
        hull.push_back(points[i]);
    }
    hull.pop_back(); // 去除最后一个重复点

    float area = 0.0;
    for (size_t i = 0UL; i < hull.size(); i++) {
        size_t j = (i + 1UL) % hull.size();
        area += (hull[i].first * hull[j].second - hull[j].first * hull[i].second);
    }
    return std::fabs(area) / 2.0;
}

// 凸多边形求交集（Sutherland-Hodgman 算法）
// 计算点 p 在直线 ab 的左侧还是右侧
float crossProduct(const std::pair<float, float>& a, const std::pair<float, float>& b, const std::pair<float, float>& p) {
    return (b.first - a.first) * (p.second - a.second) - (b.second - a.second) * (p.first - a.first);
}
// 计算线段 (p1, p2) 与 (a, b) 的交点
std::pair<float, float> computeIntersection(const std::pair<float, float>& p1, const std::pair<float, float>& p2, const std::pair<float, float>& a, const std::pair<float, float>& b) {
    float A1 = p2.second - p1.second;
    float B1 = p1.first - p2.first;
    float C1 = A1 * p1.first + B1 * p1.second;

    float A2 = b.second - a.second;
    float B2 = a.first - b.first;
    float C2 = A2 * a.first + B2 * a.second;

    float det = A1 * B2 - A2 * B1;
    if (det == 0) return p1; // 平行，不可能发生（因为输入是凸多边形）

    return {(B2 * C1 - B1 * C2) / det, (A1 * C2 - A2 * C1) / det};
}

// Sutherland-Hodgman 算法裁剪多边形
std::vector<std::pair<float, float>> polygonIntersection(std::vector<std::pair<float, float>> subjectPolygon, const std::vector<std::pair<float, float>>& clipPolygon) {
    std::vector<std::pair<float, float>> outputList = subjectPolygon;

    for (size_t i = 0UL; i < clipPolygon.size(); ++i) {
        std::vector<std::pair<float, float>> inputList = outputList;
        outputList.clear();

        std::pair<float, float> A = clipPolygon[i];
        std::pair<float, float> B = clipPolygon[(i + 1UL) % clipPolygon.size()]; // 下一个顶点

        for (size_t j = 0UL; j < inputList.size(); ++j) {
            std::pair<float, float> P = inputList[j];
            std::pair<float, float> Q = inputList[(j + 1UL) % inputList.size()];

            bool P_inside = crossProduct(A, B, P) >= 0;
            bool Q_inside = crossProduct(A, B, Q) >= 0;

            if (P_inside && Q_inside) {
                outputList.push_back(Q); // Q 在内部，直接保留
            } else if (P_inside && !Q_inside) {
                outputList.push_back(computeIntersection(P, Q, A, B)); // P 在内 Q 在外，添加交点
            } else if (!P_inside && Q_inside) {
                outputList.push_back(computeIntersection(P, Q, A, B)); // P 在外 Q 在内，添加交点
                outputList.push_back(Q);
            }
        }
    }
    return outputList;
}
float computePolygonArea(const std::vector<std::pair<float, float>>& poly) {
    if (poly.size() < 3) return 0.0;  // 少于3点无面积

    float area = 0.0;
    for (size_t i = 0UL; i < poly.size(); i++) {
        size_t j = (i + 1UL) % poly.size();
        area += poly[i].first * poly[j].second - poly[j].first * poly[i].second;
    }
    return fabs(area) / 2.0;
}

// **计算 IoU（交并比）**
float iou_bev(const DecoderBBox &box_a, const DecoderBBox &box_b) {
    std::vector<std::pair<float, float>> corners_a, corners_b;
    get_rotated_vertices(box_a, corners_a);
    get_rotated_vertices(box_b, corners_b);

    // 计算各自面积
    float area_a = box_a.l * box_a.w;
    float area_b = box_b.l * box_b.w;

    // // 计算交集面积（凸包求交）
    // std::vector<std::pair<float, float>> intersection_pts = corners_a;
    // intersection_pts.insert(intersection_pts.end(), corners_b.begin(), corners_b.end());
    // float inter_area = polygon_area(intersection_pts);

    // 法1：计算凸多边形交集面积（不调用接口）
    std::vector<std::pair<float, float>> intersection_pts = polygonIntersection(corners_a, corners_b);
    float inter_area = computePolygonArea(intersection_pts);

    // // 法2：转换为 OpenCV 的 Point2f 格式，采用 OpenCV 接口计算凸多边形交集面积
    // std::vector<cv::Point2f> contour_a, contour_b;
    // for (const auto &p : corners_a)
    //     contour_a.emplace_back(p.first, p.second);
    // for (const auto &p : corners_b)
    //     contour_b.emplace_back(p.first, p.second);
    // std::vector<cv::Point2f> intersection_contour;
    // float inter_area = cv::intersectConvexConvex(contour_a, contour_b, intersection_contour);

    return inter_area / (area_a + area_b - inter_area + 1e-6f); // 避免除零
}

void print_topk_scores(const std::vector<DecoderBBox>& topk_scores) {
    int num_print = std::min(10, static_cast<int>(topk_scores.size())); // 只打印前 10 个

    for (int i = 0; i < num_print; i++) {
        const auto& bbox = topk_scores[i];
        std::cout << "Index: " << i << " | x: " << bbox.x
                  << " y: " << bbox.y
                  << " z: " << bbox.z
                  << " l: " << bbox.l
                  << " w: " << bbox.w
                  << " h: " << bbox.h
                  << " rot: " << bbox.rot
                  << " vx: " << bbox.vx
                  << " vy: " << bbox.vy
                  << " feat_id: " << bbox.feat_id
                  << " scores: " << bbox.scores
                  << " index: " << bbox.index
                  << std::endl;
    }
}




#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif