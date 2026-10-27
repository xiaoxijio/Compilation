#ifndef SAMPLE_BEV_NMS_H
#define SAMPLE_BEV_NMS_H

#include "sample_center_point_bbox_coder.h"
#include <ratio>
#include <vector>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>


#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

float iou_bev(const DecoderBBox &box_a, const DecoderBBox &box_b);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif