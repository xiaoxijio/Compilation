#ifndef __SAMPLE_UTILS_CLASSIFY_UTILS_H
#define __SAMPLE_UTILS_CLASSIFY_UTILS_H
#include <stdint.h>
#include "ot_avp_npu_common.h"
#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
typedef struct {
    float score;
    uint32_t class;
} score_with_class;

void softmax(float* input, float* logits, uint32_t num_classes);

int32_t compute_and_display_topk(float* output_data, uint32_t num_classes, uint32_t k);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
#endif
