#include "classify_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int cmp_score_with_class(const void * elem1, const void * elem2) 
{
    score_with_class lhs = *((score_with_class*)elem1);
    score_with_class rhs = *((score_with_class*)elem2);
    if (lhs.score < rhs.score) {
        return 1;
    }
    if (lhs.score > rhs.score) {
        return -1;
    }
    return 0;
}

void softmax(float* input, float* logits, uint32_t num_classes)
{
    float denominator = 0.0f;
    for (uint32_t idx = 0; idx < num_classes; idx++) {
        denominator += expf(input[idx]);
    }
    for (uint32_t idx = 0; idx < num_classes; idx++) {
        if (denominator != 0.0f) {
            logits[idx] = expf(input[idx]) / denominator;
        }
    }
}

int32_t compute_and_display_topk(float* output_data, uint32_t num_classes, uint32_t k)
{
    if (k > num_classes) {
        printf("invalid param %d > %d\n", k, num_classes);
        return -1;
    }
/* LDRA_EXCLUDE 72 D */
    softmax(output_data, output_data, num_classes);
    score_with_class* score_with_classes = (score_with_class*)malloc(sizeof(score_with_class) * num_classes);
    if (score_with_classes  == NULL) {
        printf("malloc failed\n");
        return -1;
    }
    for (uint32_t idx = 0; idx < num_classes; idx++) {
        score_with_classes[idx].score = output_data[idx];
        score_with_classes[idx].class = idx;
    }
    qsort(score_with_classes, num_classes, sizeof(score_with_class), cmp_score_with_class);
    printf("class with scores:\n");
    for (uint32_t idx = 0; idx < k; idx++) {
        printf("class %5d, score %.6f\n", score_with_classes[idx].class, score_with_classes[idx].score);
    }
    free(score_with_classes);
    return 0;
}
