#ifndef SIMULATOR_UTILS_H
#define SIMULATOR_UTILS_H
#ifdef SIMULATOR

#include <stdint.h>

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

int32_t sample_avp_npu_malloc(void** ptr, uint64_t size);
int32_t sample_avp_npu_free(void* ptr);
int32_t sample_avp_npu_media_system_init(void);
int32_t sample_avp_npu_media_system_deinit(void);
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
#endif
#endif