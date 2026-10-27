#ifdef SIMULATOR
#include "simulator_utils.h"
#include <stdlib.h>
#include <string.h>

int32_t sample_avp_npu_malloc(void **ptr, uint64_t size)
{
    *ptr = malloc(size);
    if (*ptr == NULL) {
        return -1;
    }
    return 0;
}

int32_t sample_avp_npu_free(void *ptr)
{
    free(ptr);
    return 0;
}

int32_t sample_avp_npu_media_system_init(void)
{
    return 0;
}

int32_t sample_avp_npu_media_system_deinit(void)
{
    return 0;
}
#endif
