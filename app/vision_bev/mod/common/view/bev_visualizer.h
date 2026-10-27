#pragma once
#include "../bev_data.h"

#ifdef __cplusplus
#if __cplusplus
extern "C"
{
#endif
#endif
    class BevVisualizer
    {
    public:
        void DrawCanvaAndImgs(BevData &bev_data);
    };
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif