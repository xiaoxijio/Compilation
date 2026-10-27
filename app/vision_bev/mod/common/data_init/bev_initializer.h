#pragma once
#include "../bev_data.h"

#ifdef __cplusplus
#if __cplusplus
extern "C"
{
#endif
#endif

    class BevInitializer
    {
    public:
        void Init(BevData &bevData, std::string folder_path, int index, bool npu_out_from_file = true);
        void ParseCalibrationDictsJson(BevData &bevData);
        void ParseGtBoxesJson(BevData &bevData);
        void ParseAnnInfosJson(BevData &bevData);
    };

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif