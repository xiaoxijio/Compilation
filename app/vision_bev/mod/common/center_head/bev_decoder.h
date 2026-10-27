#pragma once
#include "../bev_data.h"


#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

class BevDecoder {
public:
    void DecodeBBox(BevData& bev_data);
};

std::string get_files_with_suffix(const char *folder_path, const char *suffix);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif