#pragma once
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <ios>
#include <stdio.h>
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>
// #include <iostream>
#include <vector>
#include "bfloat16.hpp"
#include "Tokenizer/Tokenizer.hpp"
#include "LLMEmbedSelector.hpp"
#include "cqdm.h"
#include "timer.hpp"
#include "ot_avp_npu_rts.h"
#include "sample_npu_common.h"
#include "utils/json.hpp"
#include <cstddef>
#include <unordered_map>
#include <regex>
#include <cwchar>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <stdlib.h>

namespace fs = std::filesystem;


typedef void (*LLMRuningCallback)(int *p_token, int n_token, const char *p_str, float token_per_sec, void *reserve);



void writeFloatArrayToFile(const std::string& file_path, const float* data, size_t size, int indice, int m) {
    (void)indice;
    (void)m;
    // return;
    // if (m >21)
    //  return;
    // 打开文件以写模式
    std::ofstream file(file_path);
    
    // 检查文件是否成功打开
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return;
    }

    // 写入数据，每行一个浮点数
    for (size_t i = 0; i < size; ++i) {
        file << data[i] << "\n";
    }

    // 关闭文件
    file.close();
}

void writeFp16ArrayToBinFile(const std::string& file_path, const uint16_t* data, size_t size, int indices, int m) {
    // 打开文件以二进制写模式
    //  return;
    (void)indices;
    (void)m;
    std::ofstream file(file_path, std::ios::binary);
    
    // 检查文件是否成功打开
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return;
    }

    // 写入数据
    file.write(reinterpret_cast<const char*>(data), size * sizeof(uint16_t));

    // 关闭文件
    file.close();
}
struct LLMAttrType
{
    int decoder_num;
    std::string decoder_filename;
    std::string filename_post_model;
    TokenizerType tokenizer_type;
    std::string filename_tokenizer_model;
    bool b_bos = true, b_eos = false;
    std::string filename_tokens_embed;
    int tokens_embed_num;
    int position_lenth;
    int tokens_embed_size;
    int max_token_len;
    int next_token;
    int kv_cache_num;
    int kv_cache_size;
    int file_num;
    size_t token_len;
    bool b_use_mmap_load_embed = false;
    LLMRuningCallback runing_callback = nullptr;
    void *reserve = nullptr;
};

typedef struct LLMLayer {
    model_task_info layer;
    std::string filename;
    MMap layer_buffer;
    std::vector<char> layer_buffer_vec;
} LLMLayer;

class LLM
{
    t_cqdm cqdm;
private:
    std::shared_ptr<BaseTokenizer> tokenizer;
    LLaMaEmbedSelector embed_selector;

    LLMAttrType _attr;
    std::vector<LLMLayer> llama_layers;

    model_task_info post_model;
    // std::vector<std::vector<unsigned short>> k_caches, v_caches;

    bool b_stop = false;
    // tinyllama所需
    std::vector<float> cos_cache;
    std::vector<float> sin_cache;
    std::vector<float> k_cache;
    std::vector<float> v_cache;
    std::vector<float> mask;
    std::vector<uint16_t> embed;
    void *kcache_buf;
    uint32_t kcache_buf_size;
    void *vcache_buf;
    uint32_t vcache_buf_size;
    void *kupdate_buf;
    uint32_t kupdate_buf_size;
    void *vupdate_buf;
    uint32_t vupdate_buf_size;
    //定义公共内存，为模型所需最大值。
    void* decoder_runtime_buf;
    uint32_t decoder_runtime_buf_size;
    void* post_runtime_buf;
    uint32_t post_runtime_buf_size;
public:

    // kvcache kvupdate内存共用一份
    void prepare_input_output_tensor_for_tinylama_decoder(void *handle, model_task_info* info, int i)
    {
        ot_avp_npu_get_input_num(handle, &info->input_num);
        ot_avp_npu_get_output_num(handle, &info->output_num);
        info->input_dataset = ot_avp_npu_create_input_dataset(handle);
        info->output_dataset = ot_avp_npu_create_output_dataset(handle);

        unsigned int kcache_input_idx = 0xffff;
        char kcache[56] = {0};
        sprintf(kcache, "decoder_%d_kcache", i);
        ot_avp_npu_get_input_index_by_name(handle, kcache, strlen(kcache), &kcache_input_idx);
        unsigned int vcache_input_idx = 0xffff;
        char vcache[56] = {0};
        sprintf(vcache, "decoder_%d_vcache", i);
        ot_avp_npu_get_input_index_by_name(handle, vcache, strlen(vcache), &vcache_input_idx);
        unsigned int kupdate_output_idx = 0xffff;
        char kupdate[56] = {0};
        sprintf(kupdate, "decoder_%d_kupdate", i);
        ot_avp_npu_get_output_index_by_name(handle, kupdate, strlen(kupdate), &kupdate_output_idx);
        unsigned int vupdate_output_idx = 0xffff;
        char vupdate[56] = {0};
        sprintf(vupdate, "decoder_%d_vupdate", i);
        ot_avp_npu_get_output_index_by_name(handle, vupdate, strlen(vupdate), &vupdate_output_idx);

        printf("%d %d %d %d\n", kcache_input_idx, vcache_input_idx, kupdate_output_idx, vupdate_output_idx);

        for (ot_u32 i = 0; i < info->output_num; i++) {
            ot_ulong stride = ot_avp_npu_get_output_default_stride(handle, i);
            ot_ulong buffer_size = ot_avp_npu_get_output_size_by_index(handle, i);
            ot_avp_npu_shape shape;
            ot_avp_npu_get_output_shape_by_index(handle, i, &shape);
            ot_avp_data_type data_type;
            ot_avp_npu_get_output_dtype_by_index(handle, i, &data_type);

            if (i == kupdate_output_idx && kupdate_buf != NULL && kupdate_buf_size == buffer_size) {
                ot_avp_npu_set_buffer(info->output_dataset, i, (ot_u8 *)kupdate_buf, 0, buffer_size, stride);
                info->output_tensor[i].virt_addr = (ot_u64)kupdate_buf;
            } else if (i == vupdate_output_idx && vupdate_buf != NULL && vupdate_buf_size == buffer_size) {
                ot_avp_npu_set_buffer(info->output_dataset, i, (ot_u8 *)vupdate_buf, 0, buffer_size, stride);
                info->output_tensor[i].virt_addr = (ot_u64)vupdate_buf;
            } else {
                sample_avp_npu_malloc((ot_void **)&info->output_tensor[i].virt_addr, (ot_void **)&info->output_tensor[i].phys_addr, buffer_size);
                ot_avp_npu_set_buffer(info->output_dataset, i, (ot_u8 *)info->output_tensor[i].virt_addr,
                    0, buffer_size, stride);
                if (i == kupdate_output_idx) {
                    kupdate_buf = (void *)info->output_tensor[i].virt_addr;
                    kupdate_buf_size = buffer_size;
                } else if (i == vupdate_output_idx) {
                    vupdate_buf = (void *)info->output_tensor[i].virt_addr;
                    vupdate_buf_size = buffer_size;
                }
            }
            info->output_tensor[i].len = buffer_size;
            info->output_tensor[i].shape = shape;
            info->output_tensor[i].dtype = data_type;
            info->output_tensor[i].stride.dims[0] = stride;
        }

        for (ot_u32 i = 0; i < info->input_num; i++) {
            ot_ulong stride = ot_avp_npu_get_input_default_stride(handle, i);
            ot_ulong buffer_size = ot_avp_npu_get_input_size_by_index(handle, i);
            ot_avp_npu_shape shape;
            ot_avp_npu_get_input_shape_by_index(handle, i, &shape);
            ot_avp_data_type data_type;
            ot_avp_npu_get_input_dtype_by_index(handle, i, &data_type);

            if (i == kcache_input_idx && kcache_buf != NULL && kcache_buf_size == buffer_size) {
                ot_avp_npu_set_buffer(info->input_dataset, i, (ot_u8 *)kcache_buf, 0, buffer_size, stride);
                info->input_tensor[i].virt_addr = (ot_u64)kcache_buf;
            } else if (i == vcache_input_idx && vcache_buf != NULL && vcache_buf_size == buffer_size) {
                ot_avp_npu_set_buffer(info->input_dataset, i, (ot_u8 *)vcache_buf, 0, buffer_size, stride);
                info->input_tensor[i].virt_addr = (ot_u64)vcache_buf;
            } else {
                sample_avp_npu_malloc((ot_void **)&info->input_tensor[i].virt_addr, (ot_void **)&info->input_tensor[i].phys_addr, buffer_size);
                ot_avp_npu_set_buffer(info->input_dataset, i, (ot_u8 *)info->input_tensor[i].virt_addr,
                    0, buffer_size, stride);
                if (i == kcache_input_idx) {
                    kcache_buf = (void *)info->input_tensor[i].virt_addr;
                    kcache_buf_size = buffer_size;
                } else if (i == vcache_input_idx) {
                    vcache_buf = (void *)info->input_tensor[i].virt_addr;
                    vcache_buf_size = buffer_size;
                }
            }
            info->input_tensor[i].len = buffer_size;
            info->input_tensor[i].shape = shape;
            info->input_tensor[i].dtype = data_type;
            info->input_tensor[i].stride.dims[0] = stride;
        }
    }

    void destory_input_output_tensor_for_tinylama_decoder(void *handle, model_task_info* info, int i)
    {
        unsigned int kcache_input_idx = 0xffff;
        std::string kcache = "decoder_" + std::to_string(i) + "_kcache";
        ot_avp_npu_get_input_index_by_name(handle, kcache.data(), kcache.size(), &kcache_input_idx);

        unsigned int vcache_input_idx = 0xffff;
        std::string vcache = "decoder_" + std::to_string(i) + "_vcache";
        ot_avp_npu_get_input_index_by_name(handle, vcache.data(), vcache.size(), &vcache_input_idx);

        unsigned int kupdate_output_idx = 0xffff;
        std::string kupdate = "decoder_" + std::to_string(i) + "_kupdate";
        ot_avp_npu_get_output_index_by_name(handle, kupdate.data(), kupdate.size(), &kupdate_output_idx);

        unsigned int vupdate_output_idx = 0xffff;
        std::string vupdate = "decoder_" + std::to_string(i) + "_vupdate";
        ot_avp_npu_get_output_index_by_name(handle, vupdate.data(), vupdate.size(), &vupdate_output_idx);

        ot_avp_npu_destroy_dataset(info->input_dataset);
        ot_avp_npu_destroy_dataset(info->output_dataset);
        for (ot_u32 i = 0; i < info->input_num; i++) {
            if (!(i == kcache_input_idx || i == vcache_input_idx)) {
                ot_avp_destroy_tensor(&info->input_tensor[i]);
            }
        }
        for (ot_u32 i = 0; i < info->output_num; i++) {
            if (!(i == kupdate_output_idx || i == vupdate_output_idx)) {
                ot_avp_destroy_tensor(&info->output_tensor[i]);
            }
        }
    }

    void QwenTypeInit() {
        _attr.tokens_embed_num = 151936;
        _attr.tokens_embed_size = 1536;
        _attr.position_lenth = 64;
        _attr.max_token_len = 1024;
        _attr.kv_cache_num = 56;
        _attr.decoder_num = 28;
        _attr.kv_cache_size = 256;
        _attr.file_num = 2;
        if(_attr.decoder_filename.empty())
        {
            _attr.decoder_filename = "./tinyllama-output/decoder_%d.ortm";
        }
        if(_attr.filename_tokenizer_model.empty())
        {
            _attr.filename_tokenizer_model = "tokenizer.model";
        }
        if(_attr.filename_tokens_embed.empty())
        {
            _attr.filename_tokens_embed = "tinyllama.model.embed_tokens.weight.bfloat16.bin";
        }
        if(_attr.filename_post_model.empty())
        {
            _attr.filename_post_model = "./tinyllama-output/post_model.ortm";
        }
        return; 
    }

    //对tinyllama相关的参数进行初始化
    void LLMAttrTypeInit()
    {
        _attr.tokens_embed_num = 32000;
        _attr.tokens_embed_size = 2048;
        _attr.position_lenth = 32;
        _attr.kv_cache_num = 44; // auto calc
        _attr.kv_cache_size = 256; // auto calc
        _attr.max_token_len = 1024;
        _attr.decoder_num = 22;
        _attr.file_num = 2;
        if(_attr.decoder_filename.empty())
        {
            _attr.decoder_filename = "./tinyllama-output/decoder_%d.ortm";
        }
        if(_attr.filename_tokenizer_model.empty())
        {
            _attr.filename_tokenizer_model = "tokenizer.model";
        }
        if(_attr.filename_tokens_embed.empty())
        {
            _attr.filename_tokens_embed = "tinyllama.model.embed_tokens.weight.bfloat16.bin";
        }
        if(_attr.filename_post_model.empty())
        {
            _attr.filename_post_model = "./tinyllama-output/post_model.ortm";
        }
        return;
    }

    void LoaderDecoders()
    {
        llama_layers.resize(_attr.decoder_num);
        // char buff[1024] = {0};
        char model_path[1024] = {0};
        for (int i = 0; i < _attr.decoder_num; i++)
        {
            sprintf(model_path, _attr.decoder_filename.c_str(), i);
            llama_layers[i].filename = model_path;
            LoadDecoder(i);
            update_cqdm(&cqdm, i + _attr.file_num, "count", model_path);
        }
        for (int i = 0; i < _attr.decoder_num; i++) {
            prepare_input_output_tensor(llama_layers[i].layer.handle, &llama_layers[i].layer);
        }
    }
    //根据tokennizer_type对不同模型的参数进行初始化
    void AttrTypeInit()
    {
        switch (_attr.tokenizer_type) {
            case TKT_Qwen:
                QwenTypeInit();
                break;
            case TKT_LLaMa:
                LLMAttrTypeInit();
                break;
            default:
                break;
        }
    }

    bool Init(LLMAttrType attr)
    {
        this->_attr = attr;
        //对模型参数进行初始化
        AttrTypeInit();
        std::tie(cos_cache, sin_cache) = BuildRopeCache(_attr.max_token_len,  _attr.position_lenth * 2);
        k_cache.resize(_attr.decoder_num * _attr.max_token_len * _attr.kv_cache_size, 0.0);
        v_cache.resize(_attr.decoder_num * _attr.max_token_len * _attr.kv_cache_size, 0.0);
        mask.resize(_attr.max_token_len, -std::numeric_limits<float>::infinity());
        if (_attr.decoder_num > 1) {
            this->cqdm = create_cqdm(_attr.decoder_num + _attr.file_num + 1, 32);
        } else {
            this->cqdm = create_cqdm(_attr.file_num + 1, 32);
        }
        ALOGI("Model init start");
        if (!_attr.filename_tokenizer_model.empty())
        {
            tokenizer = CreateTokenizer(attr.tokenizer_type);
            if (!tokenizer->Init(_attr.filename_tokenizer_model, _attr.b_bos, _attr.b_eos))
            {
                ALOGE("tokenizer.Init(%s, %d, %d) failed", _attr.filename_tokenizer_model.c_str(), _attr.b_bos, _attr.b_eos);
                return false;
            }
            update_cqdm(&cqdm, 0, "count", "tokenizer init ok");
        }
        if (!_attr.filename_tokens_embed.empty())
        {
            if (!embed_selector.Init(_attr.filename_tokens_embed, _attr.tokens_embed_num, _attr.tokens_embed_size,_attr.b_use_mmap_load_embed))
            {
                ALOGE("embed_selector.Init(%s, %d, %d) failed", _attr.filename_tokens_embed.c_str(), _attr.tokens_embed_num, _attr.tokens_embed_size);
                return false;
            }
            update_cqdm(&cqdm, 1, "count", "embed_selector init ok");
        }
        //如果_attr.decoder_num数量大于1，说明有多个decoder模型需要加载，则执行这一步。
        if(_attr.decoder_num > 1)
        {
            LoaderDecoders();
        }
        int ret = ot_avp_npu_load_model(_attr.filename_post_model.c_str(), &post_model.handle);
        if (ret != 0)
        {
            ALOGE("init post model(%s) failed", _attr.filename_post_model.c_str());
            return false;
        }
        ret = ot_avp_npu_get_runtime_buffer_size(post_model.handle, &post_runtime_buf_size);
        if (post_runtime_buf_size != 0U) {
            if (post_runtime_buf == nullptr) {
                ot_phys_addr phy_addr;
                sample_avp_npu_malloc(&post_runtime_buf, &phy_addr, post_runtime_buf_size);
            }
            ret = ot_avp_npu_set_runtime_buffer(post_model.handle, (ot_u8 *)post_runtime_buf, post_runtime_buf_size);
            if (ret != 0) {
                return false;
            }
        }
        prepare_input_output_tensor(post_model.handle, &post_model);
        if (_attr.decoder_num == 1) {
            update_cqdm(&cqdm, _attr.file_num, "count", "init post model ok\n");
        }else {
            update_cqdm(&cqdm, _attr.decoder_num + _attr.file_num, "count", "init post model ok\n");
        }
        // Reset();
        ALOGI("Model init ok");
        return true;
    }
    bool LoadDecoder(int i) {
        int ret = ot_avp_npu_load_model(llama_layers[i].filename.c_str(), &llama_layers[i].layer.handle);
        if (ret != 0)
        {
            ALOGE("init model(%s) failed", llama_layers[i].filename.c_str());
            return false;
        }
        if (decoder_runtime_buf == NULL) {
            // FIXME: 应该申请所有模型最大的runtime buf size.

            ret = ot_avp_npu_get_runtime_buffer_size(llama_layers[i].layer.handle, &decoder_runtime_buf_size);
            ot_phys_addr rts_phys_addr;
            ret = sample_avp_npu_malloc(&decoder_runtime_buf, &rts_phys_addr, decoder_runtime_buf_size);
            if (ret != 0) {
                printf("malloc runtime buf fail\n");
                return false;
            }
        }
        // ret = ot_avp_npu_set_runtime_buffer(llama_layers[i].layer.handle, (ot_u8*)decoder_runtime_buf, decoder_runtime_buf_size);
        // if (ret != 0) {
        //     return false;
        // }
        // prepare_input_output_tensor_for_tinylama_decoder(llama_layers[i].layer.handle, &llama_layers[i].layer, i);
        return true;
    }

    void UnloadDecoder(int i) {
        if (decoder_runtime_buf != NULL) {
            sample_avp_npu_free(decoder_runtime_buf);
            decoder_runtime_buf = NULL;
        }
        if (post_runtime_buf != NULL) {
            sample_avp_npu_free(post_runtime_buf);
            post_runtime_buf = NULL;
        }
        if (kcache_buf != NULL) {
            sample_avp_npu_free(kcache_buf);
            kcache_buf = NULL;
        }
        if (vcache_buf != NULL) {
            sample_avp_npu_free(vcache_buf);
            vcache_buf = NULL;
        }
        // destory_input_output_tensor_for_tinylama_decoder(llama_layers[i].layer.handle, &llama_layers[i].layer, i);
        destory_input_output_tensor(llama_layers[i].layer.handle, &llama_layers[i].layer);
        ot_avp_npu_unload_model(llama_layers[i].layer.handle);
    }

    LLMAttrType *getAttr()
    {
        return &_attr;
    }

    void Deinit()
    {
        if (_attr.decoder_num > 1)
        {
            for (int i = 0; i < _attr.decoder_num; i++)
            {
                UnloadDecoder(i);
            }
        }
        ot_avp_npu_unload_model(post_model.handle);
        destory_input_output_tensor(post_model.handle, &post_model);
        if(!_attr.filename_tokens_embed.empty())
        {
            embed_selector.Deinit();
        }
    }

    std::tuple<std::vector<float>, std::vector<float>> BuildRopeCache(int64_t size, int64_t dim,
        int64_t base = 10000, int64_t scaling = 1) {
        std::vector<float> theta(static_cast<size_t>(dim / 2), 0);
        for (auto idx = 0U; idx < static_cast<size_t>(dim / 2); idx += 1) {
            theta[idx] = 1.0 / (std::pow(base, static_cast<float>(idx * 2) / dim));
        }
        std::vector<float> seqIdx(size, 0);
        for (auto idx = 0U; idx < size; idx++) {
            seqIdx[idx] = static_cast<float>(idx) / scaling;
        }
        std::vector<float> cos(size * static_cast<size_t>(dim / 2), 0);
        std::vector<float> sin(size * static_cast<size_t>(dim / 2), 0);
        for (auto outerIdx = 0U; outerIdx < size; outerIdx++) {
            for (auto innerIdx = 0U; innerIdx < static_cast<size_t>(dim / 2); innerIdx++) {
                auto idxTheta = seqIdx[outerIdx] * theta[innerIdx];
                cos[outerIdx * static_cast<size_t>(dim / 2) + innerIdx] = std::cos(idxTheta);
                sin[outerIdx * static_cast<size_t>(dim / 2) + innerIdx] = std::sin(idxTheta);
            }
        }
        return std::tie(cos, sin);
    }

    void Stop()
    {
        b_stop = true;
    }

    std::vector<int> Encode(std::string input_str)
    {
        switch (_attr.tokenizer_type) {
            case TKT_LLaMa:
            case TKT_Qwen:
                return tokenizer->Encode(input_str);
            default:
                return std::vector<int>();
        }
    }

    ot_avp_tensor *GetTensorByTensorName(model_task_info* layer, bool b_input, std::string name)
    {
        unsigned int idx = 0;
        int ret = -1;
        if (b_input) {
            ret = ot_avp_npu_get_input_index_by_name(layer->handle, name.c_str(), name.size(), &idx);
        } else {
            ret = ot_avp_npu_get_output_index_by_name(layer->handle, name.c_str(), name.size(), &idx);
        }
        if (ret != 0) {
            ALOGE("get idx by name[%s] fail.\n",name.c_str());
            return NULL;
        }
        return (b_input ? &layer->input_tensor[idx]: &layer->output_tensor[idx]);
    }


    int32_t FillTensorFromBuf(ot_avp_tensor* tensor, uint8_t* buf, uint32_t buf_size, ot_avp_data_type buf_dtype)
    {
        if (tensor->dtype == buf_dtype) {
            (void)memcpy((void *)tensor->virt_addr, buf, buf_size);
            return 0;
        }
        if (tensor->dtype == OT_AVP_DTYPE_F32 && buf_dtype == OT_AVP_DTYPE_F16) {
            float *out_data = (float *)tensor->virt_addr;
            uint16_t *in_data = (uint16_t *)buf;
            uint32_t buf_num = buf_size / sizeof(uint16_t);
            for (uint32_t i = 0; i < buf_num; i++) {
                out_data[i] = data_convert_fp16_to_fp32(in_data[i]);
            }
            return 0;
        } else if (tensor->dtype == OT_AVP_DTYPE_F16 && buf_dtype == OT_AVP_DTYPE_F32) {
            uint16_t *out_data = (uint16_t *)tensor->virt_addr;
            float *in_data = (float *)buf;
            uint32_t buf_num = buf_size / sizeof(float);
            for (uint32_t i = 0; i < buf_num; i++) {
                out_data[i] = data_convert_fp32_to_fp16(in_data[i]);
            }
            return 0;
        }

        ALOGE("not support FillTensorFromBuf.");
        return -1;
    }

    int32_t FillBufFromTensor(uint8_t* buf, uint32_t buf_size, ot_avp_data_type buf_dtype, ot_avp_tensor* in_tensor)
    {
        if (in_tensor->dtype == buf_dtype) {
            (void)memcpy(buf, (ot_void *)in_tensor->virt_addr, buf_size);
            return 0;
        }
        if (in_tensor->dtype == OT_AVP_DTYPE_F16 && buf_dtype == OT_AVP_DTYPE_F32) {
            float *out_data = (float *)buf;
            uint16_t *in_data = (uint16_t *)in_tensor->virt_addr;
            uint32_t buf_num = buf_size / sizeof(float);
            for (uint32_t i = 0; i < buf_num; i++) {
                out_data[i] = data_convert_fp16_to_fp32(in_data[i]);
            }
            return 0;
        } else if (in_tensor->dtype == OT_AVP_DTYPE_F32 && buf_dtype == OT_AVP_DTYPE_F16) {
            uint16_t *out_data = (uint16_t *)buf;
            float *in_data = (float *)in_tensor->virt_addr;
            uint32_t buf_num = buf_size / sizeof(uint16_t);
            for (uint32_t i = 0; i < buf_num; i++) {
                out_data[i] = data_convert_fp32_to_fp16(in_data[i]);
            }
            return 0;
        }

        ALOGE("not support FillBufFromTensor.");
        return -1;
    }

    void writeFloatArrayToBinFile(const std::string& file_path, const float* data, size_t size) {
        std::ofstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << file_path << std::endl;
            return;
        }
        file.write(reinterpret_cast<const char*>(data), size * sizeof(float));
        file.close();
    }

    bool LLamaInputPrepare(model_task_info* layer, unsigned int indices, int m)
    {
        float input_pos_val = indices;
        ot_avp_tensor* input_pos = GetTensorByTensorName(layer, true, "input_pos");
        int32_t ret = FillTensorFromBuf(input_pos, (uint8_t *)&input_pos_val, sizeof(float), OT_AVP_DTYPE_F32);
        if (ret != 0) {
            return false;
        }
        ot_avp_tensor* llama_model_mask = GetTensorByTensorName(layer, true, "model_mask");
        ret = FillTensorFromBuf(llama_model_mask, (uint8_t *)mask.data(), mask.size() * sizeof(float), OT_AVP_DTYPE_F32);
        if (ret != 0) {
            return false;
        }
        if (m == 0) {
            ot_avp_tensor* llama_model_embedding = GetTensorByTensorName(layer, true, "model_embedding");
            ret = FillTensorFromBuf(llama_model_embedding, (uint8_t *)embed.data(), embed.size() * sizeof(uint16_t), OT_AVP_DTYPE_F16);
            // std::string embed_file = "./embeded/embeded_decoder_" + std::to_string(m) + "token_" + std::to_string(indices) +".bin";
            // writeFp16ArrayToBinFile(embed_file, output, embed.size(), indices, m);
        }
        ot_avp_tensor* llama_model_cos = GetTensorByTensorName(layer, true, "model_cos");
        ret = FillTensorFromBuf(llama_model_cos, (uint8_t *)(cos_cache.data() + indices * _attr.position_lenth), _attr.position_lenth * sizeof(float), OT_AVP_DTYPE_F32);

        ot_avp_tensor* llama_model_sin = GetTensorByTensorName(layer, true, "model_sin");
        ret = FillTensorFromBuf(llama_model_sin, (uint8_t *)(sin_cache.data() + indices * _attr.position_lenth), _attr.position_lenth * sizeof(float), OT_AVP_DTYPE_F32);
        // m=0;
        std::string kcache_name = "decoder_" + std::to_string(m) + "_kcache";
        ot_avp_tensor* kcache = GetTensorByTensorName(layer, true, kcache_name);
        ret = FillTensorFromBuf(kcache, (uint8_t *)(k_cache.data() + m * _attr.max_token_len * _attr.kv_cache_size), _attr.max_token_len * _attr.kv_cache_size * sizeof(float), OT_AVP_DTYPE_F32);
        std::string vcache_name = "decoder_" + std::to_string(m) + "_vcache";
        ot_avp_tensor* vcache = GetTensorByTensorName(layer, true, vcache_name);
        ret = FillTensorFromBuf(vcache, (uint8_t *)(v_cache.data() + m * _attr.max_token_len * _attr.kv_cache_size), _attr.max_token_len * _attr.kv_cache_size * sizeof(float), OT_AVP_DTYPE_F32);
        return true;
    }
    //根据_attr.tokenizer_type对不同模型进行输入数据的准备
    bool ModelInputPrepare(model_task_info* model, unsigned int indices, int m)
    {
        switch (_attr.tokenizer_type) {
            case TKT_Qwen:
            case TKT_LLaMa:
                return LLamaInputPrepare(model, indices, m);
            default:
                return false;
        }
    }

    bool LLamaDecoderUpdate(model_task_info* layer, unsigned int indices, int m)
    {
        if (m+1<_attr.decoder_num) {
            uint16_t* output = (uint16_t*)llama_layers[m].layer.output_tensor[0].virt_addr;
            uint16_t* input = (uint16_t*)llama_layers[m+1].layer.input_tensor[0].virt_addr;
            for (size_t i =0; i<embed.size(); i++)
            {
                input[i] = output[i];
            }
            // std::string embed_file = "./embeded/embeded_decoder_" + std::to_string(m+1) + "token_" + std::to_string(indices) +".bin";
            // writeFp16ArrayToBinFile(embed_file, output, embed.size(), indices, m);
        }
        // m=0;
        std::string kcache_name = "decoder_" + std::to_string(m) + "_kcache";
        GetTensorByTensorName(layer, true, kcache_name);
        std::string kupdate_name = "decoder_" + std::to_string(m) + "_kupdate";
        ot_avp_tensor* kupdate = GetTensorByTensorName(layer, false, kupdate_name);
        (void)memcpy(k_cache.data() + m * _attr.max_token_len * _attr.kv_cache_size + indices * _attr.kv_cache_size, (float *)kupdate->virt_addr,sizeof(float) * _attr.kv_cache_size);
        std::string vcache_name = "decoder_" + std::to_string(m) + "_vcache";
        GetTensorByTensorName(layer, true, vcache_name);
        std::string vupdate_name = "decoder_" + std::to_string(m) + "_vupdate";
        ot_avp_tensor* vupdate = GetTensorByTensorName(layer, false, vupdate_name);
        (void)memcpy(v_cache.data() + m * _attr.max_token_len * _attr.kv_cache_size + indices * _attr.kv_cache_size, (float *)vupdate->virt_addr,sizeof(float) * _attr.kv_cache_size);
        return true;
    }

    bool ModelInputUpdate(model_task_info* model, unsigned int indices, int m)
    {
        switch (_attr.tokenizer_type) {
            case TKT_Qwen:
            case TKT_LLaMa:
                return LLamaDecoderUpdate(model, indices, m);
            default:
                return false;
        }
    }

    float* PostModelOutput(model_task_info post_model)
    {
        unsigned int input_input_idx = 0;
        uint16_t *input_input_ptr = (uint16_t *)post_model.input_tensor[input_input_idx].virt_addr;
        uint16_t *output_ptr = (uint16_t *)llama_layers[_attr.decoder_num-1].layer.output_tensor[input_input_idx].virt_addr;
        (void)memcpy(input_input_ptr, output_ptr, embed.size() * sizeof(uint16_t));

        ot_avp_npu_execute(post_model.handle, post_model.input_dataset, post_model.output_dataset);

        unsigned int output_output_idx = 0;
        ot_avp_npu_get_output_index_by_name(post_model.handle, "output", strlen("output"), &output_output_idx);
        unsigned char *output_output_ptr = (unsigned char *)post_model.output_tensor[output_output_idx].virt_addr;
        return (float *)output_output_ptr;
    }

    float* GetPostModelOutput(model_task_info post_model)
    {
        switch (_attr.tokenizer_type) {
            case TKT_Qwen:
            case TKT_LLaMa:
                return PostModelOutput(post_model);
            default:
                return nullptr;
        }
    }

    bool Isend(int max_index)
    {
        switch (_attr.tokenizer_type) {
            case TKT_Qwen:
            case TKT_LLaMa:
                return tokenizer->isEnd(max_index);
            default:
                return false;
        }
    }

    std::string Decode(std::vector<int> token)
    {
        switch (_attr.tokenizer_type) {
            case TKT_LLaMa:
            case TKT_Qwen:
                return tokenizer->Decode(token);
            default:
                return "";
        }
    }

    void RecoverStateToInitial() 
    {
        (void)memset(k_cache.data(), 0, k_cache.size()*sizeof(float));
        (void)memset(v_cache.data(), 0, v_cache.size()*sizeof(float));
        for(size_t i=0; i<mask.size(); i++)
        {
            mask[i] = -std::numeric_limits<float>::infinity();
        }
        return;
    }

    std::string Run(std::string input_str)
    {
        b_stop = false;
        std::string final_out;
        std::vector<int> cached_token;
        std::vector<int> token_ids = Encode(input_str);
        int len_of_input = token_ids.size();
        timer t_cost;
        _attr.next_token = token_ids[0];
        t_cqdm cqdm = create_cqdm(_attr.max_token_len, 32);
        bool b_hit_eos = false;
        for (int indices = 0; indices < _attr.max_token_len ; indices++)
        {
            if(mask.size()>0)
            {
                mask[indices] = 0.0;
            }
            if (b_stop)
            {
                break;
            }
            if (!_attr.filename_tokens_embed.empty())
            {
                embed_selector.getByIndex(_attr.next_token, embed);
            }
            //fprintf(stderr, "%s.%d indices=%d\n", __FUNCTION__, __LINE__, indices);
            for (int m = 0; m < _attr.decoder_num; m++)
            {
                model_task_info *cur_model = (_attr.decoder_num > 1) ? &llama_layers[m].layer : &post_model;
                ModelInputPrepare(cur_model, indices, m);
                auto ret = ot_avp_npu_execute(cur_model->handle, cur_model->input_dataset, cur_model->output_dataset);
                if (ret != 0) {
                    fprintf(stderr, "execute token %d decoder %d fail.\n", indices, m);
                    return "fail";
                }
                ModelInputUpdate(cur_model, indices, m);
            }
            if (indices + 1 < len_of_input) {
                _attr.next_token = token_ids[indices + 1];
            } else {
                float *post_out = GetPostModelOutput(post_model);
                
                float max_val = -MAXFLOAT;
                int max_index = 0;
                for (int i = 0; i < _attr.tokens_embed_num; i++)
                {
                    float tmp = post_out[i];
                    if (tmp > max_val)
                    {
                        max_val = tmp;
                        max_index = i;
                    }
                }
                if (indices == _attr.max_token_len - 1) {
                    max_index = 151643;
                }
                _attr.next_token = max_index;
                if (Isend(max_index))
                {
                    if (cached_token.size())
                    {
                        float t_cost_ms = t_cost.cost();
                        float token_per_sec = token_ids.size() / (t_cost_ms / 1000);
                        auto tmp_out = Decode(cached_token);
                        _attr.runing_callback(cached_token.data(), cached_token.size(), tmp_out.c_str(), token_per_sec, _attr.reserve);
                        cached_token.clear();
                    }
                    b_hit_eos = true;
                    break;
                }

                token_ids.push_back(max_index);

                if (_attr.runing_callback)
                {
                    cached_token.push_back(max_index);
                    if (cached_token.size() >= 3)
                    {
                        float t_cost_ms = t_cost.cost();
                        float token_per_sec = token_ids.size() / (t_cost_ms / 1000);
                        auto tmp_out = Decode(cached_token);
                        _attr.runing_callback(cached_token.data(), cached_token.size(), tmp_out.c_str(), token_per_sec, _attr.reserve);
                        cached_token.clear();
                    }
                }

            }

            if (_attr.runing_callback == nullptr)
                update_cqdm(&cqdm, indices, "token", "");
            if (b_hit_eos)
            {
                break;
            }
        }
        RecoverStateToInitial();
        printf("\n\n");
        fflush(stdout);
        float t_cost_ms = t_cost.cost();
        ALOGN("hit eos,avg %.2f token/s\n", token_ids.size() / (t_cost_ms / 1000));

        token_ids.erase(token_ids.begin(), token_ids.begin() + len_of_input);

        return final_out;

    }
};
