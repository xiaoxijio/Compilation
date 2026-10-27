#include "runner/Tokenizer/Tokenizer.hpp"
#include "signal.h"

#include "runner/LLM.hpp"

#include "cmdline.hpp"
#include "ot_avp_npu_rts.h"

#ifndef SIMULATOR
#include "ot_smr.h"
#endif

static LLM lLaMa;

static void __sigExit(int iSigNo)
{
    (void)iSigNo;
    lLaMa.Stop();
    return;
}

void llm_running_callback(int *p_token, int n_token, const char *p_str, float token_per_sec, void *reserve)
{
    (void)p_token;
    (void)n_token;
    (void)token_per_sec;
    (void)reserve;
    fprintf(stdout, "%s", p_str);
    fflush(stdout);
}

std::string prompt_complete(std::string prompt, TokenizerType tokenizer_type)
{
    std::ostringstream oss_prompt;
    switch (tokenizer_type)
    {
    case TKT_LLaMa:
        oss_prompt << "<|user|>\n"
                   << prompt << "</s><|assistant|>\n";
        break;
    case TKT_Qwen:
        oss_prompt << "<|im_start|>system\nYou are a helpful assistant.<|im_end|>";
        oss_prompt << "\n<|im_start|>user\n"
                   << prompt << "<|im_end|>\n<|im_start|>assistant\n";
        break;
    case TKT_OPT:
        oss_prompt << "</s>"<<prompt;
        break;
    default:
        oss_prompt << prompt;
        break;
    }

    return oss_prompt.str();
}

#ifndef SIMULATOR
int ot_sdk_init()
{
    // smr init
    ot_smr_attr smr_attr = {};
    const char *region_name = (const char *)"anony";

    strcpy((char *)smr_attr.regions[0].region_name, region_name);
    smr_attr.regions[0].addr = (ot_phys_addr)0x80000000;
    smr_attr.regions[0].len = 0x80000000;  // 2GB
    smr_attr.allocator = OT_SMR_ALLOCATOR_BBAA;
    smr_attr.num = 1;

    if (ot_smr_config(&smr_attr) != 0) {
        printf("SMR already config!\n");
    }
    ot_smr_init();

    // npu init
    extra_config extra_npu_config[2];
    const char* layer_path_key   = "DUMP_PATH";
    const char* layer_path_value = "LayerDump";
    (void)memcpy(&(extra_npu_config[0].key), layer_path_key, strlen(layer_path_key) + 1);
    (void)memcpy(&(extra_npu_config[0].value), layer_path_value, strlen(layer_path_value) + 1);

    ot_avp_npu_config config = {
        .core_ids = 2,
        .dump_layer = OT_FALSE,   // dump layer
        .perf_mode = OT_FALSE,  // perf_mode
        .log_level = 3,   // log level info
        .extra_config_num = 2,
        .extra_configs = extra_npu_config,
    };

    int ret = ot_avp_npu_init(&config);
    if (ret != 0) {
        ot_smr_deinit();
        return -1;
    }

    return 0;
}

void ot_sdk_deinit()
{
    ot_avp_npu_deinit();
    ot_smr_deinit();
}
#endif
int main(int argc, char *argv[])
{
    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT, __sigExit);
    LLMAttrType attr;
    std::string prompt = "Hi";
    bool b_continue = false;

    cmdline::parser cmd;
    cmd.add<std::string>("prompt", 'p', "prompt", true, prompt);
    cmd.add<std::string>("decoder_filename", 0, "model path", false, attr.decoder_filename);
    cmd.add<int>("tokenizer_type", 0, "tokenizer type 0:LLaMa 1:Qwen 2:HTTP 3:Phi3 4:TKT_END 5:TKT_OPT 6:TKT_CPM", false, attr.tokenizer_type);
    cmd.add<std::string>("filename_post_model", 0, "post model path", false, attr.filename_post_model);
    cmd.add<std::string>("filename_tokenizer_model", 0, "tokenizer model path", false, attr.filename_tokenizer_model);
    cmd.add<std::string>("filename_tokens_embed", 0, "tokens embed path", false, attr.filename_tokens_embed);
    cmd.add<bool>("bos", 0, "", false, attr.b_bos);
    cmd.add<bool>("eos", 0, "", false, attr.b_eos);
    cmd.add<int>("decoder_num", 0, "num of model(for template)", false, attr.decoder_num);
    cmd.add<int>("tokens_embed_num", 0, "tokens embed num", false, attr.tokens_embed_num);
    cmd.add<int>("tokens_embed_size", 0, "tokens embed size", false, attr.tokens_embed_size);
    cmd.add<bool>("use_mmap_load_embed", 0, "it can save os memory", false, attr.b_use_mmap_load_embed);
    cmd.add<bool>("live_print", 0, "print in live if set true, else print in end", false);

    cmd.add<bool>("continue", 0, "continuous dialogue", false, b_continue);

    cmd.parse_check(argc, argv);
    attr.tokenizer_type = (TokenizerType)cmd.get<int>("tokenizer_type");
    attr.filename_tokenizer_model = cmd.get<std::string>("filename_tokenizer_model");
    attr.decoder_filename = cmd.get<std::string>("decoder_filename");
    attr.filename_tokens_embed = cmd.get<std::string>("filename_tokens_embed");
    attr.filename_post_model = cmd.get<std::string>("filename_post_model");
    attr.decoder_num = cmd.get<int>("decoder_num");
    prompt = cmd.get<std::string>("prompt");
    attr.b_eos = false;
    bool b_live_print = true;
    if (b_live_print)
    {
        attr.runing_callback = llm_running_callback;
        attr.reserve = 0;
    }

    b_continue = true;
    #ifndef SIMULATOR
    if (ot_sdk_init() != 0) {
        printf("ot_sdk_init init fail.\n");
        return -1;
    }
    #endif
    if (!lLaMa.Init(attr))
    {
        #ifndef SIMULATOR
        ot_sdk_deinit();
        #endif
        return -1;
    }

    if (prompt != "")
    {
        auto output = lLaMa.Run(prompt_complete(prompt, attr.tokenizer_type));
        if (!b_live_print)
            printf("%s\n", output.c_str());
    }

    //
    if (b_continue)
    {
        printf("Type \"q\" to exit, Ctrl+c to stop current running\n");
    }

    while (b_continue)
    {
        printf(">> ");
        fflush(stdout);
        std::string input;
        std::getline(std::cin, input);
        if (input == "q")
        {
            break;
        }
        if (input == "")
        {
            continue;
        }

        auto output = lLaMa.Run(prompt_complete(input, attr.tokenizer_type));
        if (!b_live_print)
            printf("%s\n", output.c_str());
    }

        lLaMa.Deinit();
        #ifndef SIMULATOR
        ot_sdk_deinit();
        #endif
        return 0;

}
