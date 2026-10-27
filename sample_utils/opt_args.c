#include "opt_args.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stddef.h>
// 参数类型定义
typedef enum {
    ARG_TYPE_NONE,      // 无参数选项（如 -p, -d）
    ARG_TYPE_STRING,    // 字符串参数
    ARG_TYPE_UINT,      // 无符号整数
    ARG_TYPE_INT,       // 有符号整数
} arg_type_t;

// 参数规格结构
typedef struct {
    const char* opt;          // 选项字符串，如 "-m"
    arg_type_t type;          // 参数类型
    const char* description;  // 描述（用于错误信息）
    size_t offset;            // 在 arg_config 中的字段偏移
    bool required;           // 是否必需
    union {
        struct { uint32_t min; uint32_t max; } range; // 数值范围
    } constraints;
} arg_spec_t;

// 定义所有支持的参数规格
static const arg_spec_t arg_specs[] = {
    {"-m", ARG_TYPE_STRING, "Specify the ortm board/sim model", offsetof(arg_config, model_path), false, {{0}}},
    {"-i", ARG_TYPE_STRING, "Specify input file/directory", offsetof(arg_config, input_path), false, {{0}}},
    {"-o", ARG_TYPE_STRING, "Specify output directory", offsetof(arg_config, output_path), false, {{0}}},
    {"-M", ARG_TYPE_NONE, "Multi-core performance test(can only use -m/-r/-s)", offsetof(arg_config, multi_core), false, {{0}}},
    {"-p", ARG_TYPE_NONE, "Show profiling info", offsetof(arg_config, perf_mode), false, {{0}}},
    {"-d", ARG_TYPE_NONE, "Enable dump mode", offsetof(arg_config, dump_layer), false, {{0}}},
    {"-r", ARG_TYPE_INT, "Repeat times", offsetof(arg_config, repeat), false, {{0}}},
    {"-f", ARG_TYPE_NONE, "input is folder enable",  offsetof(arg_config,is_folder), false, {{0}}},
    {"-s", ARG_TYPE_UINT, "Using SRAM size. support 0/1/2/3. Unit is MB", offsetof(arg_config, buffer_type), false, .constraints.range = {0, 3}},
    {"-c", ARG_TYPE_UINT, "Bind NPU core(0/1/2). 0 means not bind. 1 means first core. 2 means second core", offsetof(arg_config, core_ids), false,.constraints.range = {0, 2}},
    {"-t", ARG_TYPE_UINT, "Target Dump with binary/txt(0/1). 0 means binary. 1 means txt", offsetof(arg_config, output_type), false,.constraints.range = {0, 1}}
};
#define ARG_COUNT (sizeof(arg_specs) / sizeof(arg_specs[0]))

// 验证函数
static bool is_valid_option(const char* arg) {
    return arg && strlen(arg) == 2 && arg[0] == '-';
}
static bool is_non_negative_number(const char* str) {
    if (!str || !str[0]) return false;
    
    for (size_t i = 0; str[i]; i++) {
        if (!isdigit((unsigned char)str[i])) return false;
    }
    return true;
}
static bool is_integer(const char* str) {
    if (!str || !str[0]) return false;
    int i = 0;
    // 允许正负号开头
    if (str[i] == '-' || str[i] == '+') i++;
    // 符号后必须有数字
    if (str[i] == '\0') return false;
    for (; str[i]; i++) {
        if (!isdigit((unsigned char)str[i])) return false;
    }
    return true;
}
static bool is_in_range(uint32_t value, uint32_t min, uint32_t max) {
    return value >= min && value <= max;
}

// 查找参数规格
static const arg_spec_t* find_arg_spec(const char* opt) {
    for (size_t i = 0; i < ARG_COUNT; i++) {
        if (strcmp(arg_specs[i].opt, opt) == 0) {
            return &arg_specs[i];
        }
    }
    return NULL;
}
// 设置配置值
static void set_config_value(arg_config* config, const arg_spec_t* spec, void* value) {
    char* config_ptr = (char*)config + spec->offset;
    
    switch (spec->type) {
        case ARG_TYPE_STRING:
            *(char**)config_ptr = value;
            break;
        case ARG_TYPE_NONE:
            *(bool*)config_ptr = true;
            break;
        case ARG_TYPE_UINT:
            *(uint32_t*)config_ptr = *(uint32_t*)value;
            break;
        case ARG_TYPE_INT:
            *(int32_t*)config_ptr = *(int32_t*)value;
            break;
    }
}
// 解析单个参数
static int parse_argument(const char* opt, char* value, arg_config* config, const char* prog_name) {
    const arg_spec_t* spec = find_arg_spec(opt);
    if (!spec) {
        fprintf(stderr, "Error: Unsupported option '%s'\n", opt);
        fprintf(stderr, "Please use %s -h to view the help\n", prog_name);
        return -1;
    }
    
    // 检查是否需要参数
    if (spec->type != ARG_TYPE_NONE && (!value || (spec->type!= ARG_TYPE_INT && value[0] == '-'))) {
        fprintf(stderr, "Error: Option '%s' requires argument (%s).\n", opt, spec->description);
        return -1;
    }
    
    // 参数验证和赋值
    switch (spec->type) {
        case ARG_TYPE_STRING:
            if (!value || strlen(value) == 0) {
                fprintf(stderr, "Error: The parameter of option '%s' cannot be empty\n", opt);
                return -1;
            }
            set_config_value(config, spec, value);
            break;
            
        case ARG_TYPE_INT:
            if (!is_integer(value)) {
                fprintf(stderr, "Error: The option '%s' requires a integer parameter, but received '%s'\n", opt, value);
                return -1;
            }
            {
                int32_t num = (int32_t)atoi(value);
                set_config_value(config, spec, &num);
            }
            break;
            
        case ARG_TYPE_UINT:
            if (!is_non_negative_number(value)) {
                fprintf(stderr, "Error: The option '%s' requires an unsigned integer parameter, but received '%s'\n", opt, value);
                return -1;
            }
            {
                uint32_t num = (uint32_t)atoi(value);
                if (spec->constraints.range.max > 0 && 
                    !is_in_range(num, spec->constraints.range.min, spec->constraints.range.max)) {
                    fprintf(stderr, "Error: The value of option '%s' must be within the range of %u-%u\n", 
                            opt, spec->constraints.range.min, spec->constraints.range.max);
                    return -1;
                }
                set_config_value(config, spec, &num);
            }
            break;
            
        case ARG_TYPE_NONE:
            set_config_value(config, spec, NULL);
            break;
    }
    
    return 0;
}
// Print usage information
void print_usage(const char* prog_name) {
    printf("\nUsage: %s [options]\n\n", prog_name);
    printf("Options:\n");
    
    for (size_t i = 0; i < ARG_COUNT; i++) {
        const arg_spec_t* spec = &arg_specs[i];
        printf("  %-4s", spec->opt);
        
        switch (spec->type) {
            case ARG_TYPE_STRING: printf("<path>   "); break;
            case ARG_TYPE_UINT: printf("<number> "); break;
            case ARG_TYPE_NONE: printf("         "); break;
            case ARG_TYPE_INT: printf("<integer>"); break;
        }
        
        printf(" : %s", spec->description);
        
        if (spec->required) {
            printf(" [required]");
        }
        
        if (spec->constraints.range.max > 0) {
            printf(" (range: %u-%u)", spec->constraints.range.min, spec->constraints.range.max);
        }
        
        printf("\n");
    }
    
    printf("\nExamples:\n");
    printf("  %s -m model.ortm -i input.bin -o ./ \n", prog_name);
    printf("  %s -m model.ortm -i input_dir -f -p\n", prog_name);
}
// 主解析函数
int parse_optarg(int argc, char** argv, arg_config* config) {
    const char* prog_name = argv[0];

    // 检查帮助选项
    if (argc == 2 && strcmp(argv[1], "-h") == 0) {
        print_usage(prog_name);
        return 1;  // 特殊返回码表示显示帮助
    }
    
    // 解析参数
    for (int i = 1; i < argc; i++) {
        char* arg = argv[i];
        
        if (is_valid_option(arg)) {
            char* value = (i + 1 < argc) ? argv[i + 1] : NULL;
            
            if (parse_argument(arg, value, config, prog_name) != 0) {
                print_usage(prog_name);
                return -1;
            }
            
            // 跳过已消费的参数
            if (find_arg_spec(arg)->type != ARG_TYPE_NONE && value) {
                i++;
            }
        } else {
            fprintf(stderr, "Error: Invalid parameter format '%s'\n", arg);
            fprintf(stderr, "The parameters must start with '-' , for example: -m model.ortm\n");
            print_usage(prog_name);
            return -1;
        }
    }
    
    // 验证必需参数
    for (size_t i = 0; i < ARG_COUNT; i++) {
        if (arg_specs[i].required) {
            char* config_ptr = (char*)config + arg_specs[i].offset;
            char* str_value = *(char**)config_ptr;
            
            if (!str_value || strlen(str_value) == 0) {
                fprintf(stderr, "Error: An option must be specified '%s' (%s)\n", 
                        arg_specs[i].opt, arg_specs[i].description);
                print_usage(prog_name);
                return -1;
            }
        }
    }
    
    // 验证互斥条件
    if (config->perf_mode && config->repeat > 1) {
        fprintf(stderr, "perf mode can not set -r\n");
        config->repeat = 1;
    }
    
    return 0;
}
