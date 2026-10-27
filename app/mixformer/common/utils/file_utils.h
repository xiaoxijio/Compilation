#ifndef FILE_UTILS_H
#define FILE_UTILS_H
#include <string.h>
#include <stdint.h>

unsigned long int get_bin_file_size(const char* path);
int read_data_from_bin_file(const char* path, uint8_t* buf, unsigned long int size);
// void mkdir_dir(char* dir_path);
#endif
