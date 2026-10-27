#include "file_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <math.h>

#define FILE_PATH_MAX (1024)

unsigned long int get_file_by_line(const char* path)
{
    unsigned long int size = 0UL;
    FILE *file = fopen(path, "rb");
    int ch;
    while ((ch = fgetc(file)) != EOF) {
        if (ch == '\n') {
            size++;
        }
    }
    fclose(file);
    return size;
}


unsigned long int get_bin_file_size(const char* path)
{
    unsigned long int size = 0UL;
    FILE *fp = fopen(path, "rb");
    if (fp) {
        fseek(fp, 0, SEEK_END);
        size = ftell(fp);
        fclose(fp);
    }
    return size;
}

int read_data_from_bin_file(const char* path, uint8_t* buf, unsigned long int size)
{
    FILE *infile;
    if ((infile = fopen(path, "rb")) == NULL) {
        printf("can't open path: %s", path);
        return -1;
    }

    fseek(infile, 0, SEEK_SET);
    const unsigned long int read_size = fread(buf, 1, size, infile);
    if (read_size < size) {
        printf("file_size[%ld] is smaller than size[%ld].", read_size, size);
        fclose(infile);
        return -1;
    }
    fclose(infile);
    return 0;
}

// void mkdir_dir(char* dir_path)
// {
//     char command[FILE_PATH_MAX] = {0};
//     int ret = snprintf(command, FILE_PATH_MAX, "rm -rf %s", dir_path);
//     if (ret < 0) {
//         printf("mkdir directory fail, dir path is too long\n");
//         return;
//     }
//     ret = system(command);
//     if (mkdir(dir_path, 0755) != 0) {
//         printf("create directory:%s failed", dir_path);
//     }
// }