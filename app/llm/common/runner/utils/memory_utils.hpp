#pragma once
#include <sys/mman.h>
#include <fcntl.h>
#include <stdio.h>
#include <fstream>
#include <vector>

bool file_exist(const std::string &path);

bool read_file(const std::string &path, std::vector<char> &data);
bool read_file(const std::string &path, char **data, size_t *len);
class MMap
{
private:
    void *_add = nullptr;
    int _size;

public:
    MMap() {}
    MMap(const char *file)
    {
        (void)open_file(file);
    }
    ~MMap()
    {
        close_file();
    }

    bool open_file(const char *file)
    {
        _add = _mmap(file, &_size);
        if (_add == nullptr)
        {
            return false;
        }
        return true;
    }

    void close_file()
    {
        if (_add != nullptr)
        {
            munmap(_add, _size);
/* LDRA_EXCLUDE 71 S */
            _add = nullptr;
            _size = 0;
        }
    }

    size_t size()
    {
        return (size_t)_size;
    }

    void *data()
    {
        return _add;
    }

    static void *_mmap(const char *model_file, int *model_size)
    {
        auto *file_fp = fopen(model_file, "r");
        if (!file_fp)
        {

            return nullptr;
        }
        (void)fseek(file_fp, 0, SEEK_END);
        *model_size = (int)ftell(file_fp);
        (void)fclose(file_fp);
        if(*model_size == -1){
            perror("ftell");
            return nullptr;
        }
        /* LDRA_EXCLUDE 83 S */
        int fd = open(model_file, O_RDWR, 0644);
        void *mmap_add = mmap(NULL, *model_size, PROT_WRITE, MAP_SHARED, fd, 0);
        return mmap_add;
    }
};
