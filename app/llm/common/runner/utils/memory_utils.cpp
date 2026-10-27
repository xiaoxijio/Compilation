#include "memory_utils.hpp"

bool file_exist(const std::string &path)
{
    auto flag = false;

    std::fstream fs(path, std::ios::in | std::ios::binary);
    flag = fs.is_open();
    fs.close();

    return flag;
}

bool read_file(const std::string &path, std::vector<char> &data)
{

    std::fstream fs(path, std::ios::in | std::ios::binary);

    if (!fs.is_open())
    {
        return false;
    }

    // get file size
    fs.seekg(0, std::ios::end);
    size_t file_size = fs.tellg();
    fs.seekg(0, std::ios::beg);

    if (file_size == 0U)
    {
        return false;
    }

    data.resize(file_size);
    fs.read(data.data(), file_size);
    // data.insert(data.end(), std::istreambuf_iterator<char>(fs), std::istreambuf_iterator<char>());

    fs.close();

    return true;
}

bool read_file(const std::string &path, char **data, size_t *len)
{
    FILE *fp = fopen(path.c_str(), "rb");

    if (fp == nullptr)
    {
        return false;
    }

    (void)fseek(fp, 0, SEEK_END);

    long file_len = ftell(fp);
    if (file_len == -1L) {
        // ftell failed
        perror("ftell failed");
        (void)fclose(fp);
        return false;
    }
    *len = (size_t)file_len;
    (void)fseek(fp, 0, SEEK_SET);

/* LDRA_EXCLUDE 71 S */
    *data = new char[*len];

   size_t items_read = fread(*data, *len, 1, fp);
   if (items_read < *len) {
       if (feof(fp) != 0) {
           // 到达文件末尾
           printf("End of file reached.\n");
       } else if (ferror(fp) != 0) {
           // 发生错误
           perror("File read error");
           (void)fclose(fp);
           if (*data != nullptr) {
               delete[] *data;
               *data = nullptr;
           }
           return false;
       } else {
        /*non-execution*/
    }
   }
    (void)fclose(fp);

    return true;
}
