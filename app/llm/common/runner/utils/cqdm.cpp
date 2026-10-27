#include "cqdm.h"
#include <math.h>
// Returns the current time in milliseconds
static long long get_msec_now(void)
{
    struct timeval time;

    if (gettimeofday(&time, NULL) == -1) {
        return (0);
    }
    return ((long long)time.tv_sec * 1000LL) + ((long long)time.tv_usec / 1000LL);
}

t_cqdm create_cqdm(int total, int size)
{
    t_cqdm cqdm;

    cqdm.total = total;
    cqdm.count = 0;
    cqdm.average_time = 0.0f;
    cqdm.last_time = (double)get_msec_now();
    cqdm.total_time = 0.0f;
    cqdm.size = size;
    return (cqdm);
}

void update_cqdm(t_cqdm *cqdm, int x, const char *unit, const char *log_str)
{
    long long now = 0;
    double temp = 0.0;

    now = get_msec_now();

    cqdm->count++;
    cqdm->total_time += (((double)now - cqdm->last_time) / 1000.0);
    if (cqdm->count != 0) {
        cqdm->average_time = cqdm->total_time / (double)cqdm->count;
    }
    cqdm->last_time = (double)now;
    if (cqdm->total != 0) {
        /* LDRA_EXCLUDE 444 S */
        temp = ((double)(x + 1) / (double)cqdm->total);
    }

/* LDRA_EXCLUDE 589 S */
    fprintf(stdout, "\033[1;30;33m\r%3f%% | ", round(temp * 100.0f));
    int loop = (int)round(temp * (double)cqdm->size);
    for (int i = 0; i < loop; i++){
        fprintf(stdout, "#");
    }
    loop = (int)round((double)cqdm->size - (temp * (double)cqdm->size));
    for (int i = 0; i < loop; i++){
        fprintf(stdout, " ");
    }
    if (cqdm->average_time != 0.0) {
        fprintf(stdout, " | %3d / %3d [%2.2fs<%2.2fs, %2.2f %s/s] \033[1;30;32m%s\033[0m", x + 1, cqdm->total,
                cqdm->total_time, cqdm->average_time * (double)cqdm->total, 1.0 / cqdm->average_time, unit, log_str);
    }
    (void)fflush(stdout);
}
