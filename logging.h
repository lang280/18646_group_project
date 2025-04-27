#ifndef LOGGING_H
#define LOGGING_H

#include <stdio.h>
#include <time.h>

// 日志级别控制
#define LOG_LEVEL_NONE  0  // 不输出任何日志
#define LOG_LEVEL_ERROR 1  // 只输出错误
#define LOG_LEVEL_INFO  2  // 输出基本信息
#define LOG_LEVEL_DEBUG 3  // 输出详细信息
#define LOG_LEVEL LOG_LEVEL_INFO  // 设置全局日志级别

// 日志宏
#define LOG_ERROR(fmt, ...) if (LOG_LEVEL >= LOG_LEVEL_ERROR) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)  if (LOG_LEVEL >= LOG_LEVEL_INFO)  printf("[INFO] " fmt "\n", ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) if (LOG_LEVEL >= LOG_LEVEL_DEBUG) printf("[DEBUG] " fmt "\n", ##__VA_ARGS__)

// 计时器结构体
typedef struct {
    clock_t start;
    clock_t end;
    const char* name;
} Timer;

// 计时器函数
static inline Timer timer_start(const char* name) {
    Timer t = { clock(), 0, name };
    if (LOG_LEVEL >= LOG_LEVEL_DEBUG)
        printf("[TIMER] %s started\n", name);
    return t;
}

static inline double timer_stop(Timer* t) {
    t->end = clock();
    double elapsed = (double)(t->end - t->start) / CLOCKS_PER_SEC;
    if (LOG_LEVEL >= LOG_LEVEL_INFO && t->name)
        printf("[TIMER] %s completed in %.2f seconds\n", t->name, elapsed);
    return elapsed;
}

// 进度条显示
static inline void progress_init(int total, const char* desc) {
    if (LOG_LEVEL >= LOG_LEVEL_INFO) {
        printf("%s: ", desc);
        fflush(stdout);
    }
}

static inline void progress_update(int current, int total) {
    if (LOG_LEVEL >= LOG_LEVEL_INFO) {
        // 每处理 total/20 个数据显示一个点
        if (current % (total / 20 > 0 ? total / 20 : 1) == 0) {
            printf(".");
            fflush(stdout);
        }
    }
}

static inline void progress_finish(double accuracy, int correct, int total, double time_taken) {
    if (LOG_LEVEL >= LOG_LEVEL_INFO) {
        if (time_taken > 0) {
            printf(" Done in %.2f seconds | Accuracy: %.4f (%d/%d correct)\n", 
                   time_taken, accuracy, correct, total);
        } else {
            printf(" Done | Accuracy: %.4f (%d/%d correct)\n", accuracy, correct, total);
        }
    }
}

#endif /* LOGGING_H */ 