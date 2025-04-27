#define _POSIX_C_SOURCE 199309L // clock_gettime

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "forward.h" // Include our CUDA-optimized forward propagation
#include "logging.h" // Include logging utilities

// ─────────────── 内存检查宏 ───────────────
#define CHECK_PTR(p, msg)                      \
    if (!(p))                                  \
    {                                          \
        LOG_ERROR("Failed to malloc: %s", msg); \
        exit(1);                               \
    }

// ─────────────── 超参数（已缩小） ───────────────
#define OLD_INPUT_SIZE 28
#define NEW_INPUT_SIZE 256
#define INPUT_NODES (NEW_INPUT_SIZE * NEW_INPUT_SIZE)
#define HIDDEN_NODES 256
#define OUTPUT_NODES 10
#define MAX_PATH 256
#define DEFAULT_TRAIN_IMAGES "mnist_train_images.bin"
#define DEFAULT_TRAIN_LABELS "mnist_train_labels.bin"
#define DEFAULT_TEST_IMAGES "mnist_test_images.bin"
#define DEFAULT_TEST_LABELS "mnist_test_labels.bin"
#define DEFAULT_MODEL "model.bin"
#define MAX_TRAIN 10000 // 只用40000张图片，便于调试和本地运行
#define MAX_TEST 10000

// ─────────────── 动态数据全局变量 ───────────────
double *training_images = NULL;         // [MAX_TRAIN][28*28]
double *test_images = NULL;             // [MAX_TEST][28*28]
double *training_images_resized = NULL; // [MAX_TRAIN][INPUT_NODES]
double *test_images_resized = NULL;     // [MAX_TEST][INPUT_NODES]
double *training_labels = NULL;         // [MAX_TRAIN][OUTPUT_NODES]
double *test_labels = NULL;             // [MAX_TEST][OUTPUT_NODES]
double *weight1 = NULL;                 // [INPUT_NODES][HIDDEN_NODES]
double *weight2 = NULL;                 // [HIDDEN_NODES][OUTPUT_NODES]
double *bias1 = NULL;                   // [HIDDEN_NODES]
double *bias2 = NULL;                   // [OUTPUT_NODES]

int correct_predictions;
int forward_prob_output;

// ─────────────── 计时辅助 ───────────────
static double g_ff_time = 0, g_bp_time = 0, g_wu_time = 0;
static inline double diff_sec(struct timespec a, struct timespec b)
{
    return (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
}

// ReLU及其导数
double relu_derivative(double x) { return x > 0 ? 1 : 0; }
int max_index(double *arr, int size)
{
    int max_i = 0;
    for (int i = 1; i < size; i++)
        if (arr[i] > arr[max_i])
            max_i = i;
    return max_i;
}

void input_string(const char *prompt, char *buf, int maxlen, const char *default_val)
{
    printf("%s (default: %s): ", prompt, default_val);
    if (fgets(buf, maxlen, stdin) == NULL)
        buf[0] = 0;
    int len = strlen(buf);
    if (len && buf[len - 1] == '\n')
        buf[len - 1] = 0;
    if (strlen(buf) == 0)
        strncpy(buf, default_val, maxlen);
}

// 最近邻插值放大图片
void resize_28_to_256(const double *src, double *dst)
{
    int new_size = NEW_INPUT_SIZE;
    int old_size = OLD_INPUT_SIZE;
    for (int y = 0; y < new_size; ++y)
    {
        int src_y = y * old_size / new_size;
        for (int x = 0; x < new_size; ++x)
        {
            int src_x = x * old_size / new_size;
            dst[y * new_size + x] = src[src_y * old_size + src_x];
        }
    }
}

// MNIST数据加载
int load_mnist_images(const char *filename, double *arr, int max_count)
{
    LOG_INFO("Loading images from %s", filename);
    
    FILE *f = fopen(filename, "rb");
    if (!f)
    {
        LOG_ERROR("Error opening %s", filename);
        exit(1);
    }
    fseek(f, 16, SEEK_SET);
    int i, j;
    unsigned char pixel;
    for (i = 0; i < max_count; i++)
        for (j = 0; j < OLD_INPUT_SIZE * OLD_INPUT_SIZE; j++)
            if (fread(&pixel, 1, 1, f) != 1)
                goto END;
            else
                arr[i * OLD_INPUT_SIZE * OLD_INPUT_SIZE + j] = pixel / 255.0;
END:
    fclose(f);
    LOG_INFO("Loaded %d images", i);
    return i;
}

int load_mnist_labels(const char *filename, double *arr, int max_count)
{
    LOG_INFO("Loading labels from %s", filename);
    
    FILE *f = fopen(filename, "rb");
    if (!f)
    {
        LOG_ERROR("Error opening %s", filename);
        exit(1);
    }
    fseek(f, 8, SEEK_SET);
    int i, j;
    unsigned char label;
    for (i = 0; i < max_count; i++)
    {
        if (fread(&label, 1, 1, f) != 1)
            goto END;
        for (j = 0; j < OUTPUT_NODES; j++)
            arr[i * OUTPUT_NODES + j] = (j == label) ? 1.0 : 0.0;
    }
END:
    fclose(f);
    LOG_INFO("Loaded %d labels", i);
    return i;
}

// 权重文件保存 / 读取
void save_weights_biases(const char *file_name)
{
    LOG_INFO("Saving model to %s", file_name);
    
    FILE *file = fopen(file_name, "wb");
    if (!file)
    {
        LOG_ERROR("Error opening file to save model");
        exit(1);
    }
    fwrite(weight1, sizeof(double), INPUT_NODES * HIDDEN_NODES, file);
    fwrite(weight2, sizeof(double), HIDDEN_NODES * OUTPUT_NODES, file);
    fwrite(bias1, sizeof(double), HIDDEN_NODES, file);
    fwrite(bias2, sizeof(double), OUTPUT_NODES, file);
    fclose(file);
    
    LOG_INFO("Model saved successfully");
}

void load_weights_biases(const char *file_name)
{
    LOG_INFO("Loading model from %s", file_name);
    
    FILE *file = fopen(file_name, "rb");
    if (!file)
    {
        LOG_ERROR("Error opening file to load model");
        exit(1);
    }
    fread(weight1, sizeof(double), INPUT_NODES * HIDDEN_NODES, file);
    fread(weight2, sizeof(double), HIDDEN_NODES * OUTPUT_NODES, file);
    fread(bias1, sizeof(double), HIDDEN_NODES, file);
    fread(bias2, sizeof(double), OUTPUT_NODES, file);
    fclose(file);
    
    LOG_INFO("Model loaded successfully");
}

// 前向/反向/更新
void train(double *input, double *output, int correct_label)
{
    struct timespec t0, t1, t2, t3;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    double hidden[HIDDEN_NODES];
    double output_layer[OUTPUT_NODES];

    // 使用CUDA加速的前向传播
    forward_propagate(input, weight1, weight2, bias1, bias2, hidden, output_layer, 0);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    g_ff_time += diff_sec(t0, t1);

    if (max_index(output_layer, OUTPUT_NODES) == correct_label)
        forward_prob_output++;

    double error[OUTPUT_NODES], delta2[OUTPUT_NODES];
    for (int i = 0; i < OUTPUT_NODES; i++)
    {
        error[i] = output[i] - output_layer[i];
        delta2[i] = error[i] * output_layer[i] * (1 - output_layer[i]);
    }
    double delta1[HIDDEN_NODES] = {0};
    for (int i = 0; i < HIDDEN_NODES; i++)
    {
        double sum = 0;
        for (int j = 0; j < OUTPUT_NODES; j++)
            sum += delta2[j] * weight2[i * OUTPUT_NODES + j];
        delta1[i] = sum * relu_derivative(hidden[i]);
    }
    clock_gettime(CLOCK_MONOTONIC, &t2);
    g_bp_time += diff_sec(t1, t2);

    const double lr = 0.05;
    for (int i = 0; i < INPUT_NODES; i++)
        for (int j = 0; j < HIDDEN_NODES; j++)
            weight1[i * HIDDEN_NODES + j] += lr * delta1[j] * input[i];
    for (int i = 0; i < HIDDEN_NODES; i++)
    {
        bias1[i] += lr * delta1[i];
        for (int j = 0; j < OUTPUT_NODES; j++)
            weight2[i * OUTPUT_NODES + j] += lr * delta2[j] * hidden[i];
    }
    for (int i = 0; i < OUTPUT_NODES; i++)
        bias2[i] += lr * delta2[i];

    clock_gettime(CLOCK_MONOTONIC, &t3);
    g_wu_time += diff_sec(t2, t3);
}

// 测试
void test(double *input, int correct_label)
{
    double hidden[HIDDEN_NODES], output_layer[OUTPUT_NODES];
    
    // 使用CUDA加速的前向传播
    forward_propagate(input, weight1, weight2, bias1, bias2, hidden, output_layer, 0);
    
    if (max_index(output_layer, OUTPUT_NODES) == correct_label)
        correct_predictions++;
}

// 权重初始化
void init_weights()
{
    LOG_INFO("Initializing weights with Xavier initialization");
    
    srand(time(NULL));
    double w1_lim = sqrt(6.0 / (INPUT_NODES + HIDDEN_NODES));
    double w2_lim = sqrt(6.0 / (HIDDEN_NODES + OUTPUT_NODES));
    for (int i = 0; i < INPUT_NODES; i++)
        for (int j = 0; j < HIDDEN_NODES; j++)
            weight1[i * HIDDEN_NODES + j] = ((double)rand() / RAND_MAX * 2 - 1) * w1_lim;
    for (int i = 0; i < HIDDEN_NODES; i++)
        bias1[i] = 0;
    for (int i = 0; i < HIDDEN_NODES; i++)
        for (int j = 0; j < OUTPUT_NODES; j++)
            weight2[i * OUTPUT_NODES + j] = ((double)rand() / RAND_MAX * 2 - 1) * w2_lim;
    for (int i = 0; i < OUTPUT_NODES; i++)
        bias2[i] = 0;
    
    LOG_INFO("Weights initialized successfully");
}

// 训练模式
void train_mode()
{
    char train_img_path[MAX_PATH], train_label_path[MAX_PATH];
    char model_path[MAX_PATH], buf[32];
    int epochs = 10;

    input_string("Enter train image file path", train_img_path, MAX_PATH, DEFAULT_TRAIN_IMAGES);
    input_string("Enter train label file path", train_label_path, MAX_PATH, DEFAULT_TRAIN_LABELS);
    input_string("Enter model save path", model_path, MAX_PATH, DEFAULT_MODEL);
    printf("Enter number of epochs (default: 10): ");
    if (fgets(buf, 32, stdin) && buf[0] != '\n')
        epochs = atoi(buf);
    if (epochs < 1)
        epochs = 10;

    Timer load_timer = timer_start("Loading training data");
    int train_count = load_mnist_images(train_img_path, training_images, MAX_TRAIN);
    int label_count = load_mnist_labels(train_label_path, training_labels, MAX_TRAIN);
    double load_time = timer_stop(&load_timer);
    
    if (label_count != train_count)
    {
        LOG_ERROR("Image/Label count mismatch: %d images, %d labels", train_count, label_count);
        exit(1);
    }
    printf("Train set: imgs %d | labels %d | load %.3fs\n", train_count, label_count, load_time);

    LOG_INFO("Resizing images from 28x28 to 256x256");
    Timer resize_timer = timer_start("Image resizing");
    for (int i = 0; i < train_count; ++i)
        resize_28_to_256(training_images + i * OLD_INPUT_SIZE * OLD_INPUT_SIZE, training_images_resized + i * INPUT_NODES);
    timer_stop(&resize_timer);

    init_weights();

    LOG_INFO("Starting training for %d epochs", epochs);
    Timer total_timer = timer_start("Total training");
    double total_training_time = 0.0;
    
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        char epoch_desc[50];
        snprintf(epoch_desc, sizeof(epoch_desc), "Epoch %d/%d", epoch + 1, epochs);
        
        Timer epoch_timer = timer_start(epoch_desc);
        g_ff_time = g_bp_time = g_wu_time = 0;
        forward_prob_output = 0;
        
        progress_init(train_count, epoch_desc);
        
        for (int i = 0; i < train_count; i++)
        {
            progress_update(i, train_count);
            int correct_label = max_index(training_labels + i * OUTPUT_NODES, OUTPUT_NODES);
            train(training_images_resized + i * INPUT_NODES, training_labels + i * OUTPUT_NODES, correct_label);
        }
        
        double epoch_time = timer_stop(&epoch_timer);
        total_training_time += epoch_time;
        
        double accuracy = (double)forward_prob_output / train_count;
        LOG_INFO("Epoch %d details - FF: %.3fs | BP: %.3fs | WU: %.3fs", 
                 epoch + 1, g_ff_time, g_bp_time, g_wu_time);
        progress_finish(accuracy, forward_prob_output, train_count, epoch_time);
    }
    
    timer_stop(&total_timer);
    LOG_INFO("Average epoch time: %.2f seconds", total_training_time / epochs);
    
    save_weights_biases(model_path);
}

// 推理/测试模式
void infer_mode()
{
    char test_img_path[MAX_PATH], test_label_path[MAX_PATH];
    char model_path[MAX_PATH];
    input_string("Enter test image file path", test_img_path, MAX_PATH, DEFAULT_TEST_IMAGES);
    input_string("Enter test label file path", test_label_path, MAX_PATH, DEFAULT_TEST_LABELS);
    input_string("Enter model path", model_path, MAX_PATH, DEFAULT_MODEL);

    Timer load_timer = timer_start("Loading test data");
    int test_count = load_mnist_images(test_img_path, test_images, MAX_TEST);
    int label_count = load_mnist_labels(test_label_path, test_labels, MAX_TEST);
    timer_stop(&load_timer);
    
    if (label_count != test_count)
    {
        LOG_ERROR("Image/Label count mismatch: %d images, %d labels", test_count, label_count);
        exit(1);
    }

    LOG_INFO("Resizing images from 28x28 to 256x256");
    Timer resize_timer = timer_start("Image resizing");
    for (int i = 0; i < test_count; ++i)
        resize_28_to_256(test_images + i * OLD_INPUT_SIZE * OLD_INPUT_SIZE, test_images_resized + i * INPUT_NODES);
    timer_stop(&resize_timer);

    load_weights_biases(model_path);

    LOG_INFO("Testing model");
    Timer test_timer = timer_start("Model testing");
    correct_predictions = 0;
    
    progress_init(test_count, "Testing");
    
    for (int i = 0; i < test_count; i++)
    {
        progress_update(i, test_count);
        int correct_label = max_index(test_labels + i * OUTPUT_NODES, OUTPUT_NODES);
        test(test_images_resized + i * INPUT_NODES, correct_label);
    }
    
    double test_time = timer_stop(&test_timer);
    double accuracy = (double)correct_predictions / test_count;
    progress_finish(accuracy, correct_predictions, test_count, test_time);
}

// 经典一键 train+test
void classic_train_and_test()
{
    LOG_INFO("== Classic Train + Test ==");
    
    Timer train_load_timer = timer_start("Loading training data");
    int train_count = load_mnist_images(DEFAULT_TRAIN_IMAGES, training_images, MAX_TRAIN);
    int train_lbl = load_mnist_labels(DEFAULT_TRAIN_LABELS, training_labels, MAX_TRAIN);
    timer_stop(&train_load_timer);
    
    if (train_count != train_lbl)
    {
        LOG_ERROR("Train data mismatch: %d images, %d labels", train_count, train_lbl);
        exit(1);
    }

    Timer test_load_timer = timer_start("Loading test data");
    int test_count = load_mnist_images(DEFAULT_TEST_IMAGES, test_images, MAX_TEST);
    int test_lbl = load_mnist_labels(DEFAULT_TEST_LABELS, test_labels, MAX_TEST);
    timer_stop(&test_load_timer);
    
    if (test_count != test_lbl)
    {
        LOG_ERROR("Test data mismatch: %d images, %d labels", test_count, test_lbl);
        exit(1);
    }

    LOG_INFO("Resizing images from 28x28 to 256x256");
    Timer resize_timer = timer_start("Image resizing");
    for (int i = 0; i < train_count; ++i)
        resize_28_to_256(training_images + i * OLD_INPUT_SIZE * OLD_INPUT_SIZE, training_images_resized + i * INPUT_NODES);
    for (int i = 0; i < test_count; ++i)
        resize_28_to_256(test_images + i * OLD_INPUT_SIZE * OLD_INPUT_SIZE, test_images_resized + i * INPUT_NODES);
    timer_stop(&resize_timer);

    init_weights();

    int epochs = 10;
    LOG_INFO("Starting training for %d epochs", epochs);
    Timer total_timer = timer_start("Total training");
    double total_training_time = 0.0;
    
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        char epoch_desc[50];
        snprintf(epoch_desc, sizeof(epoch_desc), "Epoch %d/%d", epoch + 1, epochs);
        
        Timer epoch_timer = timer_start(epoch_desc);
        g_ff_time = g_bp_time = g_wu_time = 0;
        forward_prob_output = 0;
        
        progress_init(train_count, epoch_desc);
        
        for (int i = 0; i < train_count; i++)
        {
            progress_update(i, train_count);
            int correct_label = max_index(training_labels + i * OUTPUT_NODES, OUTPUT_NODES);
            train(training_images_resized + i * INPUT_NODES, training_labels + i * OUTPUT_NODES, correct_label);
        }
        
        double epoch_time = timer_stop(&epoch_timer);
        total_training_time += epoch_time;
        
        double accuracy = (double)forward_prob_output / train_count;
        LOG_INFO("Epoch %d details - FF: %.3fs | BP: %.3fs | WU: %.3fs", 
                 epoch + 1, g_ff_time, g_bp_time, g_wu_time);
        progress_finish(accuracy, forward_prob_output, train_count, epoch_time);
    }
    
    timer_stop(&total_timer);
    LOG_INFO("Average epoch time: %.2f seconds", total_training_time / epochs);
    
    save_weights_biases(DEFAULT_MODEL);

    LOG_INFO("Testing model");
    Timer test_timer = timer_start("Model testing");
    correct_predictions = 0;
    
    progress_init(test_count, "Testing");
    
    for (int i = 0; i < test_count; i++)
    {
        progress_update(i, test_count);
        int correct_label = max_index(test_labels + i * OUTPUT_NODES, OUTPUT_NODES);
        test(test_images_resized + i * INPUT_NODES, correct_label);
    }
    
    double test_time = timer_stop(&test_timer);
    double accuracy = (double)correct_predictions / test_count;
    progress_finish(accuracy, correct_predictions, test_count, test_time);
}

// 主程序
int main()
{
    training_images = (double *)malloc(sizeof(double) * MAX_TRAIN * OLD_INPUT_SIZE * OLD_INPUT_SIZE);
    CHECK_PTR(training_images, "training_images");
    test_images = (double *)malloc(sizeof(double) * MAX_TEST * OLD_INPUT_SIZE * OLD_INPUT_SIZE);
    CHECK_PTR(test_images, "test_images");
    training_images_resized = (double *)malloc(sizeof(double) * MAX_TRAIN * INPUT_NODES);
    CHECK_PTR(training_images_resized, "training_images_resized");
    test_images_resized = (double *)malloc(sizeof(double) * MAX_TEST * INPUT_NODES);
    CHECK_PTR(test_images_resized, "test_images_resized");
    training_labels = (double *)malloc(sizeof(double) * MAX_TRAIN * OUTPUT_NODES);
    CHECK_PTR(training_labels, "training_labels");
    test_labels = (double *)malloc(sizeof(double) * MAX_TEST * OUTPUT_NODES);
    CHECK_PTR(test_labels, "test_labels");
    weight1 = (double *)malloc(sizeof(double) * INPUT_NODES * HIDDEN_NODES);
    CHECK_PTR(weight1, "weight1");
    weight2 = (double *)malloc(sizeof(double) * HIDDEN_NODES * OUTPUT_NODES);
    CHECK_PTR(weight2, "weight2");
    bias1 = (double *)malloc(sizeof(double) * HIDDEN_NODES);
    CHECK_PTR(bias1, "bias1");
    bias2 = (double *)malloc(sizeof(double) * OUTPUT_NODES);
    CHECK_PTR(bias2, "bias2");

    LOG_INFO("===== Deep Neural Network (C版, 256x256输入, CUDA加速前向传播) =====");
    LOG_INFO("1. 训练（自定义参数和路径）");
    LOG_INFO("2. 推理/测试（自定义模型和测试集路径）");
    LOG_INFO("3. 经典一键训练+推理（默认路径） [默认选项]");
    printf("请选择 (1/2/3, 回车默认3): ");

    char choice[8];
    if (fgets(choice, 8, stdin) == NULL || choice[0] == '\n')
        choice[0] = '3';

    if (choice[0] == '1')
        train_mode();
    else if (choice[0] == '2')
        infer_mode();
    else
        classic_train_and_test();

    free(training_images);
    free(test_images);
    free(training_images_resized);
    free(test_images_resized);
    free(training_labels);
    free(test_labels);
    free(weight1);
    free(weight2);
    free(bias1);
    free(bias2);

    return 0;
}
