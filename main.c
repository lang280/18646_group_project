#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// 超参数
#define INPUT_NODES 784
#define HIDDEN_NODES 256
#define OUTPUT_NODES 10
#define MAX_PATH 256
#define DEFAULT_TRAIN_IMAGES "mnist_train_images.bin"
#define DEFAULT_TRAIN_LABELS "mnist_train_labels.bin"
#define DEFAULT_TEST_IMAGES "mnist_test_images.bin"
#define DEFAULT_TEST_LABELS "mnist_test_labels.bin"
#define DEFAULT_MODEL "model.bin"

// 为了灵活，最大只分配空间，实际数据量可自定义（下同）
#define MAX_TRAIN 60000
#define MAX_TEST 10000

// 数据全局变量
double training_images[MAX_TRAIN][INPUT_NODES];
double training_labels[MAX_TRAIN][OUTPUT_NODES];
double test_images[MAX_TEST][INPUT_NODES];
double test_labels[MAX_TEST][OUTPUT_NODES];

// 权重和偏置
double weight1[INPUT_NODES][HIDDEN_NODES];
double weight2[HIDDEN_NODES][OUTPUT_NODES];
double bias1[HIDDEN_NODES];
double bias2[OUTPUT_NODES];

int correct_predictions;
int forward_prob_output;

// ReLU及其导数
double relu(double x) { return x > 0 ? x : 0; }
double relu_derivative(double x) { return x > 0 ? 1 : 0; }
double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
int max_index(double arr[], int size)
{
    int max_i = 0;
    for (int i = 1; i < size; i++)
    {
        if (arr[i] > arr[max_i])
            max_i = i;
    }
    return max_i;
}

// 读取一行字符串到buffer，并去除末尾换行
void input_string(const char *prompt, char *buf, int maxlen, const char *default_val)
{
    printf("%s (default: %s): ", prompt, default_val);
    if (fgets(buf, maxlen, stdin) == NULL)
        buf[0] = 0;
    int len = strlen(buf);
    if (len > 0 && buf[len - 1] == '\n')
        buf[len - 1] = 0;
    if (strlen(buf) == 0)
        strncpy(buf, default_val, maxlen);
}

// 加载MNIST二进制文件
int load_mnist_images(const char *filename, double arr[][INPUT_NODES], int max_count)
{
    FILE *f = fopen(filename, "rb");
    if (!f)
    {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(f, 16, SEEK_SET);
    int i;
    for (i = 0; i < max_count; i++)
    {
        for (int j = 0; j < INPUT_NODES; j++)
        {
            unsigned char pixel;
            if (fread(&pixel, sizeof(unsigned char), 1, f) != 1)
                goto END;
            arr[i][j] = (double)pixel / 255.0;
        }
    }
END:
    fclose(f);
    return i; // 实际读取数量
}
int load_mnist_labels(const char *filename, double arr[][OUTPUT_NODES], int max_count)
{
    FILE *f = fopen(filename, "rb");
    if (!f)
    {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(f, 8, SEEK_SET);
    int i;
    for (i = 0; i < max_count; i++)
    {
        unsigned char label;
        if (fread(&label, sizeof(unsigned char), 1, f) != 1)
            goto END;
        for (int j = 0; j < OUTPUT_NODES; j++)
            arr[i][j] = (j == label) ? 1.0 : 0.0;
    }
END:
    fclose(f);
    return i; // 实际读取数量
}

void save_weights_biases(const char *file_name)
{
    FILE *file = fopen(file_name, "wb");
    if (!file)
    {
        printf("Error opening file to save model\n");
        exit(1);
    }
    fwrite(weight1, sizeof(double), INPUT_NODES * HIDDEN_NODES, file);
    fwrite(weight2, sizeof(double), HIDDEN_NODES * OUTPUT_NODES, file);
    fwrite(bias1, sizeof(double), HIDDEN_NODES, file);
    fwrite(bias2, sizeof(double), OUTPUT_NODES, file);
    fclose(file);
}
void load_weights_biases(const char *file_name)
{
    FILE *file = fopen(file_name, "rb");
    if (!file)
    {
        printf("Error opening file to load model\n");
        exit(1);
    }
    fread(weight1, sizeof(double), INPUT_NODES * HIDDEN_NODES, file);
    fread(weight2, sizeof(double), HIDDEN_NODES * OUTPUT_NODES, file);
    fread(bias1, sizeof(double), HIDDEN_NODES, file);
    fread(bias2, sizeof(double), OUTPUT_NODES, file);
    fclose(file);
}

void train(double input[INPUT_NODES], double output[OUTPUT_NODES], int correct_label)
{
    double hidden[HIDDEN_NODES];
    double output_layer[OUTPUT_NODES];

    // 前向传播
    for (int i = 0; i < HIDDEN_NODES; i++)
    {
        double sum = bias1[i];
        for (int j = 0; j < INPUT_NODES; j++)
            sum += input[j] * weight1[j][i];
        hidden[i] = relu(sum);
    }
    for (int i = 0; i < OUTPUT_NODES; i++)
    {
        double sum = bias2[i];
        for (int j = 0; j < HIDDEN_NODES; j++)
            sum += hidden[j] * weight2[j][i];
        output_layer[i] = sigmoid(sum);
    }
    int index = max_index(output_layer, OUTPUT_NODES);
    if (index == correct_label)
        forward_prob_output++;

    // 反向传播
    double error[OUTPUT_NODES];
    for (int i = 0; i < OUTPUT_NODES; i++)
        error[i] = output[i] - output_layer[i];
    double delta2[OUTPUT_NODES];
    for (int i = 0; i < OUTPUT_NODES; i++)
        delta2[i] = error[i] * output_layer[i] * (1 - output_layer[i]);
    double delta1[HIDDEN_NODES] = {0};
    for (int i = 0; i < HIDDEN_NODES; i++)
    {
        double sum = 0;
        for (int j = 0; j < OUTPUT_NODES; j++)
            sum += delta2[j] * weight2[i][j];
        delta1[i] = sum * relu_derivative(hidden[i]);
    }
    double learning_rate = 0.05;
    for (int i = 0; i < INPUT_NODES; i++)
        for (int j = 0; j < HIDDEN_NODES; j++)
            weight1[i][j] += learning_rate * delta1[j] * input[i];
    for (int i = 0; i < HIDDEN_NODES; i++)
    {
        bias1[i] += learning_rate * delta1[i];
        for (int j = 0; j < OUTPUT_NODES; j++)
            weight2[i][j] += learning_rate * delta2[j] * hidden[i];
    }
    for (int i = 0; i < OUTPUT_NODES; i++)
        bias2[i] += learning_rate * delta2[i];
}

void test(double input[INPUT_NODES], int correct_label)
{
    double hidden[HIDDEN_NODES];
    double output_layer[OUTPUT_NODES];
    for (int i = 0; i < HIDDEN_NODES; i++)
    {
        double sum = bias1[i];
        for (int j = 0; j < INPUT_NODES; j++)
            sum += input[j] * weight1[j][i];
        hidden[i] = relu(sum);
    }
    for (int i = 0; i < OUTPUT_NODES; i++)
    {
        double sum = bias2[i];
        for (int j = 0; j < HIDDEN_NODES; j++)
            sum += hidden[j] * weight2[j][i];
        output_layer[i] = sigmoid(sum);
    }
    int index = max_index(output_layer, OUTPUT_NODES);
    if (index == correct_label)
        correct_predictions++;
}

void init_weights()
{
    srand(time(NULL));
    double w1_limit = sqrt(6.0 / (INPUT_NODES + HIDDEN_NODES));
    double w2_limit = sqrt(6.0 / (HIDDEN_NODES + OUTPUT_NODES));
    for (int i = 0; i < INPUT_NODES; i++)
        for (int j = 0; j < HIDDEN_NODES; j++)
            weight1[i][j] = ((double)rand() / RAND_MAX * 2 - 1) * w1_limit;
    for (int i = 0; i < HIDDEN_NODES; i++)
        for (int j = 0; j < OUTPUT_NODES; j++)
            weight2[i][j] = ((double)rand() / RAND_MAX * 2 - 1) * w2_limit;
    for (int i = 0; i < HIDDEN_NODES; i++)
        bias1[i] = 0;
    for (int i = 0; i < OUTPUT_NODES; i++)
        bias2[i] = 0;
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

    printf("Loading train images...\n");
    int train_count = load_mnist_images(train_img_path, training_images, MAX_TRAIN);
    printf("Loaded %d training images\n", train_count);
    printf("Loading train labels...\n");
    int label_count = load_mnist_labels(train_label_path, training_labels, MAX_TRAIN);
    if (label_count != train_count)
    {
        printf("Image/Label count mismatch!\n");
        exit(1);
    }
    printf("Loaded %d training labels\n", label_count);

    printf("Initializing weights...\n");
    init_weights();

    printf("Start Training...\n");
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        forward_prob_output = 0;
        for (int i = 0; i < train_count; i++)
        {
            int correct_label = max_index(training_labels[i], OUTPUT_NODES);
            train(training_images[i], training_labels[i], correct_label);
        }
        printf("Epoch %d : Training Accuracy: %.4f\n", epoch + 1, (double)forward_prob_output / train_count);
    }
    save_weights_biases(model_path);
    printf("Model saved to %s\n", model_path);
}

// 推理/测试模式
void infer_mode()
{
    char test_img_path[MAX_PATH], test_label_path[MAX_PATH];
    char model_path[MAX_PATH];
    input_string("Enter test image file path", test_img_path, MAX_PATH, DEFAULT_TEST_IMAGES);
    input_string("Enter test label file path", test_label_path, MAX_PATH, DEFAULT_TEST_LABELS);
    input_string("Enter model path", model_path, MAX_PATH, DEFAULT_MODEL);

    printf("Loading test images...\n");
    int test_count = load_mnist_images(test_img_path, test_images, MAX_TEST);
    printf("Loaded %d test images\n", test_count);
    printf("Loading test labels...\n");
    int label_count = load_mnist_labels(test_label_path, test_labels, MAX_TEST);
    if (label_count != test_count)
    {
        printf("Image/Label count mismatch!\n");
        exit(1);
    }
    printf("Loaded %d test labels\n", label_count);

    printf("Loading model...\n");
    load_weights_biases(model_path);

    correct_predictions = 0;
    for (int i = 0; i < test_count; i++)
    {
        int correct_label = max_index(test_labels[i], OUTPUT_NODES);
        test(test_images[i], correct_label);
    }
    printf("Testing Accuracy: %.4f\n", (double)correct_predictions / test_count);
}

// 一键训练+推理
void classic_train_and_test()
{
    printf("== Classic Train + Test ==\n");
    printf("Loading training set...\n");
    int train_count = load_mnist_images(DEFAULT_TRAIN_IMAGES, training_images, MAX_TRAIN);
    load_mnist_labels(DEFAULT_TRAIN_LABELS, training_labels, MAX_TRAIN);
    printf("Loaded %d training images\n", train_count);

    printf("Loading test set...\n");
    int test_count = load_mnist_images(DEFAULT_TEST_IMAGES, test_images, MAX_TEST);
    load_mnist_labels(DEFAULT_TEST_LABELS, test_labels, MAX_TEST);
    printf("Loaded %d test images\n", test_count);

    printf("Initializing weights...\n");
    init_weights();

    printf("Start Training...\n");
    int epochs = 10;
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        forward_prob_output = 0;
        for (int i = 0; i < train_count; i++)
        {
            int correct_label = max_index(training_labels[i], OUTPUT_NODES);
            train(training_images[i], training_labels[i], correct_label);
        }
        printf("Epoch %d : Training Accuracy: %.4f\n", epoch + 1, (double)forward_prob_output / train_count);
    }
    save_weights_biases(DEFAULT_MODEL);
    printf("Model saved to %s\n", DEFAULT_MODEL);

    correct_predictions = 0;
    for (int i = 0; i < test_count; i++)
    {
        int correct_label = max_index(test_labels[i], OUTPUT_NODES);
        test(test_images[i], correct_label);
    }
    printf("Testing Accuracy: %.4f\n", (double)correct_predictions / test_count);
}

int main()
{
    printf("===== Deep Neural Network (C版) - 菜单模式 =====\n");
    printf("1. 训练（自定义参数和路径）\n");
    printf("2. 推理/测试（自定义模型和测试集路径）\n");
    printf("3. 经典一键训练+推理（默认路径） [默认选项]\n");
    printf("请选择 (1/2/3, 回车默认3): ");

    char choice[8];
    if (fgets(choice, 8, stdin) == NULL || choice[0] == '\n')
        choice[0] = '3';
    if (choice[0] == '1')
    {
        train_mode();
    }
    else if (choice[0] == '2')
    {
        infer_mode();
    }
    else
    {
        classic_train_and_test();
    }
    return 0;
}
