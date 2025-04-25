#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_NODES 784  // 28*28 pixels
#define HIDDEN_NODES 256 // Number of hidden nodes
#define OUTPUT_NODES 10  // 10 digits (0-9)
#define NUM_TRAINING_IMAGES 60000
#define NUM_TEST_IMAGES 10000
#define NUMBER_OF_EPOCHS 10

// 数据集全局变量
double training_images[NUM_TRAINING_IMAGES][INPUT_NODES];
double training_labels[NUM_TRAINING_IMAGES][OUTPUT_NODES];
double test_images[NUM_TEST_IMAGES][INPUT_NODES];
double test_labels[NUM_TEST_IMAGES][OUTPUT_NODES];

// 权重和偏置
double weight1[INPUT_NODES][HIDDEN_NODES];
double weight2[HIDDEN_NODES][OUTPUT_NODES];
double bias1[HIDDEN_NODES];
double bias2[OUTPUT_NODES];

// 正确预测数
int correct_predictions;
int forward_prob_output;

// ReLU激活
double relu(double x) { return x > 0 ? x : 0; }
double relu_derivative(double x) { return x > 0 ? 1 : 0; }

// Sigmoid激活（用于输出层）
double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

// 取最大值索引
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

// 加载MNIST数据集
void load_mnist()
{
    FILE *training_images_file = fopen("mnist_train_images.bin", "rb");
    FILE *training_labels_file = fopen("mnist_train_labels.bin", "rb");
    FILE *test_images_file = fopen("mnist_test_images.bin", "rb");
    FILE *test_labels_file = fopen("mnist_test_labels.bin", "rb");
    if (!training_images_file || !training_labels_file || !test_images_file || !test_labels_file)
    {
        printf("Error opening dataset files\n");
        exit(1);
    }
    // 跳过MNIST官方文件头
    fseek(training_images_file, 16, SEEK_SET);
    fseek(training_labels_file, 8, SEEK_SET);
    fseek(test_images_file, 16, SEEK_SET);
    fseek(test_labels_file, 8, SEEK_SET);

    // 读取训练集
    for (int i = 0; i < NUM_TRAINING_IMAGES; i++)
    {
        for (int j = 0; j < INPUT_NODES; j++)
        {
            unsigned char pixel;
            fread(&pixel, sizeof(unsigned char), 1, training_images_file);
            training_images[i][j] = (double)pixel / 255.0;
        }
    }
    for (int i = 0; i < NUM_TRAINING_IMAGES; i++)
    {
        unsigned char label;
        fread(&label, sizeof(unsigned char), 1, training_labels_file);
        for (int j = 0; j < OUTPUT_NODES; j++)
        {
            training_labels[i][j] = (j == label) ? 1.0 : 0.0;
        }
    }
    // 读取测试集
    for (int i = 0; i < NUM_TEST_IMAGES; i++)
    {
        for (int j = 0; j < INPUT_NODES; j++)
        {
            unsigned char pixel;
            fread(&pixel, sizeof(unsigned char), 1, test_images_file);
            test_images[i][j] = (double)pixel / 255.0;
        }
    }
    for (int i = 0; i < NUM_TEST_IMAGES; i++)
    {
        unsigned char label;
        fread(&label, sizeof(unsigned char), 1, test_labels_file);
        for (int j = 0; j < OUTPUT_NODES; j++)
        {
            test_labels[i][j] = (j == label) ? 1.0 : 0.0;
        }
    }
    fclose(training_images_file);
    fclose(training_labels_file);
    fclose(test_images_file);
    fclose(test_labels_file);
}

// 保存/加载权重与偏置
void save_weights_biases(char *file_name)
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
void load_weights_biases(char *file_name)
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

// 训练单张图片
void train(double input[INPUT_NODES], double output[OUTPUT_NODES],
           double weight1[INPUT_NODES][HIDDEN_NODES], double weight2[HIDDEN_NODES][OUTPUT_NODES],
           double bias1[HIDDEN_NODES], double bias2[OUTPUT_NODES], int correct_label)
{
    double hidden[HIDDEN_NODES];
    double output_layer[OUTPUT_NODES];

    // Forward
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

    // Backward
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
    // 更新参数
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

// 测试单张图片
void test(double input[INPUT_NODES],
          double weight1[INPUT_NODES][HIDDEN_NODES], double weight2[HIDDEN_NODES][OUTPUT_NODES],
          double bias1[HIDDEN_NODES], double bias2[OUTPUT_NODES], int correct_label)
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
    // printf("Prediction: %d\n", index); // 若不需要每张都打印可注释
    if (index == correct_label)
        correct_predictions++;
}

// 主函数
int main()
{
    srand(time(NULL)); // 随机种子
    // Xavier初始化
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

    printf("Loading MNIST data...\n");
    load_mnist();
    printf("Training...\n");

    // 训练
    for (int epoch = 0; epoch < NUMBER_OF_EPOCHS; epoch++)
    {
        forward_prob_output = 0;
        for (int i = 0; i < NUM_TRAINING_IMAGES; i++)
        {
            int correct_label = max_index(training_labels[i], OUTPUT_NODES);
            train(training_images[i], training_labels[i], weight1, weight2, bias1, bias2, correct_label);
        }
        printf("Epoch %d : Training Accuracy: %.4f\n", epoch + 1, (double)forward_prob_output / NUM_TRAINING_IMAGES);
    }
    save_weights_biases("model.bin");
    printf("Model saved as model.bin\n");

    // 测试
    correct_predictions = 0;
    for (int i = 0; i < NUM_TEST_IMAGES; i++)
    {
        int correct_label = max_index(test_labels[i], OUTPUT_NODES);
        test(test_images[i], weight1, weight2, bias1, bias2, correct_label);
    }
    printf("Testing Accuracy: %.4f\n", (double)correct_predictions / NUM_TEST_IMAGES);
    return 0;
}
