#ifndef NN_H
#define NN_H

#include <cuda_runtime.h>

// Neural network constants matching new.c definitions
#define OLD_INPUT_SIZE 28
#define NEW_INPUT_SIZE 256
#define INPUT_NODES (NEW_INPUT_SIZE * NEW_INPUT_SIZE)
#define HIDDEN_NODES 256
#define OUTPUT_NODES 10

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the CUDA training environment with support for up to max_batch_size
cudaError_t init_fused_training(int max_batch_size);

// Clean up all CUDA resources
void cleanup_fused_training();

// Main training function that processes a batch of images
void train_batch_fused(
    const float* batch_input,
    const float* batch_targets,
    float* weight1,
    float* weight2, 
    float* bias1,
    float* bias2,
    int batch_size,
    float learning_rate,
    int* correct_predictions
);

#ifdef __cplusplus
}
#endif

#endif // NN_H 