#ifndef FUSED_H
#define FUSED_H

// Matching constants from existing headers
#define OLD_INPUT_SIZE 28
#define NEW_INPUT_SIZE 256
#define INPUT_NODES (NEW_INPUT_SIZE * NEW_INPUT_SIZE)
#define HIDDEN_NODES 256
#define OUTPUT_NODES 10

// Batch processing constants
#define DEFAULT_BATCH_SIZE 32

#ifdef __cplusplus
extern "C" {
#endif

void cleanup_fused_training();

/**
 * Performs a fused forward and backward pass on a batch of input images.
 * This function handles the entire training process for a batch, including:
 * 1. Forward propagation (input -> hidden -> output)
 * 2. Error calculation
 * 3. Backward propagation (calculate deltas)
 * 4. Weight and bias updates
 * 
 * @param batch_input Pointer to batch of input images [batch_size * INPUT_NODES]
 * @param batch_targets Pointer to batch of target outputs [batch_size * OUTPUT_NODES]
 * @param weight1 Pointer to weight matrix between input and hidden layer [INPUT_NODES * HIDDEN_NODES]
 * @param weight2 Pointer to weight matrix between hidden and output layer [HIDDEN_NODES * OUTPUT_NODES]
 * @param bias1 Pointer to bias vector for hidden layer [HIDDEN_NODES]
 * @param bias2 Pointer to bias vector for output layer [OUTPUT_NODES]
 * @param batch_size Number of images in the batch
 * @param learning_rate Learning rate for weight updates
 * @param correct_predictions Output parameter: will be incremented by the number of correct predictions in the batch
 */
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

#endif /* FUSED_H */ 