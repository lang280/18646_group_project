#include "nn.h"
#include "forward.h"
#include "backward.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <time.h>

// Logging control
#define LOG_FUSED 1

// Error checking macro for CUDA operations
#define CUDA_CHECK(operation, description) \
    do { \
        cudaStatus = (operation); \
        if (cudaStatus != cudaSuccess) { \
            fprintf(stderr, "%s failed: %s\n", (description), cudaGetErrorString(cudaStatus)); \
            goto Error; \
        } \
    } while(0)

// CUDA thread/block parameters
#define THREADS_PER_BLOCK 256
#define MAX_BLOCKS 65535

// ────────── Persistent device memory pointers ──────────
// These are allocated once and reused for all batches
static float *d_batch_input = NULL;
static float *d_batch_hidden = NULL; 
static float *d_batch_output = NULL;
static float *d_batch_targets = NULL;
static float *d_batch_hidden_delta = NULL;
static float *d_batch_output_delta = NULL;
static float *d_weight1 = NULL;
static float *d_weight2 = NULL;
static float *d_bias1 = NULL;
static float *d_bias2 = NULL;
static int *d_correct_count = NULL;
static cudaStream_t cuda_stream = NULL;

// State tracking
static bool is_initialized = false;
static int current_max_batch_size = 0;

// ────────── Initialization and cleanup functions ──────────

// Initialize the CUDA training environment
extern "C"
cudaError_t init_fused_training(int max_batch_size) {
    if (is_initialized && max_batch_size <= current_max_batch_size) {
        // Already initialized with sufficient capacity
        return cudaSuccess;
    }
    
    // Cleanup previous resources if any
    cleanup_fused_training();
    
    cudaError_t cudaStatus;
    
    // Create stream
    cudaStreamCreate(&cuda_stream);
    
    // Allocate device memory with capacity for the maximum batch size
    size_t batch_input_size = max_batch_size * INPUT_NODES * sizeof(float);
    size_t batch_hidden_size = max_batch_size * HIDDEN_NODES * sizeof(float);
    size_t batch_output_size = max_batch_size * OUTPUT_NODES * sizeof(float);
    size_t weight1_size = INPUT_NODES * HIDDEN_NODES * sizeof(float);
    size_t weight2_size = HIDDEN_NODES * OUTPUT_NODES * sizeof(float);
    size_t bias1_size = HIDDEN_NODES * sizeof(float);
    size_t bias2_size = OUTPUT_NODES * sizeof(float);
    
    // Allocate memory for inputs, activations, deltas
    CUDA_CHECK(cudaMalloc((void**)&d_batch_input, batch_input_size), "cudaMalloc for batch input");
    CUDA_CHECK(cudaMalloc((void**)&d_batch_hidden, batch_hidden_size), "cudaMalloc for batch hidden");
    CUDA_CHECK(cudaMalloc((void**)&d_batch_output, batch_output_size), "cudaMalloc for batch output");
    CUDA_CHECK(cudaMalloc((void**)&d_batch_targets, batch_output_size), "cudaMalloc for batch targets");
    CUDA_CHECK(cudaMalloc((void**)&d_batch_hidden_delta, batch_hidden_size), "cudaMalloc for hidden delta");
    CUDA_CHECK(cudaMalloc((void**)&d_batch_output_delta, batch_output_size), "cudaMalloc for output delta");
    
    // Allocate memory for weights and biases
    CUDA_CHECK(cudaMalloc((void**)&d_weight1, weight1_size), "cudaMalloc for weight1");
    CUDA_CHECK(cudaMalloc((void**)&d_weight2, weight2_size), "cudaMalloc for weight2");
    CUDA_CHECK(cudaMalloc((void**)&d_bias1, bias1_size), "cudaMalloc for bias1");
    CUDA_CHECK(cudaMalloc((void**)&d_bias2, bias2_size), "cudaMalloc for bias2");
    
    // Allocate memory for correct prediction count
    CUDA_CHECK(cudaMalloc((void**)&d_correct_count, sizeof(int)), "cudaMalloc for correct count");
    
    // Update state
    is_initialized = true;
    current_max_batch_size = max_batch_size;
    
    return cudaSuccess;
    
Error:
    // Cleanup on error
    cleanup_fused_training();
    return cudaStatus;
}

// Cleanup all CUDA resources
extern "C"
void cleanup_fused_training() {
    // Free device memory
    if (d_batch_input) cudaFree(d_batch_input);
    if (d_batch_hidden) cudaFree(d_batch_hidden);
    if (d_batch_output) cudaFree(d_batch_output);
    if (d_batch_targets) cudaFree(d_batch_targets);
    if (d_batch_hidden_delta) cudaFree(d_batch_hidden_delta);
    if (d_batch_output_delta) cudaFree(d_batch_output_delta);
    if (d_weight1) cudaFree(d_weight1);
    if (d_weight2) cudaFree(d_weight2);
    if (d_bias1) cudaFree(d_bias1);
    if (d_bias2) cudaFree(d_bias2);
    if (d_correct_count) cudaFree(d_correct_count);
    
    // Destroy stream
    if (cuda_stream) cudaStreamDestroy(cuda_stream);
    
    // Reset pointers
    d_batch_input = NULL;
    d_batch_hidden = NULL;
    d_batch_output = NULL;
    d_batch_targets = NULL;
    d_batch_hidden_delta = NULL;
    d_batch_output_delta = NULL;
    d_weight1 = NULL;
    d_weight2 = NULL;
    d_bias1 = NULL;
    d_bias2 = NULL;
    d_correct_count = NULL;
    cuda_stream = NULL;
    
    // Reset state
    is_initialized = false;
    current_max_batch_size = 0;
}

// ────────── Main fused training function ──────────

extern "C"
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
)
{
    // For performance logging
    static int call_count = 0;
    static double total_time = 0.0;
    clock_t start_time = 0, end_time = 0;
    
    // Only log occasionally
    bool should_log = LOG_FUSED && (call_count % 10 == 0);
    
    if (should_log) {
        start_time = clock();
        printf("[FUSED] Starting batch #%d with %d images\n", call_count, batch_size);
    }
    
    // CUDA error status
    cudaError_t cudaStatus;
    
    // Host-side counter for correct predictions
    int host_correct_count = 0;
    
    // Create CUDA events for kernel timing - declare at the beginning to fix control flow issues
    cudaEvent_t kernel_start = NULL;
    cudaEvent_t kernel_stop = NULL;
    float kernel_time = 0.0f;
    
    // Grid dimensions for kernels - declare at the beginning
    int total_hidden_neurons = 0;
    int hidden_blocks = 0;
    int total_output_neurons = 0;
    int output_blocks = 0;
    int prediction_blocks = 0;
    int input_hidden_blocks = 0;
    int hidden_output_blocks = 0;
    
    // Ensure environment is initialized
    if (!is_initialized || batch_size > current_max_batch_size) {
        if (should_log) {
            printf("[FUSED] Initializing training environment with batch size %d\n", batch_size);
        }
        init_fused_training(batch_size);
    }
    
    // Reset correct count
    cudaMemset(d_correct_count, 0, sizeof(int));
    
    if (should_log) {
        printf("[FUSED] Copying data to device...\n");
    }
    
    // Copy input data, weights, and biases to device
    size_t batch_input_size = batch_size * INPUT_NODES * sizeof(float);
    size_t batch_output_size = batch_size * OUTPUT_NODES * sizeof(float);
    size_t weight1_size = INPUT_NODES * HIDDEN_NODES * sizeof(float);
    size_t weight2_size = HIDDEN_NODES * OUTPUT_NODES * sizeof(float);
    size_t bias1_size = HIDDEN_NODES * sizeof(float);
    size_t bias2_size = OUTPUT_NODES * sizeof(float);
    
    CUDA_CHECK(cudaMemcpy(d_batch_input, batch_input, batch_input_size, cudaMemcpyHostToDevice), "cudaMemcpy for batch input");
    CUDA_CHECK(cudaMemcpy(d_batch_targets, batch_targets, batch_output_size, cudaMemcpyHostToDevice), "cudaMemcpy for batch targets");
    CUDA_CHECK(cudaMemcpy(d_weight1, weight1, weight1_size, cudaMemcpyHostToDevice), "cudaMemcpy for weight1");
    CUDA_CHECK(cudaMemcpy(d_weight2, weight2, weight2_size, cudaMemcpyHostToDevice), "cudaMemcpy for weight2");
    CUDA_CHECK(cudaMemcpy(d_bias1, bias1, bias1_size, cudaMemcpyHostToDevice), "cudaMemcpy for bias1");
    CUDA_CHECK(cudaMemcpy(d_bias2, bias2, bias2_size, cudaMemcpyHostToDevice), "cudaMemcpy for bias2");

    // Execute operations with CUDA streams
    // Create CUDA events for kernel timing
    CUDA_CHECK(cudaEventCreate(&kernel_start), "Create kernel start event");
    CUDA_CHECK(cudaEventCreate(&kernel_stop), "Create kernel stop event");
    
    // ────────── Forward Pass ──────────
    if (should_log) {
        printf("[FUSED] Performing forward pass...\n");
    }
    
    // Calculate grid dimensions for hidden layer
    total_hidden_neurons = batch_size * HIDDEN_NODES;
    hidden_blocks = (total_hidden_neurons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    hidden_blocks = hidden_blocks > MAX_BLOCKS ? MAX_BLOCKS : hidden_blocks;
    
    // Time hidden layer kernel
    CUDA_CHECK(cudaEventRecord(kernel_start, cuda_stream), "Record hidden layer start event");
    
    // Launch hidden layer kernel
    hidden_layer_kernel_batch_vectorized4<<<hidden_blocks, THREADS_PER_BLOCK, HIDDEN_NODES * sizeof(float), cuda_stream>>>(
        d_batch_input, d_weight1, d_bias1, d_batch_hidden, 
        INPUT_NODES, HIDDEN_NODES, batch_size
    );
    
    CUDA_CHECK(cudaEventRecord(kernel_stop, cuda_stream), "Record hidden layer stop event");
    CUDA_CHECK(cudaEventSynchronize(kernel_stop), "Synchronize hidden layer stop event");
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop), "Calculate hidden layer time");
    if (should_log) printf("[TIMING] Hidden layer kernel: %.4f ms\n", kernel_time);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError(), "hidden_layer_kernel_vectorized4 launch");
    
    // Calculate grid dimensions for output layer
    total_output_neurons = batch_size * OUTPUT_NODES;
    output_blocks = (total_output_neurons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    output_blocks = output_blocks > MAX_BLOCKS ? MAX_BLOCKS : output_blocks;
    
    // Time output layer kernel
    CUDA_CHECK(cudaEventRecord(kernel_start, cuda_stream), "Record output layer start event");
    
    // Launch output layer kernel
    output_layer_kernel_batch<<<output_blocks, THREADS_PER_BLOCK, OUTPUT_NODES * sizeof(float), cuda_stream>>>(
        d_batch_hidden, d_weight2, d_bias2, d_batch_output, 
        HIDDEN_NODES, OUTPUT_NODES, batch_size
    );
    
    CUDA_CHECK(cudaEventRecord(kernel_stop, cuda_stream), "Record output layer stop event");
    CUDA_CHECK(cudaEventSynchronize(kernel_stop), "Synchronize output layer stop event");
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop), "Calculate output layer time");
    if (should_log) printf("[TIMING] Output layer kernel: %.4f ms\n", kernel_time);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError(), "output_layer_kernel_batch launch");
    
    // ────────── Count Correct Predictions ──────────
    if (should_log) {
        printf("[FUSED] Counting correct predictions...\n");
    }
    
    // Time prediction kernel
    CUDA_CHECK(cudaEventRecord(kernel_start, cuda_stream), "Record prediction start event");
    
    // Launch kernel to count correct predictions
    prediction_blocks = (batch_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    count_correct_predictions_kernel<<<prediction_blocks, THREADS_PER_BLOCK, 0, cuda_stream>>>(
        d_batch_output, d_batch_targets, d_correct_count, 
        OUTPUT_NODES, batch_size
    );
    
    CUDA_CHECK(cudaEventRecord(kernel_stop, cuda_stream), "Record prediction stop event");
    CUDA_CHECK(cudaEventSynchronize(kernel_stop), "Synchronize prediction stop event");
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop), "Calculate prediction time");
    if (should_log) printf("[TIMING] Count predictions kernel: %.4f ms\n", kernel_time);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError(), "count_correct_predictions_kernel launch");
    
    // ────────── Backward Pass ──────────
    if (should_log && learning_rate > 0.0f) {
        printf("[FUSED] Performing backward pass...\n");
    }
    
    // Only perform backward pass if we're training (learning_rate > 0)
    if (learning_rate > 0.0f) {
        // Time output delta kernel
        CUDA_CHECK(cudaEventRecord(kernel_start, cuda_stream), "Record output delta start event");
        
        // Calculate output layer deltas
        calculate_output_delta_kernel<<<output_blocks, THREADS_PER_BLOCK, 2 * THREADS_PER_BLOCK * sizeof(float), cuda_stream>>>(
            d_batch_output, d_batch_targets, d_batch_output_delta, 
            OUTPUT_NODES, batch_size
        );
        
        CUDA_CHECK(cudaEventRecord(kernel_stop, cuda_stream), "Record output delta stop event");
        CUDA_CHECK(cudaEventSynchronize(kernel_stop), "Synchronize output delta stop event");
        CUDA_CHECK(cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop), "Calculate output delta time");
        if (should_log) printf("[TIMING] Output delta kernel: %.4f ms\n", kernel_time);
        
        // Check for kernel launch errors
        CUDA_CHECK(cudaGetLastError(), "calculate_output_delta_kernel launch");
        
        // Time hidden delta kernel
        CUDA_CHECK(cudaEventRecord(kernel_start, cuda_stream), "Record hidden delta start event");
        
        // Calculate hidden layer deltas
        calculate_hidden_delta_kernel<<<hidden_blocks, THREADS_PER_BLOCK, 
            (THREADS_PER_BLOCK + THREADS_PER_BLOCK * OUTPUT_NODES + OUTPUT_NODES) * sizeof(float), cuda_stream>>>(
            d_batch_hidden, d_batch_output_delta, d_weight2, d_batch_hidden_delta, 
            HIDDEN_NODES, OUTPUT_NODES, batch_size
        );
        
        CUDA_CHECK(cudaEventRecord(kernel_stop, cuda_stream), "Record hidden delta stop event");
        CUDA_CHECK(cudaEventSynchronize(kernel_stop), "Synchronize hidden delta stop event");
        CUDA_CHECK(cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop), "Calculate hidden delta time");
        if (should_log) printf("[TIMING] Hidden delta kernel: %.4f ms\n", kernel_time);
        
        // Check for kernel launch errors
        CUDA_CHECK(cudaGetLastError(), "calculate_hidden_delta_kernel launch");
        
        // ────────── Update Weights ──────────
        if (should_log) {
            printf("[FUSED] Updating weights...\n");
        }
        
        // Time input-hidden weight update kernel
        CUDA_CHECK(cudaEventRecord(kernel_start, cuda_stream), "Record input-hidden weights start event");
        
        // Update input-hidden weights (using vectorized4 version)
        input_hidden_blocks = (INPUT_NODES * HIDDEN_NODES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        input_hidden_blocks = input_hidden_blocks > MAX_BLOCKS ? MAX_BLOCKS : input_hidden_blocks;
        
        update_input_hidden_weights_kernel_vectorized4<<<input_hidden_blocks, THREADS_PER_BLOCK, 
            HIDDEN_NODES * sizeof(float), cuda_stream>>>(
            d_batch_input, d_batch_hidden_delta, d_weight1, d_bias1,
            INPUT_NODES, HIDDEN_NODES, batch_size, learning_rate
        );
        
        CUDA_CHECK(cudaEventRecord(kernel_stop, cuda_stream), "Record input-hidden weights stop event");
        CUDA_CHECK(cudaEventSynchronize(kernel_stop), "Synchronize input-hidden weights stop event");
        CUDA_CHECK(cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop), "Calculate input-hidden weights time");
        if (should_log) printf("[TIMING] Input-hidden weights kernel: %.4f ms\n", kernel_time);
        
        // Check for kernel launch errors
        CUDA_CHECK(cudaGetLastError(), "update_input_hidden_weights_kernel_vectorized4 launch");
        
        // Time hidden-output weight update kernel
        CUDA_CHECK(cudaEventRecord(kernel_start, cuda_stream), "Record hidden-output weights start event");
        
        // Update hidden-output weights
        hidden_output_blocks = (HIDDEN_NODES * OUTPUT_NODES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        hidden_output_blocks = hidden_output_blocks > MAX_BLOCKS ? MAX_BLOCKS : hidden_output_blocks;
        
        update_hidden_output_weights_kernel<<<hidden_output_blocks, THREADS_PER_BLOCK, 
            (THREADS_PER_BLOCK + OUTPUT_NODES) * sizeof(float), cuda_stream>>>(
            d_batch_hidden, d_batch_output_delta, d_weight2, d_bias2,
            HIDDEN_NODES, OUTPUT_NODES, batch_size, learning_rate
        );
        
        CUDA_CHECK(cudaEventRecord(kernel_stop, cuda_stream), "Record hidden-output weights stop event");
        CUDA_CHECK(cudaEventSynchronize(kernel_stop), "Synchronize hidden-output weights stop event");
        CUDA_CHECK(cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop), "Calculate hidden-output weights time");
        if (should_log) printf("[TIMING] Hidden-output weights kernel: %.4f ms\n", kernel_time);
        
        // Check for kernel launch errors
        CUDA_CHECK(cudaGetLastError(), "update_hidden_output_weights_kernel launch");
    }
    
    // Synchronize to ensure all operations are complete
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream), "Stream synchronize after all kernels");
    
    // Cleanup timing events
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    
    // Copy correct count back to host
    CUDA_CHECK(cudaMemcpy(&host_correct_count, d_correct_count, sizeof(int), cudaMemcpyDeviceToHost), 
               "cudaMemcpy for correct count");
    
    // Update the external correct_predictions counter
    *correct_predictions += host_correct_count;
    
    // ────────── Copy Updated Weights Back to Host ──────────
    if (should_log && learning_rate > 0.0f) {
        printf("[FUSED] Copying updated weights back to host...\n");
    }
    
    // Only copy updated weights and biases back if we're training
    if (learning_rate > 0.0f) {
        // Copy updated weights and biases back to host
        CUDA_CHECK(cudaMemcpy(weight1, d_weight1, weight1_size, cudaMemcpyDeviceToHost), "cudaMemcpy for weight1 back");
        CUDA_CHECK(cudaMemcpy(weight2, d_weight2, weight2_size, cudaMemcpyDeviceToHost), "cudaMemcpy for weight2 back");
        CUDA_CHECK(cudaMemcpy(bias1, d_bias1, bias1_size, cudaMemcpyDeviceToHost), "cudaMemcpy for bias1 back");
        CUDA_CHECK(cudaMemcpy(bias2, d_bias2, bias2_size, cudaMemcpyDeviceToHost), "cudaMemcpy for bias2 back");
    }
    
    // ────────── Log Performance and Increment Call Counter ──────────
    if (should_log) {
        end_time = clock();
        double elapsed = (double)(end_time - start_time) / CLOCKS_PER_SEC;
        total_time += elapsed;
        
        printf("[FUSED] Batch training completed in %.6f seconds\n", elapsed);
        printf("[FUSED] Average time per batch: %.6f seconds\n", total_time / (call_count + 1));
        printf("[FUSED] Accuracy for this batch: %.2f%% (%d/%d correct)\n", 
               100.0 * host_correct_count / batch_size, host_correct_count, batch_size);
    }
    
    call_count++;
    return;
    
Error:
    fprintf(stderr, "[FUSED] Error in train_batch_fused, call #%d\n", call_count);
    return;
} 