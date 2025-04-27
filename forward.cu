#include "forward.h"
#include <cuda_runtime.h>

// Logging control - set to 0 to disable all logging, 1 to enable
#define LOG_FORWARD 0

// Error checking macro for CUDA operations
#define CUDA_CHECK(operation, description) \
    cudaStatus = (operation); \
    if (cudaStatus != cudaSuccess) { \
        fprintf(stderr, "%s failed: %s\n", (description), cudaGetErrorString(cudaStatus)); \
        goto Error; \
    }

// Memory check macro
#define CHECK_PTR(ptr, msg) \
    if (!(ptr)) { \
        fprintf(stderr, "CUDA memory allocation failed: %s\n", (msg)); \
        return; \
    }

// CUDA optimization parameters
#define TILE_WIDTH 16        // Tile size for shared memory
#define BLOCK_SIZE 256       // Threads per block
#define MAX_BLOCKS 65535     // Maximum number of blocks

// CPU implementations of activation functions
extern "C" {
    double relu(double x) { 
        return x > 0 ? x : 0; 
    }

    double sigmoid(double x) { 
        return 1.0 / (1.0 + exp(-x)); 
    }
}

// Optimized CUDA kernel for calculating hidden layer activations with tiled matrix multiplication
__global__ void hidden_layer_kernel(
    const double* __restrict__ input,
    const double* __restrict__ weights,
    const double* __restrict__ bias,
    double* __restrict__ output,
    const int input_size,
    const int output_size
) {
    // Use shared memory for input tiles to reduce global memory accesses
    __shared__ double tile_input[TILE_WIDTH];
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_tid = threadIdx.x;
    
    // Each thread computes one output node
    if (tid < output_size) {
        double sum = bias[tid]; // Start with bias
        
        // Process input in tiles to improve memory access patterns
        for (int t = 0; t < input_size; t += TILE_WIDTH) {
            // Load a tile of input data into shared memory
            if (t + local_tid < input_size && local_tid < TILE_WIDTH) {
                tile_input[local_tid] = input[t + local_tid];
            }
            
            // Wait for all threads to load their data
            __syncthreads();
            
            // Process the tile
            const int tile_end = min(TILE_WIDTH, input_size - t);
            for (int j = 0; j < tile_end; j++) {
                sum += tile_input[j] * weights[(t + j) * output_size + tid];
            }
            
            // Wait for all threads to finish using the tile before loading a new one
            __syncthreads();
        }
        
        // Apply ReLU activation
        output[tid] = sum > 0 ? sum : 0;
    }
}

// CUDA kernel for calculating output layer activations with Sigmoid
__global__ void output_layer_kernel(
    const double* __restrict__ input,
    const double* __restrict__ weights,
    const double* __restrict__ bias,
    double* __restrict__ output,
    const int input_size,
    const int output_size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < output_size) {
        double sum = bias[tid];
        for (int j = 0; j < input_size; j++) {
            sum += input[j] * weights[j * output_size + tid];
        }
        // Apply Sigmoid activation
        output[tid] = 1.0 / (1.0 + exp(-sum));
    }
}

extern "C"
void forward_propagate(
    const double* input,
    const double* weight1,
    const double* weight2,
    const double* bias1,
    const double* bias2,
    double* hidden,
    double* output,
    int num_threads
) {
    // For logging performance
    static int call_count = 0;
    static double total_time = 0.0;
    static bool first_call = true;
    
    // Only log occasionally to avoid flooding output
    bool should_log = LOG_FORWARD && (call_count % 100 == 0);
    
    if (LOG_FORWARD && first_call) {
        // Show GPU info on first run
        cudaDeviceProp deviceProp;
        cudaGetDeviceCount(NULL);
        cudaGetDeviceProperties(&deviceProp, 0);
        fprintf(stderr, "\n[CUDA] Using GPU: %s\n", deviceProp.name);
        fprintf(stderr, "[CUDA] Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        fprintf(stderr, "[CUDA] Total global memory: %.2f GB\n", deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        fprintf(stderr, "[CUDA] Multiprocessors: %d\n", deviceProp.multiProcessorCount);
        first_call = false;
    }
    
    // Start timing if logging this call
    clock_t start_time = 0;
    if (should_log) {
        start_time = clock();
        fprintf(stderr, "[CUDA] Forward pass #%d with input size %d\n", call_count, INPUT_NODES);
    }
    
    // Declare all variables at the top of the function
    cudaError_t cudaStatus;
    
    // Device memory pointers - initialize to NULL
    double *d_input = NULL, *d_weight1 = NULL, *d_weight2 = NULL;
    double *d_bias1 = NULL, *d_bias2 = NULL, *d_hidden = NULL, *d_output = NULL;
    
    // Variables for kernel launch
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = 0;
    
    // Track if we registered host memory
    bool hostRegistered = false;
    
    if (should_log) {
        fprintf(stderr, "[CUDA] Allocating GPU memory...\n");
    }
    
    // Use pinned memory for faster transfers
    cudaStatus = cudaHostRegister((void*)input, INPUT_NODES * sizeof(double), cudaHostRegisterDefault);
    if (cudaStatus == cudaSuccess) {
        hostRegistered = true;
    } else {
        // Non-fatal error, continue with unpinned memory
        fprintf(stderr, "Warning: Could not register host memory: %s\n", cudaGetErrorString(cudaStatus));
    }
    
    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc((void**)&d_input, INPUT_NODES * sizeof(double)), "cudaMalloc for d_input");
    CUDA_CHECK(cudaMalloc((void**)&d_weight1, INPUT_NODES * HIDDEN_NODES * sizeof(double)), "cudaMalloc for d_weight1");
    CUDA_CHECK(cudaMalloc((void**)&d_bias1, HIDDEN_NODES * sizeof(double)), "cudaMalloc for d_bias1");
    CUDA_CHECK(cudaMalloc((void**)&d_hidden, HIDDEN_NODES * sizeof(double)), "cudaMalloc for d_hidden");
    CUDA_CHECK(cudaMalloc((void**)&d_weight2, HIDDEN_NODES * OUTPUT_NODES * sizeof(double)), "cudaMalloc for d_weight2");
    CUDA_CHECK(cudaMalloc((void**)&d_bias2, OUTPUT_NODES * sizeof(double)), "cudaMalloc for d_bias2");
    CUDA_CHECK(cudaMalloc((void**)&d_output, OUTPUT_NODES * sizeof(double)), "cudaMalloc for d_output");
    
    if (should_log) {
        fprintf(stderr, "[CUDA] Copying data to GPU...\n");
    }
    
    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_input, input, INPUT_NODES * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy for d_input");
    CUDA_CHECK(cudaMemcpy(d_weight1, weight1, INPUT_NODES * HIDDEN_NODES * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy for d_weight1");
    CUDA_CHECK(cudaMemcpy(d_bias1, bias1, HIDDEN_NODES * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy for d_bias1");
    CUDA_CHECK(cudaMemcpy(d_weight2, weight2, HIDDEN_NODES * OUTPUT_NODES * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy for d_weight2");
    CUDA_CHECK(cudaMemcpy(d_bias2, bias2, OUTPUT_NODES * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy for d_bias2");
    
    if (should_log) {
        fprintf(stderr, "[CUDA] Running hidden layer kernel...\n");
    }
    
    // Launch kernel for hidden layer computation
    // Calculate appropriate grid size (capped at MAX_BLOCKS)
    blocksPerGrid = min((HIDDEN_NODES + threadsPerBlock - 1) / threadsPerBlock, MAX_BLOCKS);
    
    hidden_layer_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_input, d_weight1, d_bias1, d_hidden, INPUT_NODES, HIDDEN_NODES
    );
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError(), "hidden_layer_kernel launch");
    CUDA_CHECK(cudaDeviceSynchronize(), "cudaDeviceSynchronize after hidden layer");
    
    if (should_log) {
        fprintf(stderr, "[CUDA] Running output layer kernel...\n");
    }
    
    // Launch kernel for output layer computation
    blocksPerGrid = min((OUTPUT_NODES + threadsPerBlock - 1) / threadsPerBlock, MAX_BLOCKS);
    
    output_layer_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_hidden, d_weight2, d_bias2, d_output, HIDDEN_NODES, OUTPUT_NODES
    );
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError(), "output_layer_kernel launch");
    CUDA_CHECK(cudaDeviceSynchronize(), "cudaDeviceSynchronize after output layer");
    
    if (should_log) {
        fprintf(stderr, "[CUDA] Copying results back to CPU...\n");
    }
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(hidden, d_hidden, HIDDEN_NODES * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy for hidden");
    CUDA_CHECK(cudaMemcpy(output, d_output, OUTPUT_NODES * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy for output");
    
    // Log performance if needed
    if (should_log) {
        clock_t end_time = clock();
        double elapsed = (double)(end_time - start_time) / CLOCKS_PER_SEC;
        total_time += elapsed;
        
        fprintf(stderr, "[CUDA] Forward pass completed in %.6f seconds\n", elapsed);
        fprintf(stderr, "[CUDA] Average time per pass: %.6f seconds\n", total_time / (call_count + 1));
    }
    
    call_count++;
    
    // Unregister host memory if we registered it
    if (hostRegistered) {
        cudaHostUnregister((void*)input);
    }
    goto Cleanup;  // Skip error message if successful

Error:
    if (LOG_FORWARD) {
        fprintf(stderr, "[CUDA] Error in forward propagation, call #%d\n", call_count);
    }

Cleanup:
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight1);
    cudaFree(d_bias1);
    cudaFree(d_hidden);
    cudaFree(d_weight2);
    cudaFree(d_bias2);
    cudaFree(d_output);
} 