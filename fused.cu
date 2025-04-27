#include "fused.h"
#include <cuda_runtime.h>
#include <cuda.h>  // Added for CUdeviceptr
#include <stdio.h>

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

// Modified version for graph creation that doesn't use goto
#define CUDA_CHECK_GRAPH(operation, description, cleanup_action) \
    do { \
        cudaError_t localStatus = (operation); \
        if (localStatus != cudaSuccess) { \
            fprintf(stderr, "%s failed: %s\n", (description), cudaGetErrorString(localStatus)); \
            cleanup_action; \
            return localStatus; \
        } \
    } while(0)

// CUDA thread/block parameters
#define THREADS_PER_BLOCK 256
#define MAX_BLOCKS 65535

// Uncomment to enable CUDA graph optimization
#define USE_CUDA_GRAPH 0

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

// ────────── Activation functions ──────────

// Device functions for activation and derivatives
__device__ inline float relu(float x) {
    return fmaxf(0.0f, x);
}

__device__ inline float relu_derivative(float x) {
    return (x > 0.0f) ? 1.0f : 0.0f;
}

__device__ inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ inline float sigmoid_derivative(float y) {
    return y * (1.0f - y);
}

// ────────── Forward pass kernels ──────────

// Optimized kernel for hidden layer forward pass (with ReLU activation)
__global__ void hidden_layer_kernel_batch(
    const float* __restrict__ batch_input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ batch_hidden,
    int input_size,
    int hidden_size,
    int batch_size
) {
   // Shared memory for bias
    extern __shared__ float shared_mem[];
    float* s_bias = shared_mem;
    
    // Load bias into shared memory (only need first hidden_size elements)
    int tid = threadIdx.x;
    if (tid < hidden_size) {
        s_bias[tid] = bias[tid];
    }
    __syncthreads();
    
    // Calculate global thread ID
    int global_tid = blockIdx.x * blockDim.x + tid;
    
    // Each thread calculates one hidden node activation for one image
    if (global_tid < hidden_size * batch_size) {
        int batch_idx = global_tid / hidden_size;     // Which image in the batch
        int hidden_idx = global_tid % hidden_size;    // Which hidden neuron
        
        // Base pointers for this image/neuron
        const float* input = batch_input + batch_idx * input_size;
        float* hidden = batch_hidden + batch_idx * hidden_size;
        
        // Calculate the sum for this hidden neuron
        float sum = s_bias[hidden_idx];  // Use shared memory for bias
        
        // Calculate dot product using global memory for weights with loop unrolling
        int j = 0;
        
        // Main loop with compiler-directed unrolling
        #pragma unroll 4
        for (j = 0; j < (input_size / 4) * 4; j += 4) {
            sum += input[j] * weights[j * hidden_size + hidden_idx];
            sum += input[j+1] * weights[(j+1) * hidden_size + hidden_idx];
            sum += input[j+2] * weights[(j+2) * hidden_size + hidden_idx];
            sum += input[j+3] * weights[(j+3) * hidden_size + hidden_idx];
        }
        
        // Handle remaining elements
        for (; j < input_size; j++) {
            sum += input[j] * weights[j * hidden_size + hidden_idx];
        }
        
        // Apply ReLU activation and store result
        hidden[hidden_idx] = relu(sum);
    }
}

// Vectorized kernel using float4 for more efficient memory access
__global__ void hidden_layer_kernel_batch_vectorized4(
    const float* __restrict__ batch_input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ batch_hidden,
    int input_size,
    int hidden_size,
    int batch_size
) {
    // Shared memory for bias
    extern __shared__ float shared_mem[];
    float* s_bias = shared_mem;
    
    // Load bias into shared memory
    int tid = threadIdx.x;
    if (tid < hidden_size) {
        s_bias[tid] = bias[tid];
    }
    __syncthreads();
    
    // Calculate global thread ID
    int global_tid = blockIdx.x * blockDim.x + tid;
    
    // Each thread calculates one hidden node activation for one image
    if (global_tid < hidden_size * batch_size) {
        int batch_idx = global_tid / hidden_size;     // Which image in the batch
        int hidden_idx = global_tid % hidden_size;    // Which hidden neuron
        
        // Base pointers for this image/neuron
        const float* input = batch_input + batch_idx * input_size;
        float* hidden = batch_hidden + batch_idx * hidden_size;
        
        // Cache bias in register
        float sum = s_bias[hidden_idx];
        
        // Cache weight offset in register for better addressing
        int weight_offset = hidden_idx;
        
        // Process four elements at a time using float4 vectorization
        int j = 0;
        int vectorized_limit = (input_size / 4) * 4;  // Round down to multiple of 4
        
        for (; j < vectorized_limit; j += 4) {
            // Load four inputs at once
            float4 input_quad;
            input_quad.x = input[j];
            input_quad.y = input[j+1];
            input_quad.z = input[j+2];
            input_quad.w = input[j+3];
            
            // Load four weights at once
            float4 weight_quad;
            weight_quad.x = weights[j * hidden_size + weight_offset];
            weight_quad.y = weights[(j+1) * hidden_size + weight_offset];
            weight_quad.z = weights[(j+2) * hidden_size + weight_offset];
            weight_quad.w = weights[(j+3) * hidden_size + weight_offset];
            
            // Compute products and accumulate
            sum += input_quad.x * weight_quad.x;
            sum += input_quad.y * weight_quad.y;
            sum += input_quad.z * weight_quad.z;
            sum += input_quad.w * weight_quad.w;
        }
        
        // Handle remaining elements
        for (; j < input_size; j++) {
            sum += input[j] * weights[j * hidden_size + weight_offset];
        }
        
        // Apply ReLU activation and store result
        hidden[hidden_idx] = relu(sum);
    }
}

// Kernel for output layer forward pass (with Sigmoid activation)
__global__ void output_layer_kernel_batch(
    const float* __restrict__ batch_hidden,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ batch_output,
    int hidden_size,
    int output_size,
    int batch_size
) {
    // Shared memory for bias
    extern __shared__ float s_bias[];
    
    // Load bias into shared memory
    int tid = threadIdx.x;
    if (tid < output_size) {
        s_bias[tid] = bias[tid];
    }
    __syncthreads();
    
    // Calculate global thread ID
    int global_tid = blockIdx.x * blockDim.x + tid;
    
    // Each thread calculates one output node activation for one image
    if (global_tid < output_size * batch_size) {
        int batch_idx = global_tid / output_size;     // Which image in the batch
        int output_idx = global_tid % output_size;    // Which output neuron
        
        // Base pointers for this image/neuron
        const float* hidden = batch_hidden + batch_idx * hidden_size;
        float* output = batch_output + batch_idx * output_size;
        
        // Calculate the sum for this output neuron
        float sum = s_bias[output_idx];  // Use shared memory for bias
        
        // Calculate dot product with loop unrolling
        int j = 0;
        
        // Main loop with compiler-directed unrolling
        #pragma unroll 4
        for (j = 0; j < (hidden_size / 4) * 4; j += 4) {
            sum += hidden[j] * weights[j * output_size + output_idx];
            sum += hidden[j+1] * weights[(j+1) * output_size + output_idx];
            sum += hidden[j+2] * weights[(j+2) * output_size + output_idx];
            sum += hidden[j+3] * weights[(j+3) * output_size + output_idx];
        }
        
        // Handle remaining elements
        for (; j < hidden_size; j++) {
            sum += hidden[j] * weights[j * output_size + output_idx];
        }
        
        // Apply Sigmoid activation and store result
        output[output_idx] = sigmoid(sum);
    }
}

// Advanced optimized kernel for output layer forward pass (with Sigmoid activation)
__global__ void output_layer_kernel_batch_advanced(
    const float* __restrict__ batch_hidden,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ batch_output,
    int hidden_size,
    int output_size,
    int batch_size
) {
    // Shared memory for bias
    extern __shared__ float s_bias[];
    
    // Load bias into shared memory
    int tid = threadIdx.x;
    if (tid < output_size) {
        s_bias[tid] = bias[tid];
    }
    __syncthreads();
    
    // Calculate global thread ID
    int global_tid = blockIdx.x * blockDim.x + tid;
    
    // Each thread calculates one output node activation for one image
    if (global_tid < output_size * batch_size) {
        int batch_idx = global_tid / output_size;     // Which image in the batch
        int output_idx = global_tid % output_size;    // Which output neuron
        
        // Base pointers for this image/neuron
        const float* hidden = batch_hidden + batch_idx * hidden_size;
        float* output = batch_output + batch_idx * output_size;
        
        // Cache bias in register
        float sum = s_bias[output_idx];
        
        // Cache weight offset in register for better addressing
        int weight_offset = output_idx;
        
        // Main processing loop with compiler-directed unrolling
        int j = 0;
        #pragma unroll 8
        for (; j < (hidden_size / 8) * 8; j += 8) {
            sum += hidden[j] * weights[j * output_size + weight_offset];
            sum += hidden[j+1] * weights[(j+1) * output_size + weight_offset];
            sum += hidden[j+2] * weights[(j+2) * output_size + weight_offset];
            sum += hidden[j+3] * weights[(j+3) * output_size + weight_offset];
            sum += hidden[j+4] * weights[(j+4) * output_size + weight_offset];
            sum += hidden[j+5] * weights[(j+5) * output_size + weight_offset];
            sum += hidden[j+6] * weights[(j+6) * output_size + weight_offset];
            sum += hidden[j+7] * weights[(j+7) * output_size + weight_offset];
        }
        
        // Handle remaining elements with 4-way unrolling
        #pragma unroll 4
        for (; j < (hidden_size / 4) * 4; j += 4) {
            sum += hidden[j] * weights[j * output_size + weight_offset];
            sum += hidden[j+1] * weights[(j+1) * output_size + weight_offset];
            sum += hidden[j+2] * weights[(j+2) * output_size + weight_offset];
            sum += hidden[j+3] * weights[(j+3) * output_size + weight_offset];
        }
        
        // Handle final elements
        for (; j < hidden_size; j++) {
            sum += hidden[j] * weights[j * output_size + weight_offset];
        }
        
        // Apply Sigmoid activation and store result
        output[output_idx] = sigmoid(sum);
    }
}

// ────────── Backward pass kernels ──────────

// Kernel to calculate output layer deltas
__global__ void calculate_output_delta_kernel(
    const float* __restrict__ batch_output,
    const float* __restrict__ batch_targets,
    float* __restrict__ batch_output_delta,
    int output_size,
    int batch_size
) {
    extern __shared__ float shared_mem[];
    float* s_output = shared_mem;
    float* s_targets = shared_mem + blockDim.x;
    
    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + tid;
    
    // Determine batch and output indices
    if (global_tid < output_size * batch_size) {
        int batch_idx = global_tid / output_size;
        int output_idx = global_tid % output_size;
        
        // Load values into shared memory
        int offset = batch_idx * output_size + output_idx;
        s_output[tid] = batch_output[offset];
        s_targets[tid] = batch_targets[offset];
        
        __syncthreads();
        
        // Compute error and derivative
        float output_val = s_output[tid];
        float target_val = s_targets[tid];
        
        batch_output_delta[offset] = (output_val - target_val) * sigmoid_derivative(output_val);
    }
}

// Kernel to calculate hidden layer deltas
__global__ void calculate_hidden_delta_kernel(
    const float* __restrict__ batch_hidden,
    const float* __restrict__ batch_output_delta,
    const float* __restrict__ weights,
    float* __restrict__ batch_hidden_delta,
    int hidden_size,
    int output_size,
    int batch_size
) {
    extern __shared__ float shared_mem[];
    // Allocate shared memory (weights will be loaded in chunks if needed)
    float* s_hidden = shared_mem;
    float* s_weights_chunk = shared_mem + blockDim.x;
    float* s_output_delta = s_weights_chunk + blockDim.x * output_size;
    
    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + tid;
    
    if (global_tid < hidden_size * batch_size) {
        int batch_idx = global_tid / hidden_size;
        int hidden_idx = global_tid % hidden_size;
        
        // Load hidden value to shared memory
        s_hidden[tid] = batch_hidden[batch_idx * hidden_size + hidden_idx];
        
        // Base pointers for this batch
        const float* output_delta = batch_output_delta + batch_idx * output_size;
        
        // Load output deltas for this batch to shared memory
        if (tid < output_size) {
            s_output_delta[tid] = output_delta[tid];
        }
        
        // Ensure hidden values and output deltas are loaded
        __syncthreads();
        
        // Calculate weighted sum of output deltas
        float sum = 0.0f;
        
        // Process weights in chunks to avoid shared memory limitations
        for (int j = 0; j < output_size; j++) {
            // Load one weight per thread
            if (tid < blockDim.x) {
                s_weights_chunk[tid] = weights[hidden_idx * output_size + j];
            }
            __syncthreads();
            
            sum += s_output_delta[j] * s_weights_chunk[tid % blockDim.x];
        }
        
        // Multiply by derivative of ReLU
        batch_hidden_delta[batch_idx * hidden_size + hidden_idx] = sum * relu_derivative(s_hidden[tid]);
    }
}

// ────────── Weight update kernels ──────────

// Kernel to update weights and biases for hidden layer
__global__ void update_input_hidden_weights_kernel(
    const float* __restrict__ batch_input,
    const float* __restrict__ batch_hidden_delta,
    float* __restrict__ weights,
    float* __restrict__ bias,
    int input_size,
    int hidden_size,
    int batch_size,
    float learning_rate
) {
    extern __shared__ float shared_mem[];
    float* s_gradient_accumulator = shared_mem;
    float* s_bias_gradient = shared_mem + blockDim.x;
    
    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + tid;
    
    // Initialize gradient accumulator in shared memory
    s_gradient_accumulator[tid] = 0.0f;
    
    // Threads handling hidden neuron biases also initialize bias gradient
    if (tid < hidden_size) {
        s_bias_gradient[tid] = 0.0f;
    }
    
    __syncthreads();
    
    if (global_tid < input_size * hidden_size) {
        int input_idx = global_tid / hidden_size;
        int hidden_idx = global_tid % hidden_size;
        
        // Accumulate gradients across batch directly into shared memory
        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            float input_val = batch_input[batch_idx * input_size + input_idx];
            float hidden_delta = batch_hidden_delta[batch_idx * hidden_size + hidden_idx];
            s_gradient_accumulator[tid] += input_val * hidden_delta;
            
            // For threads handling hidden neurons, also accumulate bias gradients
            if (input_idx == 0 && hidden_idx < hidden_size) {
                s_bias_gradient[hidden_idx] += hidden_delta;
            }
        }
        
        __syncthreads();
        
        // Apply gradient using learning rate
        weights[input_idx * hidden_size + hidden_idx] -= learning_rate * s_gradient_accumulator[tid] / batch_size;
    }
    
    // Update bias for hidden layer (one thread per hidden neuron)
    if (tid < hidden_size && global_tid < input_size * hidden_size) {
        // Apply bias gradient
        bias[tid] -= learning_rate * s_bias_gradient[tid] / batch_size;
    }
}

// Vectorized kernel using float4 for input-hidden weight updates
__global__ void update_input_hidden_weights_kernel_vectorized4(
    const float* __restrict__ batch_input,
    const float* __restrict__ batch_hidden_delta,
    float* __restrict__ weights,
    float* __restrict__ bias,
    int input_size,
    int hidden_size,
    int batch_size,
    float learning_rate
) {
    extern __shared__ float shared_mem[];
    float* s_bias_gradient = shared_mem;
    
    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + tid;
    
    // Initialize gradient accumulator in register for better performance
    float gradient_sum = 0.0f;
    
    // Initialize bias gradients in shared memory
    if (tid < hidden_size) {
        s_bias_gradient[tid] = 0.0f;
    }
    __syncthreads();
    
    if (global_tid < input_size * hidden_size) {
        int input_idx = global_tid / hidden_size;
        int hidden_idx = global_tid % hidden_size;
        
        // Cache indices for better addressing
        int input_offset = input_idx;
        int hidden_offset = hidden_idx;
        
        // Process batches in blocks of 4 for vectorization
        int vectorized_limit = (batch_size / 4) * 4; // Round down to multiple of 4
        
        for (int batch_idx = 0; batch_idx < vectorized_limit; batch_idx += 4) {
            // Load 4 inputs at once
            float4 input_quad;
            input_quad.x = batch_input[(batch_idx+0) * input_size + input_offset];
            input_quad.y = batch_input[(batch_idx+1) * input_size + input_offset];
            input_quad.z = batch_input[(batch_idx+2) * input_size + input_offset];
            input_quad.w = batch_input[(batch_idx+3) * input_size + input_offset];
            
            // Load 4 hidden deltas at once
            float4 delta_quad;
            delta_quad.x = batch_hidden_delta[(batch_idx+0) * hidden_size + hidden_offset];
            delta_quad.y = batch_hidden_delta[(batch_idx+1) * hidden_size + hidden_offset];
            delta_quad.z = batch_hidden_delta[(batch_idx+2) * hidden_size + hidden_offset];
            delta_quad.w = batch_hidden_delta[(batch_idx+3) * hidden_size + hidden_offset];
            
            // Compute gradient contributions and accumulate
            gradient_sum += input_quad.x * delta_quad.x;
            gradient_sum += input_quad.y * delta_quad.y;
            gradient_sum += input_quad.z * delta_quad.z;
            gradient_sum += input_quad.w * delta_quad.w;
            
            // Accumulate bias gradients if this thread handles a bias
            if (input_idx == 0) {
                atomicAdd(&s_bias_gradient[hidden_idx], delta_quad.x + delta_quad.y + delta_quad.z + delta_quad.w);
            }
        }
        
        // Handle the remaining items (less than 4)
        for (int batch_idx = vectorized_limit; batch_idx < batch_size; batch_idx++) {
            float input_val = batch_input[batch_idx * input_size + input_offset];
            float hidden_delta = batch_hidden_delta[batch_idx * hidden_size + hidden_offset];
            
            gradient_sum += input_val * hidden_delta;
            
            if (input_idx == 0) {
                atomicAdd(&s_bias_gradient[hidden_idx], hidden_delta);
            }
        }
        
        __syncthreads();
        
        // Update weight with a single write operation
        weights[input_idx * hidden_size + hidden_idx] -= learning_rate * gradient_sum / batch_size;
        
        // Update bias if this thread is responsible
        if (input_idx == 0 && hidden_idx < hidden_size) {
            bias[hidden_idx] -= learning_rate * s_bias_gradient[hidden_idx] / batch_size;
        }
    }
}

// Kernel to update weights and biases for output layer
__global__ void update_hidden_output_weights_kernel(
    const float* __restrict__ batch_hidden,
    const float* __restrict__ batch_output_delta,
    float* __restrict__ weights,
    float* __restrict__ bias,
    int hidden_size,
    int output_size,
    int batch_size,
    float learning_rate
) {
    extern __shared__ float shared_mem[];
    float* s_gradient_accumulator = shared_mem;
    float* s_bias_gradient = shared_mem + blockDim.x;
    
    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + tid;
    
    // Initialize gradient accumulators in shared memory
    s_gradient_accumulator[tid] = 0.0f;
    
    // Threads handling output neuron biases also initialize bias gradient
    if (tid < output_size) {
        s_bias_gradient[tid] = 0.0f;
    }
    
    __syncthreads();
    
    if (global_tid < hidden_size * output_size) {
        int hidden_idx = global_tid / output_size;
        int output_idx = global_tid % output_size;
        
        // Accumulate gradients across batch into shared memory
        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            float hidden_val = batch_hidden[batch_idx * hidden_size + hidden_idx];
            float output_delta = batch_output_delta[batch_idx * output_size + output_idx];
            s_gradient_accumulator[tid] += hidden_val * output_delta;
            
            // For threads handling output neurons, also accumulate bias gradients
            if (hidden_idx == 0 && output_idx < output_size) {
                s_bias_gradient[output_idx] += output_delta;
            }
        }
        
        __syncthreads();
        
        // Apply gradient using learning rate
        weights[hidden_idx * output_size + output_idx] -= learning_rate * s_gradient_accumulator[tid] / batch_size;
    }
    
    // Update bias for output layer (one thread per output neuron)
    if (tid < output_size && global_tid < hidden_size * output_size) {
        // Apply bias gradient
        bias[tid] -= learning_rate * s_bias_gradient[tid] / batch_size;
    }
}

// Kernel to count correct predictions in the batch
__global__ void count_correct_predictions_kernel(
    const float* __restrict__ batch_output,
    const float* __restrict__ batch_targets,
    int* __restrict__ correct_count,
    int output_size,
    int batch_size
) {
    // Shared memory for local counts (one per thread block)
    __shared__ int block_correct_count;
    
    // Initialize shared memory
    if (threadIdx.x == 0) {
        block_correct_count = 0;
    }
    __syncthreads();
    
    // Process one image per thread
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx < batch_size) {
        // Find the predicted class (max output value)
        int predicted_class = 0;
        float max_output = batch_output[batch_idx * output_size];
        
        for (int j = 1; j < output_size; j++) {
            float output_val = batch_output[batch_idx * output_size + j];
            if (output_val > max_output) {
                max_output = output_val;
                predicted_class = j;
            }
        }
        
        // Find the target class (index of 1.0)
        int target_class = 0;
        for (int j = 0; j < output_size; j++) {
            if (batch_targets[batch_idx * output_size + j] > 0.5f) {
                target_class = j;
                break;
            }
        }
        
        // Increment the count if prediction matches target
        if (predicted_class == target_class) {
            atomicAdd(&block_correct_count, 1);
        }
    }
    
    // Wait for all threads to finish
    __syncthreads();
    
    // Add block count to global count
    if (threadIdx.x == 0) {
        atomicAdd(correct_count, block_correct_count);
    }
}

// ────────── CUDA Graph structures and setup ──────────
#ifdef USE_CUDA_GRAPH
typedef struct {
    // Graph objects
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    
    // Whether graph has been created
    bool initialized;
    
    // Batch size used for graph creation
    int captured_batch_size;
    float captured_learning_rate;
    
    // Parameters needed for operation
    int input_size;
    int hidden_size;
    int output_size;
} GraphData;

// Global graph data
static GraphData graph_data = {0};

// Function to clean up graph resources
void cleanup_graph_resources() {
    if (graph_data.initialized) {
        if (graph_data.instance) {
            cudaGraphExecDestroy(graph_data.instance);
            graph_data.instance = NULL;
        }
        if (graph_data.graph) {
            cudaGraphDestroy(graph_data.graph);
            graph_data.graph = NULL;
        }
        graph_data.initialized = false;
    }
}

// Function to clean up CUDA resources
void cleanup_cuda_resources(cudaGraph_t graph, cudaGraphExec_t instance) {
    if (instance) {
        cudaGraphExecDestroy(instance);
    }
    if (graph) {
        cudaGraphDestroy(graph);
    }
}

// Function to create and initialize a CUDA graph for training
cudaError_t create_training_graph(
    int batch_size,
    float learning_rate
) {
    // Initialize all variables at the beginning
    cudaGraph_t tempGraph = NULL;
    cudaGraphExec_t tempInstance = NULL;
    cudaGraphNode_t errorNode = NULL;

    // Calculate grid dimensions for kernels
    int total_hidden_neurons = batch_size * HIDDEN_NODES;
    int hidden_blocks = (total_hidden_neurons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    hidden_blocks = hidden_blocks > MAX_BLOCKS ? MAX_BLOCKS : hidden_blocks;
    
    int total_output_neurons = batch_size * OUTPUT_NODES;
    int output_blocks = (total_output_neurons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    output_blocks = output_blocks > MAX_BLOCKS ? MAX_BLOCKS : output_blocks;
    
    int prediction_blocks = (batch_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    int input_hidden_blocks = (INPUT_NODES * HIDDEN_NODES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    input_hidden_blocks = input_hidden_blocks > MAX_BLOCKS ? MAX_BLOCKS : input_hidden_blocks;
    
    int hidden_output_blocks = (HIDDEN_NODES * OUTPUT_NODES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    hidden_output_blocks = hidden_output_blocks > MAX_BLOCKS ? MAX_BLOCKS : hidden_output_blocks;

    // Use non-goto error checking for graph creation
    cudaError_t status;
    
    // Begin capturing
    status = cudaStreamBeginCapture(cuda_stream, cudaStreamCaptureModeGlobal);
    if (status != cudaSuccess) {
        fprintf(stderr, "Begin stream capture failed: %s\n", cudaGetErrorString(status));
        return status;
    }
    
    // Reset correct count
    status = cudaMemsetAsync(d_correct_count, 0, sizeof(int), cuda_stream);
    if (status != cudaSuccess) {
        fprintf(stderr, "Reset correct count failed: %s\n", cudaGetErrorString(status));
        status = cudaStreamEndCapture(cuda_stream, &tempGraph);
        return cudaErrorUnknown;
    }
    
    // Forward pass - hidden layer (using vectorized4 version)
    hidden_layer_kernel_batch_vectorized4<<<hidden_blocks, THREADS_PER_BLOCK, HIDDEN_NODES * sizeof(float), cuda_stream>>>(
        d_batch_input, d_weight1, d_bias1, d_batch_hidden, 
        INPUT_NODES, HIDDEN_NODES, batch_size
    );
    
    // Forward pass - output layer
    output_layer_kernel_batch<<<output_blocks, THREADS_PER_BLOCK, OUTPUT_NODES * sizeof(float), cuda_stream>>>(
        d_batch_hidden, d_weight2, d_bias2, d_batch_output, 
        HIDDEN_NODES, OUTPUT_NODES, batch_size
    );
    
    // Count correct predictions
    count_correct_predictions_kernel<<<prediction_blocks, THREADS_PER_BLOCK, 0, cuda_stream>>>(
        d_batch_output, d_batch_targets, d_correct_count, 
        OUTPUT_NODES, batch_size
    );
    
    // Backward pass - output layer deltas
    calculate_output_delta_kernel<<<output_blocks, THREADS_PER_BLOCK, 2 * THREADS_PER_BLOCK * sizeof(float), cuda_stream>>>(
        d_batch_output, d_batch_targets, d_batch_output_delta, 
        OUTPUT_NODES, batch_size
    );
    
    // Backward pass - hidden layer deltas
    calculate_hidden_delta_kernel<<<hidden_blocks, THREADS_PER_BLOCK, 
        (THREADS_PER_BLOCK + THREADS_PER_BLOCK * OUTPUT_NODES + OUTPUT_NODES) * sizeof(float), cuda_stream>>>(
        d_batch_hidden, d_batch_output_delta, d_weight2, d_batch_hidden_delta, 
        HIDDEN_NODES, OUTPUT_NODES, batch_size
    );
    
    // Weight updates - input to hidden (using vectorized4 version)
    update_input_hidden_weights_kernel_vectorized4<<<input_hidden_blocks, THREADS_PER_BLOCK, 
        HIDDEN_NODES * sizeof(float), cuda_stream>>>(
        d_batch_input, d_batch_hidden_delta, d_weight1, d_bias1,
        INPUT_NODES, HIDDEN_NODES, batch_size, learning_rate
    );
    
    // Weight updates - hidden to output
    update_hidden_output_weights_kernel<<<hidden_output_blocks, THREADS_PER_BLOCK, 
        (THREADS_PER_BLOCK + OUTPUT_NODES) * sizeof(float), cuda_stream>>>(
        d_batch_hidden, d_batch_output_delta, d_weight2, d_bias2,
        HIDDEN_NODES, OUTPUT_NODES, batch_size, learning_rate
    );
    
    // End capturing
    status = cudaStreamEndCapture(cuda_stream, &tempGraph);
    if (status != cudaSuccess) {
        fprintf(stderr, "End stream capture failed: %s\n", cudaGetErrorString(status));
        return status;
    }
    
    // Create executable graph
    status = cudaGraphInstantiate(&tempInstance, tempGraph, &errorNode, NULL, 0);
    if (status != cudaSuccess) {
        fprintf(stderr, "Instantiate graph failed: %s\n", cudaGetErrorString(status));
        cleanup_cuda_resources(tempGraph, NULL);
        return status;
    }
    
    // Cleanup any existing graph
    cleanup_graph_resources();
    
    // Store the new graph
    graph_data.graph = tempGraph;
    graph_data.instance = tempInstance;
    graph_data.captured_batch_size = batch_size;
    graph_data.captured_learning_rate = learning_rate;
    graph_data.input_size = INPUT_NODES;
    graph_data.hidden_size = HIDDEN_NODES;
    graph_data.output_size = OUTPUT_NODES;
    
    // Mark graph as initialized
    graph_data.initialized = true;
    
    return cudaSuccess;
}
#endif

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
    // CUDA_CHECK(cudaStreamCreate(&cuda_stream), "Create CUDA stream");
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
    // Clean up graph
    #ifdef USE_CUDA_GRAPH
    cleanup_graph_resources();
    #endif
    
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
    
    // Flag to indicate if we're using CUDA graph
    bool using_graph = false;
    
    // Ensure environment is initialized
    if (!is_initialized || batch_size > current_max_batch_size) {
        if (should_log) {
            printf("[FUSED] Initializing training environment with batch size %d\n", batch_size);
        }
        // CUDA_CHECK(init_fused_training(batch_size), "Initialize fused training");
        init_fused_training(batch_size);
    }
    
    // Reset correct count
    // CUDA_CHECK(cudaMemset(d_correct_count, 0, sizeof(int)), "Reset correct count");
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
    
#ifdef USE_CUDA_GRAPH
    // Check if we can use CUDA graph
    if (graph_data.initialized && 
        batch_size == graph_data.captured_batch_size && 
        learning_rate == graph_data.captured_learning_rate) {
        
        // We can reuse the existing graph
        if (should_log) {
            printf("[FUSED] Executing existing CUDA graph...\n");
        }
        
        CUDA_CHECK(cudaGraphLaunch(graph_data.instance, cuda_stream), "Launch graph");
        CUDA_CHECK(cudaStreamSynchronize(cuda_stream), "Stream synchronize after graph launch");
        
        using_graph = true;
    } else {
        // Need to create a new graph for this batch size/learning rate
        if (should_log) {
            printf("[FUSED] Creating new CUDA graph for batch size %d and learning rate %.5f\n", 
                  batch_size, learning_rate);
        }
        
        // Create a new graph
        cudaStatus = create_training_graph(batch_size, learning_rate);
            
        if (cudaStatus == cudaSuccess) {
            // Execute the graph
            CUDA_CHECK(cudaGraphLaunch(graph_data.instance, cuda_stream), "Launch graph");
            CUDA_CHECK(cudaStreamSynchronize(cuda_stream), "Stream synchronize after graph launch");
            
            using_graph = true;
        } else {
            // Fall back to regular execution
            cleanup_graph_resources();
            if (should_log) {
                printf("[FUSED] Falling back to regular execution...\n");
            }
        }
    }
#endif

    // If not using graph, execute operations normally
    if (!using_graph) {
        // Create CUDA events for kernel timing
        cudaEvent_t kernel_start, kernel_stop;
        float kernel_time;
        CUDA_CHECK(cudaEventCreate(&kernel_start), "Create kernel start event");
        CUDA_CHECK(cudaEventCreate(&kernel_stop), "Create kernel stop event");
        
        // ────────── Forward Pass ──────────
        if (should_log) {
            printf("[FUSED] Performing forward pass...\n");
        }
        
        // Calculate grid dimensions for hidden layer
        int total_hidden_neurons = batch_size * HIDDEN_NODES;
        int hidden_blocks = (total_hidden_neurons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        hidden_blocks = hidden_blocks > MAX_BLOCKS ? MAX_BLOCKS : hidden_blocks;
        
        // Time hidden layer kernel
        CUDA_CHECK(cudaEventRecord(kernel_start, cuda_stream), "Record hidden layer start event");
        
        // Launch hidden layer kernel (using vectorized4 version)
        hidden_layer_kernel_batch_vectorized4<<<hidden_blocks, THREADS_PER_BLOCK, HIDDEN_NODES * sizeof(float), cuda_stream>>>(
            d_batch_input, d_weight1, d_bias1, d_batch_hidden, 
            INPUT_NODES, HIDDEN_NODES, batch_size
        );
        
        CUDA_CHECK(cudaEventRecord(kernel_stop, cuda_stream), "Record hidden layer stop event");
        CUDA_CHECK(cudaEventSynchronize(kernel_stop), "Synchronize hidden layer stop event");
        CUDA_CHECK(cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop), "Calculate hidden layer time");
        if (should_log) printf("[TIMING] Hidden layer kernel: %.4f ms\n", kernel_time);
        
        // Check for kernel launch errors
        CUDA_CHECK(cudaGetLastError(), "hidden_layer_kernel_batch_vectorized4 launch");
        
        // Calculate grid dimensions for output layer
        int total_output_neurons = batch_size * OUTPUT_NODES;
        int output_blocks = (total_output_neurons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
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
        int prediction_blocks = (batch_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
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
            int input_hidden_blocks = (INPUT_NODES * HIDDEN_NODES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
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
            int hidden_output_blocks = (HIDDEN_NODES * OUTPUT_NODES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
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
    }
    
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