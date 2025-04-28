#include "forward.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

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

// Tiled implementation of hidden layer forward pass with vectorized loads
__global__ void hidden_layer_kernel_batch_tiled(
    const float* __restrict__ batch_input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ batch_hidden,
    int input_size,
    int hidden_size,
    int batch_size
) {
    // Define tile sizes
    const int TILE_DIM_INPUT = 64;  // Size of input tile
    const int TILE_DIM_HIDDEN = 32; // Size of hidden tile
    
    // Shared memory allocation
    __shared__ float shared_mem[11 * 1024];
    float* s_input = shared_mem;                               // [TILE_DIM_INPUT]
    float* s_weights = s_input + TILE_DIM_INPUT;               // [TILE_DIM_INPUT][TILE_DIM_HIDDEN]
    float* s_bias = s_weights + (TILE_DIM_INPUT * TILE_DIM_HIDDEN);  // [TILE_DIM_HIDDEN]
    
    // Thread indices
    int tx = threadIdx.x % TILE_DIM_HIDDEN;  // Local hidden neuron ID
    int ty = threadIdx.x / TILE_DIM_HIDDEN;  // Local input ID for loading
    
    // Global indices
    int hidden_idx = blockIdx.y * TILE_DIM_HIDDEN + tx;  // Global hidden neuron ID
    int batch_idx = blockIdx.z;                          // Batch item
    
    // Local register for accumulating partial sums
    float sum = 0.0f;
    
    // Load bias into shared memory
    if (ty == 0 && hidden_idx < hidden_size) {
        s_bias[tx] = bias[hidden_idx];
        sum = bias[hidden_idx];  // Initialize sum with bias
    }
    
    // Process input in tiles
    for (int tile_start = 0; tile_start < input_size; tile_start += TILE_DIM_INPUT) {
        __syncthreads();  // Ensure previous tile processing is complete
        
        // Collaboratively load input tile into shared memory
        if (ty < TILE_DIM_INPUT/4 && (tile_start + ty*4) < input_size && batch_idx < batch_size) {
            int input_idx = tile_start + ty*4;
            // Load 4 input values at once using float4
            if (input_idx + 3 < input_size) {
                float4 input_quad = *reinterpret_cast<const float4*>(&batch_input[batch_idx * input_size + input_idx]);
                s_input[ty*4] = input_quad.x;
                s_input[ty*4+1] = input_quad.y;
                s_input[ty*4+2] = input_quad.z;
                s_input[ty*4+3] = input_quad.w;
            } else {
                // Handle boundary conditions
                for (int i = 0; i < 4 && (input_idx + i) < input_size; i++) {
                    s_input[ty*4+i] = batch_input[batch_idx * input_size + input_idx + i];
                }
            }
        }
        
        // Collaboratively load weight tile into shared memory
        // Each thread loads multiple weights
        for (int i = threadIdx.x; i < TILE_DIM_INPUT * TILE_DIM_HIDDEN; i += blockDim.x) {
            int input_offset = i / TILE_DIM_HIDDEN;
            int hidden_offset = i % TILE_DIM_HIDDEN;
            
            int input_idx = tile_start + input_offset;
            int hid_idx = blockIdx.y * TILE_DIM_HIDDEN + hidden_offset;
            
            if (input_idx < input_size && hid_idx < hidden_size) {
                s_weights[input_offset * TILE_DIM_HIDDEN + hidden_offset] = 
                    weights[input_idx * hidden_size + hid_idx];
            } else {
                s_weights[input_offset * TILE_DIM_HIDDEN + hidden_offset] = 0.0f;
            }
        }
        
        __syncthreads();  // Ensure all data is loaded
        
        // Compute partial dot products within this tile
        if (hidden_idx < hidden_size && batch_idx < batch_size) {
            // Process the tile with vectorized operations where possible
            for (int i = 0; i < TILE_DIM_INPUT; i += 4) {
                if (tile_start + i + 3 < input_size) {
                    sum += s_input[i] * s_weights[i * TILE_DIM_HIDDEN + tx];
                    sum += s_input[i+1] * s_weights[(i+1) * TILE_DIM_HIDDEN + tx];
                    sum += s_input[i+2] * s_weights[(i+2) * TILE_DIM_HIDDEN + tx];
                    sum += s_input[i+3] * s_weights[(i+3) * TILE_DIM_HIDDEN + tx];
                } else {
                    for (int j = i; j < TILE_DIM_INPUT && (tile_start + j) < input_size; j++) {
                        sum += s_input[j] * s_weights[j * TILE_DIM_HIDDEN + tx];
                    }
                    break;
                }
            }
        }
    }
    
    // Write result to global memory
    if (hidden_idx < hidden_size && batch_idx < batch_size) {
        batch_hidden[batch_idx * hidden_size + hidden_idx] = relu(sum);
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