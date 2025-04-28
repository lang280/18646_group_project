#include "backward.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

// Device functions for activation derivatives
__device__ inline float relu_derivative(float x) {
    return (x > 0.0f) ? 1.0f : 0.0f;
}

__device__ inline float sigmoid_derivative(float y) {
    return y * (1.0f - y);
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