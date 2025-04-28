#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_profiler_api.h>  // Added for profiling support

// Uncomment to enable debug output
//#define DEBUG

// Constants for network dimensions
#define INPUT_NODES 784    // 28x28 for MNIST-like input
#define HIDDEN_NODES 128   // Hidden layer size
#define OUTPUT_NODES 10    // Output layer size

// CUDA parameters
#define THREADS_PER_BLOCK 256
#define MAX_BLOCKS 65535

// Number of benchmark iterations
#define NUM_ITERATIONS 100
#define WARMUP_ITERATIONS 10

// Benchmark options
#define ENABLE_PROFILING 0   // Set to 1 to enable CUDA profiling

// Device functions for activation
__device__ inline float relu(float x) {
    // return (x > 0.0f) ? x : 0.0f;
    return fmaxf(0.0f, x);
}

// Kernel for hidden layer forward pass (with ReLU activation)
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
        
        // Calculate dot product using global memory for weights
        for (int j = 0; j < input_size; j++) {
            sum += input[j] * weights[j * hidden_size + hidden_idx];
        }
        
        // Apply ReLU activation and store result
        hidden[hidden_idx] = relu(sum);
    }
}

// Optimized kernel with 4-way loop unrolling
__global__ void hidden_layer_kernel_batch_optimized(
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

// Advanced optimized kernel with 8-way unrolling and register caching
__global__ void hidden_layer_kernel_batch_advanced(
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
        
        // Main processing loop with compiler-directed unrolling
        int j = 0;
        #pragma unroll 8
        for (; j < (input_size / 8) * 8; j += 8) {
            sum += input[j] * weights[j * hidden_size + weight_offset];
            sum += input[j+1] * weights[(j+1) * hidden_size + weight_offset];
            sum += input[j+2] * weights[(j+2) * hidden_size + weight_offset];
            sum += input[j+3] * weights[(j+3) * hidden_size + weight_offset];
            sum += input[j+4] * weights[(j+4) * hidden_size + weight_offset];
            sum += input[j+5] * weights[(j+5) * hidden_size + weight_offset];
            sum += input[j+6] * weights[(j+6) * hidden_size + weight_offset];
            sum += input[j+7] * weights[(j+7) * hidden_size + weight_offset];
        }
        
        // Handle remaining elements with 4-way unrolling
        #pragma unroll 4
        for (; j < (input_size / 4) * 4; j += 4) {
            sum += input[j] * weights[j * hidden_size + weight_offset];
            sum += input[j+1] * weights[(j+1) * hidden_size + weight_offset];
            sum += input[j+2] * weights[(j+2) * hidden_size + weight_offset];
            sum += input[j+3] * weights[(j+3) * hidden_size + weight_offset];
        }
        
        // Handle final elements
        for (; j < input_size; j++) {
            sum += input[j] * weights[j * hidden_size + weight_offset];
        }
        
        // Apply ReLU activation and store result
        hidden[hidden_idx] = relu(sum);
    }
}

// Vectorized kernel using float2 for more efficient memory access
__global__ void hidden_layer_kernel_batch_vectorized(
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
        
        // Process two elements at a time using float2 vectorization
        int j = 0;
        int vectorized_limit = (input_size / 2) * 2;  // Round down to even number
        
        for (; j < vectorized_limit; j += 2) {
            // Load two inputs at once
            float2 input_pair;
            input_pair.x = input[j];
            input_pair.y = input[j+1];
            
            // Load two weights at once
            float2 weight_pair;
            weight_pair.x = weights[j * hidden_size + weight_offset];
            weight_pair.y = weights[(j+1) * hidden_size + weight_offset];
            
            // Compute products and accumulate
            sum += input_pair.x * weight_pair.x;
            sum += input_pair.y * weight_pair.y;
        }
        
        // Handle remaining odd element if necessary
        if (j < input_size) {
            sum += input[j] * weights[j * hidden_size + weight_offset];
        }
        
        // Apply ReLU activation and store result
        hidden[hidden_idx] = relu(sum);
    }
}

// Vectorized kernel using float4 for even better memory access
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
        hidden[hidden_idx] = fmaxf(0.0f, sum);
    }
}

// Original combined kernel using both float4 vectorization and loop unrolling
__global__ void hidden_layer_kernel_batch_combined(
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
        
        // Process with vectorized loads + loop unrolling
        // First level: process blocks of 16 elements (4 float4 operations per block)
        int j = 0;
        int vector_block_limit = (input_size / 16) * 16;
        
        #pragma unroll 2
        for (; j < vector_block_limit; j += 16) {
            // Load 4 sets of float4 inputs (16 elements total)
            float4 input_quad1, input_quad2, input_quad3, input_quad4;
            input_quad1.x = input[j];
            input_quad1.y = input[j+1];
            input_quad1.z = input[j+2];
            input_quad1.w = input[j+3];
            
            input_quad2.x = input[j+4];
            input_quad2.y = input[j+5];
            input_quad2.z = input[j+6];
            input_quad2.w = input[j+7];
            
            input_quad3.x = input[j+8];
            input_quad3.y = input[j+9];
            input_quad3.z = input[j+10];
            input_quad3.w = input[j+11];
            
            input_quad4.x = input[j+12];
            input_quad4.y = input[j+13];
            input_quad4.z = input[j+14];
            input_quad4.w = input[j+15];
            
            // Load corresponding weights
            float4 weight_quad1, weight_quad2, weight_quad3, weight_quad4;
            weight_quad1.x = weights[j * hidden_size + weight_offset];
            weight_quad1.y = weights[(j+1) * hidden_size + weight_offset];
            weight_quad1.z = weights[(j+2) * hidden_size + weight_offset];
            weight_quad1.w = weights[(j+3) * hidden_size + weight_offset];
            
            weight_quad2.x = weights[(j+4) * hidden_size + weight_offset];
            weight_quad2.y = weights[(j+5) * hidden_size + weight_offset];
            weight_quad2.z = weights[(j+6) * hidden_size + weight_offset];
            weight_quad2.w = weights[(j+7) * hidden_size + weight_offset];
            
            weight_quad3.x = weights[(j+8) * hidden_size + weight_offset];
            weight_quad3.y = weights[(j+9) * hidden_size + weight_offset];
            weight_quad3.z = weights[(j+10) * hidden_size + weight_offset];
            weight_quad3.w = weights[(j+11) * hidden_size + weight_offset];
            
            weight_quad4.x = weights[(j+12) * hidden_size + weight_offset];
            weight_quad4.y = weights[(j+13) * hidden_size + weight_offset];
            weight_quad4.z = weights[(j+14) * hidden_size + weight_offset];
            weight_quad4.w = weights[(j+15) * hidden_size + weight_offset];
            
            // Compute products and accumulate
            sum += input_quad1.x * weight_quad1.x;
            sum += input_quad1.y * weight_quad1.y;
            sum += input_quad1.z * weight_quad1.z;
            sum += input_quad1.w * weight_quad1.w;
            
            sum += input_quad2.x * weight_quad2.x;
            sum += input_quad2.y * weight_quad2.y;
            sum += input_quad2.z * weight_quad2.z;
            sum += input_quad2.w * weight_quad2.w;
            
            sum += input_quad3.x * weight_quad3.x;
            sum += input_quad3.y * weight_quad3.y;
            sum += input_quad3.z * weight_quad3.z;
            sum += input_quad3.w * weight_quad3.w;
            
            sum += input_quad4.x * weight_quad4.x;
            sum += input_quad4.y * weight_quad4.y;
            sum += input_quad4.z * weight_quad4.z;
            sum += input_quad4.w * weight_quad4.w;
        }
        
        // Process remaining elements with float4 (blocks of 4)
        int vector_limit = (input_size / 4) * 4;
        for (; j < vector_limit; j += 4) {
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
        
        // Handle remaining elements individually
        for (; j < input_size; j++) {
            sum += input[j] * weights[j * hidden_size + weight_offset];
        }
        
        // Apply ReLU activation and store result
        hidden[hidden_idx] = fmaxf(0.0f, sum);
    }
}

// New transposed kernel using float4 with loop unrolling
__global__ void hidden_layer_kernel_batch_transposed(
    const float* __restrict__ batch_input,
    const float* __restrict__ weights_transposed, 
    const float* __restrict__ bias,
    float* __restrict__ batch_hidden,
    int input_size,
    int hidden_size,
    int batch_size
) {
    // Calculate global thread ID - each thread handles one output element
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Make sure we're within bounds
    if (tid < hidden_size * batch_size) {
        int batch_idx = tid / hidden_size;     // Which batch item
        int hidden_idx = tid % hidden_size;    // Which hidden neuron
        
        // Get the bias for this hidden neuron
        float sum = bias[hidden_idx];
        
        // With transposed weights, weights_transposed[hidden_idx * input_size + j] 
        // gives weight from input j to hidden neuron hidden_idx
        const float* weight_row = weights_transposed + hidden_idx * input_size;
        const float* input = batch_input + batch_idx * input_size;
        
        // Process in chunks of 16 floats (4 float4s) with loop unrolling
        int j = 0;
        int vec16_limit = (input_size / 16) * 16;  // Multiple of 16 elements
        
        for (; j < vec16_limit; j += 16) {
            // Load 4 float4s (16 floats total)
            float4 in_val1 = *reinterpret_cast<const float4*>(input + j);
            float4 in_val2 = *reinterpret_cast<const float4*>(input + j + 4);
            float4 in_val3 = *reinterpret_cast<const float4*>(input + j + 8);
            float4 in_val4 = *reinterpret_cast<const float4*>(input + j + 12);
            
            float4 w_val1 = *reinterpret_cast<const float4*>(weight_row + j);
            float4 w_val2 = *reinterpret_cast<const float4*>(weight_row + j + 4);
            float4 w_val3 = *reinterpret_cast<const float4*>(weight_row + j + 8);
            float4 w_val4 = *reinterpret_cast<const float4*>(weight_row + j + 12);
            
            // Compute dot products and accumulate
            sum += in_val1.x * w_val1.x + in_val1.y * w_val1.y + in_val1.z * w_val1.z + in_val1.w * w_val1.w;
            sum += in_val2.x * w_val2.x + in_val2.y * w_val2.y + in_val2.z * w_val2.z + in_val2.w * w_val2.w;
            sum += in_val3.x * w_val3.x + in_val3.y * w_val3.y + in_val3.z * w_val3.z + in_val3.w * w_val3.w;
            sum += in_val4.x * w_val4.x + in_val4.y * w_val4.y + in_val4.z * w_val4.z + in_val4.w * w_val4.w;
        }
        
        // Process remaining elements with individual float4 (4 floats at a time)
        int vec4_limit = (input_size / 4) * 4;  // Multiple of 4 elements
        for (; j < vec4_limit; j += 4) {
            float4 in_val = *reinterpret_cast<const float4*>(input + j);
            float4 w_val = *reinterpret_cast<const float4*>(weight_row + j);
            
            sum += in_val.x * w_val.x + in_val.y * w_val.y + in_val.z * w_val.z + in_val.w * w_val.w;
        }
        
        // Handle final elements individually
        for (; j < input_size; j++) {
            sum += input[j] * weight_row[j];
        }
        
        // Apply ReLU and store result
        batch_hidden[batch_idx * hidden_size + hidden_idx] = fmaxf(0.0f, sum);
    }
}

// New kernel using shared memory for input data and float4 vectorization
__global__ void hidden_layer_kernel_batch_shared_input(
    const float* __restrict__ batch_input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ batch_hidden,
    int input_size,
    int hidden_size,
    int batch_size
) {
    // Calculate how much shared memory we need
    extern __shared__ float shared_mem[];
    
    // Divide shared memory: 
    // First part: bias - size = hidden_size
    // Second part: input tile buffer - size determined by available shared memory
    float* s_bias = shared_mem;
    
    // Determine optimal tile size based on shared memory constraints
    // Each block handles a subset of hidden neurons and one batch item at a time
    const int TILE_SIZE = min(64, static_cast<int>((49152 - (hidden_size * sizeof(float))) / sizeof(float)));
    float* s_input_tile = s_bias + hidden_size;
    
    // Thread indices
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Load bias into shared memory
    if (tid < hidden_size) {
        s_bias[tid] = bias[tid];
    }
    
    // Calculate global thread ID
    int global_tid = blockIdx.x * blockDim.x + tid;
    
    // Each thread calculates one hidden neuron activation for one image
    if (global_tid < hidden_size * batch_size) {
        int batch_idx = global_tid / hidden_size;    // Which image in the batch
        int hidden_idx = global_tid % hidden_size;   // Which hidden neuron
        
        // Start with bias
        float sum = s_bias[hidden_idx];
        
        // Base pointer for this image's input
        const float* input = batch_input + batch_idx * input_size;
        
        // Process input in tiles that fit in shared memory
        for (int tile_start = 0; tile_start < input_size; tile_start += TILE_SIZE) {
            int tile_end = min(tile_start + TILE_SIZE, static_cast<int>(input_size));
            int tile_size = tile_end - tile_start;
            
            // Collaboratively load input tile into shared memory
            for (int i = tid; i < tile_size; i += block_size) {
                s_input_tile[i] = input[tile_start + i];
            }
            
            // Make sure all threads have loaded their data
            __syncthreads();
            
            // Process the tile with vectorized operations where possible
            int j = 0;
            
            // Process in chunks of 4 elements (float4)
            int vec_limit = (tile_size / 4) * 4;
            for (; j < vec_limit; j += 4) {
                // Access input data from shared memory (faster)
                float4 input_quad;
                input_quad.x = s_input_tile[j];
                input_quad.y = s_input_tile[j+1];
                input_quad.z = s_input_tile[j+2];
                input_quad.w = s_input_tile[j+3];
                
                // Still need to load weights from global memory (different for each neuron)
                float4 weight_quad;
                weight_quad.x = weights[(tile_start + j) * hidden_size + hidden_idx];
                weight_quad.y = weights[(tile_start + j + 1) * hidden_size + hidden_idx];
                weight_quad.z = weights[(tile_start + j + 2) * hidden_size + hidden_idx];
                weight_quad.w = weights[(tile_start + j + 3) * hidden_size + hidden_idx];
                
                // Compute dot product
                sum += input_quad.x * weight_quad.x;
                sum += input_quad.y * weight_quad.y;
                sum += input_quad.z * weight_quad.z;
                sum += input_quad.w * weight_quad.w;
            }
            
            // Handle remaining elements
            for (; j < tile_size; j++) {
                sum += s_input_tile[j] * weights[(tile_start + j) * hidden_size + hidden_idx];
            }
            
            // Synchronize before loading the next tile
            __syncthreads();
        }
        
        // Apply ReLU activation and store result
        batch_hidden[batch_idx * hidden_size + hidden_idx] = relu(sum);
    }
}

// Tiled approach with both input and weight data in shared memory
__global__ void hidden_layer_kernel_batch_shared_tiled(
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
    
    // Shared memory allocation - need to be dynamic because sizes vary
    extern __shared__ float shared_mem[];
    float* s_input = shared_mem;                              // [TILE_DIM_INPUT]
    float* s_weights = s_input + TILE_DIM_INPUT;              // [TILE_DIM_INPUT][TILE_DIM_HIDDEN]
    float* s_bias = s_weights + (TILE_DIM_INPUT * TILE_DIM_HIDDEN); // [TILE_DIM_HIDDEN]
    
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

// Error checking macro
#define CUDA_CHECK(call, msg) \
    do { \
        cudaError_t status = call; \
        if (status != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s - %s\n", msg, cudaGetErrorString(status)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Utility function to transpose weights for improved memory access pattern
// Uses a block-based approach for better cache efficiency
__global__ void transpose_weights_kernel(const float* weights, float* weights_transposed, 
                                        int input_size, int hidden_size) {
    // Use shared memory to cache a tile of the matrix
    __shared__ float tile[32][32+1]; // +1 to avoid bank conflicts
    
    // Determine the block's position
    int block_row = blockIdx.y * 32;
    int block_col = blockIdx.x * 32;
    
    // Global position for this thread
    int input_idx = block_row + threadIdx.y;
    int hidden_idx = block_col + threadIdx.x;
    
    // Transpose 32x32 blocks to maximize cache hits
    if (input_idx < input_size && hidden_idx < hidden_size) {
        // Load from global memory to shared memory
        tile[threadIdx.y][threadIdx.x] = weights[input_idx * hidden_size + hidden_idx];
    }
    
    // Ensure all threads have loaded their data to shared memory
    __syncthreads();
    
    // Transpose indices for writing
    input_idx = block_col + threadIdx.y;
    hidden_idx = block_row + threadIdx.x;
    
    // Write transposed data back to global memory
    if (input_idx < hidden_size && hidden_idx < input_size) {
        weights_transposed[input_idx * input_size + hidden_idx] = tile[threadIdx.x][threadIdx.y];
    }
}

// CPU function to transpose weights on the device
void transpose_weights(const float* d_weights, float** d_weights_transposed, 
                    int input_size, int hidden_size) {
    // Allocate memory for transposed weights if not already allocated
    if (*d_weights_transposed == NULL) {
        CUDA_CHECK(cudaMalloc(d_weights_transposed, input_size * hidden_size * sizeof(float)), 
                 "Allocate transposed weights");
    }
    
    // Set up grid and block dimensions for transposition
    // Use 32x32 blocks for optimal cache performance
    dim3 blockDim(32, 32);
    dim3 gridDim((hidden_size + blockDim.x - 1) / blockDim.x, 
                (input_size + blockDim.y - 1) / blockDim.y);
    
    // Launch transposition kernel
    transpose_weights_kernel<<<gridDim, blockDim>>>(d_weights, *d_weights_transposed, 
                                                  input_size, hidden_size);
    CUDA_CHECK(cudaGetLastError(), "Transpose weights kernel execution");
    CUDA_CHECK(cudaDeviceSynchronize(), "Transpose weights synchronization");
}

// Fill an array with random values
void fill_random(float* array, size_t size) {
    for (size_t i = 0; i < size; i++) {
        array[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // Values between -1 and 1
    }
}

// Run benchmark for a specific batch size and kernel type
void benchmark_batch_size(int batch_size, int kernel_type) {
    const char* kernel_names[] = {"original", "loop-unrolled", "advanced-optimized", 
                                 "vectorized", "vectorized4", "combined", "transposed",
                                 "shared-input", "shared-tiled"};
    printf("\nBenchmarking %s kernel with batch size %d...\n", 
           kernel_names[kernel_type], batch_size);
    
    // Allocate host memory
    size_t batch_input_size = batch_size * INPUT_NODES * sizeof(float);
    size_t batch_hidden_size = batch_size * HIDDEN_NODES * sizeof(float);
    size_t weight_size = INPUT_NODES * HIDDEN_NODES * sizeof(float);
    size_t bias_size = HIDDEN_NODES * sizeof(float);
    
    float* h_batch_input = (float*)malloc(batch_input_size);
    float* h_batch_hidden = (float*)malloc(batch_hidden_size);
    float* h_weights = (float*)malloc(weight_size);
    float* h_bias = (float*)malloc(bias_size);
    
    // Fill with random data
    fill_random(h_batch_input, batch_size * INPUT_NODES);
    fill_random(h_weights, INPUT_NODES * HIDDEN_NODES);
    fill_random(h_bias, HIDDEN_NODES);
    
    // Allocate device memory
    float *d_batch_input, *d_batch_hidden, *d_weights, *d_bias;
    float *d_weights_transposed = NULL;  // For transposed kernel
    
    CUDA_CHECK(cudaMalloc(&d_batch_input, batch_input_size), "Allocate device input");
    CUDA_CHECK(cudaMalloc(&d_batch_hidden, batch_hidden_size), "Allocate device hidden");
    CUDA_CHECK(cudaMalloc(&d_weights, weight_size), "Allocate device weights");
    CUDA_CHECK(cudaMalloc(&d_bias, bias_size), "Allocate device bias");
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_batch_input, h_batch_input, batch_input_size, 
                        cudaMemcpyHostToDevice), "Copy input to device");
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, weight_size, 
                        cudaMemcpyHostToDevice), "Copy weights to device");
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias, bias_size, 
                        cudaMemcpyHostToDevice), "Copy bias to device");
    
    // For transposed kernel, transpose the weights
    if (kernel_type == 6) {
        transpose_weights(d_weights, &d_weights_transposed, INPUT_NODES, HIDDEN_NODES);
    }
    
    // Calculate grid dimensions and shared memory size based on kernel type
    int blocks_x, blocks_y, blocks_z = 1;
    int threads_per_block;
    size_t shared_mem_size;
    int total_hidden_neurons = batch_size * HIDDEN_NODES;
    int warps_per_block, total_warps_needed;

    switch (kernel_type) {
        case 0: // Original
        case 1: // Loop unrolled
        case 2: // Advanced optimized
        case 3: // Vectorized
        case 4: // Vectorized4
        case 5: // Combined
        case 6: // Transposed
            threads_per_block = THREADS_PER_BLOCK;
            blocks_x = (total_hidden_neurons + threads_per_block - 1) / threads_per_block;
            blocks_x = blocks_x > MAX_BLOCKS ? MAX_BLOCKS : blocks_x;
            blocks_y = 1;
            shared_mem_size = HIDDEN_NODES * sizeof(float); // Just bias
            break;
            
        case 7: // Shared input
            threads_per_block = THREADS_PER_BLOCK;
            blocks_x = (total_hidden_neurons + threads_per_block - 1) / threads_per_block;
            blocks_x = blocks_x > MAX_BLOCKS ? MAX_BLOCKS : blocks_x;
            blocks_y = 1;
            // Shared memory for bias and input tile
            shared_mem_size = HIDDEN_NODES * sizeof(float) + 64 * sizeof(float); // Bias + input tile
            break;
            
        case 8: // Shared tiled
            threads_per_block = THREADS_PER_BLOCK;
            // Use 2D grid to handle multiple batch items in parallel
            // Grid dimensions: (blocks needed for inputs, blocks needed for hidden neurons, batch size)
            const int TILE_DIM_INPUT = 64;
            const int TILE_DIM_HIDDEN = 32;
            
            blocks_x = 1; // Not used directly
            blocks_y = (HIDDEN_NODES + TILE_DIM_HIDDEN - 1) / TILE_DIM_HIDDEN;
            blocks_z = batch_size;
            // Shared memory for input tile, weight tile, and bias
            shared_mem_size = (TILE_DIM_INPUT + TILE_DIM_INPUT * TILE_DIM_HIDDEN + TILE_DIM_HIDDEN) * sizeof(float);
            break;
            
        // default:
        //     printf("Invalid kernel type: %d\n", kernel_type);
        //     exit(EXIT_FAILURE);
        //     break;
    }
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start), "Create start event");
    CUDA_CHECK(cudaEventCreate(&stop), "Create stop event");
    
    // Setup grid dimensions
    dim3 grid(blocks_x, blocks_y, blocks_z);
    dim3 block(threads_per_block);
    
    // Warm up
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        switch (kernel_type) {
            case 0: // Original
                hidden_layer_kernel_batch<<<grid, block, shared_mem_size>>>(
                    d_batch_input, d_weights, d_bias, d_batch_hidden, 
                    INPUT_NODES, HIDDEN_NODES, batch_size
                );
                break;
            case 1: // Loop unrolled
                hidden_layer_kernel_batch_optimized<<<grid, block, shared_mem_size>>>(
                    d_batch_input, d_weights, d_bias, d_batch_hidden, 
                    INPUT_NODES, HIDDEN_NODES, batch_size
                );
                break;
            case 2: // Advanced optimized
                hidden_layer_kernel_batch_advanced<<<grid, block, shared_mem_size>>>(
                    d_batch_input, d_weights, d_bias, d_batch_hidden, 
                    INPUT_NODES, HIDDEN_NODES, batch_size
                );
                break;
            case 3: // Vectorized
                hidden_layer_kernel_batch_vectorized<<<grid, block, shared_mem_size>>>(
                    d_batch_input, d_weights, d_bias, d_batch_hidden, 
                    INPUT_NODES, HIDDEN_NODES, batch_size
                );
                break;
            case 4: // Vectorized4
                hidden_layer_kernel_batch_vectorized4<<<grid, block, shared_mem_size>>>(
                    d_batch_input, d_weights, d_bias, d_batch_hidden, 
                    INPUT_NODES, HIDDEN_NODES, batch_size
                );
                break;
            case 5: // Combined
                hidden_layer_kernel_batch_combined<<<grid, block, shared_mem_size>>>(
                    d_batch_input, d_weights, d_bias, d_batch_hidden, 
                    INPUT_NODES, HIDDEN_NODES, batch_size
                );
                break;
            case 6: // Transposed
                hidden_layer_kernel_batch_transposed<<<grid, block, 0>>>(
                    d_batch_input, d_weights_transposed, d_bias, d_batch_hidden, 
                    INPUT_NODES, HIDDEN_NODES, batch_size
                );
                break;
            case 7: // Shared input
                hidden_layer_kernel_batch_shared_input<<<grid, block, shared_mem_size>>>(
                    d_batch_input, d_weights, d_bias, d_batch_hidden, 
                    INPUT_NODES, HIDDEN_NODES, batch_size
                );
                break;
            case 8: // Shared tiled
                hidden_layer_kernel_batch_shared_tiled<<<grid, block, shared_mem_size>>>(
                    d_batch_input, d_weights, d_bias, d_batch_hidden, 
                    INPUT_NODES, HIDDEN_NODES, batch_size
                );
                break;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize(), "Warmup synchronize");
    
    // Enable profiling if requested
    if (ENABLE_PROFILING) {
        CUDA_CHECK(cudaProfilerStart(), "Start CUDA profiler");
    }
    
    // Benchmark timing
    float times[NUM_ITERATIONS];
    float total_time = 0.0f;
    
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        float milliseconds = 0.0f;
        
        CUDA_CHECK(cudaEventRecord(start), "Record start event");
        
        switch (kernel_type) {
            case 0: // Original
                hidden_layer_kernel_batch<<<grid, block, shared_mem_size>>>(
                    d_batch_input, d_weights, d_bias, d_batch_hidden, 
                    INPUT_NODES, HIDDEN_NODES, batch_size
                );
                break;
            case 1: // Loop unrolled
                hidden_layer_kernel_batch_optimized<<<grid, block, shared_mem_size>>>(
                    d_batch_input, d_weights, d_bias, d_batch_hidden, 
                    INPUT_NODES, HIDDEN_NODES, batch_size
                );
                break;
            case 2: // Advanced optimized
                hidden_layer_kernel_batch_advanced<<<grid, block, shared_mem_size>>>(
                    d_batch_input, d_weights, d_bias, d_batch_hidden, 
                    INPUT_NODES, HIDDEN_NODES, batch_size
                );
                break;
            case 3: // Vectorized
                hidden_layer_kernel_batch_vectorized<<<grid, block, shared_mem_size>>>(
                    d_batch_input, d_weights, d_bias, d_batch_hidden, 
                    INPUT_NODES, HIDDEN_NODES, batch_size
                );
                break;
            case 4: // Vectorized4
                hidden_layer_kernel_batch_vectorized4<<<grid, block, shared_mem_size>>>(
                    d_batch_input, d_weights, d_bias, d_batch_hidden, 
                    INPUT_NODES, HIDDEN_NODES, batch_size
                );
                break;
            case 5: // Combined
                hidden_layer_kernel_batch_combined<<<grid, block, shared_mem_size>>>(
                    d_batch_input, d_weights, d_bias, d_batch_hidden, 
                    INPUT_NODES, HIDDEN_NODES, batch_size
                );
                break;
            case 6: // Transposed
                hidden_layer_kernel_batch_transposed<<<grid, block, 0>>>(
                    d_batch_input, d_weights_transposed, d_bias, d_batch_hidden, 
                    INPUT_NODES, HIDDEN_NODES, batch_size
                );
                break;
            case 7: // Shared input
                hidden_layer_kernel_batch_shared_input<<<grid, block, shared_mem_size>>>(
                    d_batch_input, d_weights, d_bias, d_batch_hidden, 
                    INPUT_NODES, HIDDEN_NODES, batch_size
                );
                break;
            case 8: // Shared tiled
                hidden_layer_kernel_batch_shared_tiled<<<grid, block, shared_mem_size>>>(
                    d_batch_input, d_weights, d_bias, d_batch_hidden, 
                    INPUT_NODES, HIDDEN_NODES, batch_size
                );
                break;
        }
        
        CUDA_CHECK(cudaEventRecord(stop), "Record stop event");
        CUDA_CHECK(cudaEventSynchronize(stop), "Synchronize stop event");
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop), "Calculate elapsed time");
        
        times[i] = milliseconds;
        total_time += milliseconds;
    }
    
    // Disable profiling if it was enabled
    if (ENABLE_PROFILING) {
        CUDA_CHECK(cudaProfilerStop(), "Stop CUDA profiler");
    }
    
    // Calculate statistics
    float avg_time = total_time / NUM_ITERATIONS;
    
    // Calculate min and max times
    float min_time = times[0];
    float max_time = times[0];
    for (int i = 1; i < NUM_ITERATIONS; i++) {
        if (times[i] < min_time) min_time = times[i];
        if (times[i] > max_time) max_time = times[i];
    }
    
    // Calculate standard deviation
    float variance = 0.0f;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        variance += (times[i] - avg_time) * (times[i] - avg_time);
    }
    variance /= NUM_ITERATIONS;
    float std_dev = sqrt(variance);
    
    // Calculate throughput metrics
    float ops_per_image = 2.0f * INPUT_NODES * HIDDEN_NODES; // Multiply-add for each weight
    float total_gflops = (batch_size * ops_per_image) / (avg_time * 1e6f); // GFLOPS
    float memory_bandwidth = ((batch_size * INPUT_NODES + // Input data
                              batch_size * HIDDEN_NODES + // Output data
                              INPUT_NODES * HIDDEN_NODES + // Weights
                              HIDDEN_NODES) * sizeof(float)) / (avg_time * 1e6f); // GB/s
    
    // Print results
    printf("Results for %s kernel with batch size %d:\n", kernel_names[kernel_type], batch_size);
    printf("  Average time: %.4f ms\n", avg_time);
    printf("  Min time:     %.4f ms\n", min_time);
    printf("  Max time:     %.4f ms\n", max_time);
    printf("  Std dev:      %.4f ms\n", std_dev);
    
    // Print performance metrics
    printf("  Performance metrics:\n");
    printf("    Throughput:        %.2f images/sec\n", (batch_size * 1000.0f) / avg_time);
    printf("    Computational:     %.2f GFLOPS\n", total_gflops);
    printf("    Memory bandwidth:  %.2f GB/s\n", memory_bandwidth);
    
    // Free device memory
    cudaFree(d_batch_input);
    cudaFree(d_batch_hidden);
    cudaFree(d_weights);
    cudaFree(d_bias);
    if (kernel_type == 6 && d_weights_transposed != NULL) {
        cudaFree(d_weights_transposed);
    }
    
    // Free host memory
    free(h_batch_input);
    free(h_batch_hidden);
    free(h_weights);
    free(h_bias);
    
    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Correctness verification
bool verify_correctness(int batch_size) {
    printf("\nVerifying correctness with batch size %d...\n", batch_size);
    
    // Allocate host memory
    size_t batch_input_size = batch_size * INPUT_NODES * sizeof(float);
    size_t batch_hidden_size = batch_size * HIDDEN_NODES * sizeof(float);
    size_t weight_size = INPUT_NODES * HIDDEN_NODES * sizeof(float);
    size_t bias_size = HIDDEN_NODES * sizeof(float);
    
    float* h_batch_input = (float*)malloc(batch_input_size);
    float* h_weights = (float*)malloc(weight_size);
    float* h_bias = (float*)malloc(bias_size);
    float* h_output_original = (float*)malloc(batch_hidden_size);
    float* h_output_optimized = (float*)malloc(batch_hidden_size);
    float* h_output_advanced = (float*)malloc(batch_hidden_size);
    float* h_output_vectorized = (float*)malloc(batch_hidden_size);
    float* h_output_vectorized4 = (float*)malloc(batch_hidden_size);
    float* h_output_combined = (float*)malloc(batch_hidden_size);
    float* h_output_transposed = (float*)malloc(batch_hidden_size);
    float* h_output_shared_input = (float*)malloc(batch_hidden_size);
    float* h_output_shared_tiled = (float*)malloc(batch_hidden_size);
    
    // Use fixed seed for reproducibility during verification
    srand(12345);
    
    // Fill with deterministic data
    fill_random(h_batch_input, batch_size * INPUT_NODES);
    fill_random(h_weights, INPUT_NODES * HIDDEN_NODES);
    fill_random(h_bias, HIDDEN_NODES);
    
    // Allocate device memory
    float *d_batch_input, *d_weights, *d_bias;
    float *d_output_original, *d_output_optimized, *d_output_advanced;
    float *d_output_vectorized, *d_output_vectorized4, *d_output_combined;
    float *d_output_transposed, *d_weights_transposed = NULL;
    float *d_output_shared_input, *d_output_shared_tiled;
    
    CUDA_CHECK(cudaMalloc(&d_batch_input, batch_input_size), "Allocate device input");
    CUDA_CHECK(cudaMalloc(&d_weights, weight_size), "Allocate device weights");
    CUDA_CHECK(cudaMalloc(&d_bias, bias_size), "Allocate device bias");
    CUDA_CHECK(cudaMalloc(&d_output_original, batch_hidden_size), "Allocate device output original");
    CUDA_CHECK(cudaMalloc(&d_output_optimized, batch_hidden_size), "Allocate device output optimized");
    CUDA_CHECK(cudaMalloc(&d_output_advanced, batch_hidden_size), "Allocate device output advanced");
    CUDA_CHECK(cudaMalloc(&d_output_vectorized, batch_hidden_size), "Allocate device output vectorized");
    CUDA_CHECK(cudaMalloc(&d_output_vectorized4, batch_hidden_size), "Allocate device output vectorized4");
    CUDA_CHECK(cudaMalloc(&d_output_combined, batch_hidden_size), "Allocate device output combined");
    CUDA_CHECK(cudaMalloc(&d_output_transposed, batch_hidden_size), "Allocate device output transposed");
    CUDA_CHECK(cudaMalloc(&d_output_shared_input, batch_hidden_size), "Allocate device output shared_input");
    CUDA_CHECK(cudaMalloc(&d_output_shared_tiled, batch_hidden_size), "Allocate device output shared_tiled");
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_batch_input, h_batch_input, batch_input_size, cudaMemcpyHostToDevice), "Copy input to device");
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, weight_size, cudaMemcpyHostToDevice), "Copy weights to device");
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias, bias_size, cudaMemcpyHostToDevice), "Copy bias to device");
    
    // Transpose weights for the transposed kernel
    transpose_weights(d_weights, &d_weights_transposed, INPUT_NODES, HIDDEN_NODES);
    
    // Setup grid dimensions for the original kernels
    int total_hidden_neurons = batch_size * HIDDEN_NODES;
    int blocks = (total_hidden_neurons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    blocks = blocks > MAX_BLOCKS ? MAX_BLOCKS : blocks;
    
    // Clear all outputs first to ensure clean state
    CUDA_CHECK(cudaMemset(d_output_original, 0, batch_hidden_size), "Clear original output");
    CUDA_CHECK(cudaMemset(d_output_optimized, 0, batch_hidden_size), "Clear optimized output");
    CUDA_CHECK(cudaMemset(d_output_advanced, 0, batch_hidden_size), "Clear advanced output");
    CUDA_CHECK(cudaMemset(d_output_vectorized, 0, batch_hidden_size), "Clear vectorized output");
    CUDA_CHECK(cudaMemset(d_output_vectorized4, 0, batch_hidden_size), "Clear vectorized4 output");
    CUDA_CHECK(cudaMemset(d_output_combined, 0, batch_hidden_size), "Clear combined output");
    CUDA_CHECK(cudaMemset(d_output_transposed, 0, batch_hidden_size), "Clear transposed output");
    CUDA_CHECK(cudaMemset(d_output_shared_input, 0, batch_hidden_size), "Clear shared_input output");
    CUDA_CHECK(cudaMemset(d_output_shared_tiled, 0, batch_hidden_size), "Clear shared_tiled output");
    
    // Run original kernel
    hidden_layer_kernel_batch<<<blocks, THREADS_PER_BLOCK, HIDDEN_NODES * sizeof(float)>>>(
        d_batch_input, d_weights, d_bias, d_output_original, 
        INPUT_NODES, HIDDEN_NODES, batch_size
    );
    CUDA_CHECK(cudaGetLastError(), "Original kernel execution");
    CUDA_CHECK(cudaDeviceSynchronize(), "Synchronize after original kernel");
    
    // Run optimized kernel (loop unrolled)
    hidden_layer_kernel_batch_optimized<<<blocks, THREADS_PER_BLOCK, HIDDEN_NODES * sizeof(float)>>>(
        d_batch_input, d_weights, d_bias, d_output_optimized, 
        INPUT_NODES, HIDDEN_NODES, batch_size
    );
    CUDA_CHECK(cudaGetLastError(), "Optimized kernel execution");
    CUDA_CHECK(cudaDeviceSynchronize(), "Synchronize after optimized kernel");
    
    // Run advanced optimized kernel
    hidden_layer_kernel_batch_advanced<<<blocks, THREADS_PER_BLOCK, HIDDEN_NODES * sizeof(float)>>>(
        d_batch_input, d_weights, d_bias, d_output_advanced, 
        INPUT_NODES, HIDDEN_NODES, batch_size
    );
    CUDA_CHECK(cudaGetLastError(), "Advanced kernel execution");
    CUDA_CHECK(cudaDeviceSynchronize(), "Synchronize after advanced kernel");
    
    // Run vectorized kernel
    hidden_layer_kernel_batch_vectorized<<<blocks, THREADS_PER_BLOCK, HIDDEN_NODES * sizeof(float)>>>(
        d_batch_input, d_weights, d_bias, d_output_vectorized, 
        INPUT_NODES, HIDDEN_NODES, batch_size
    );
    CUDA_CHECK(cudaGetLastError(), "Vectorized kernel execution");
    CUDA_CHECK(cudaDeviceSynchronize(), "Synchronize after vectorized kernel");
    
    // Run vectorized4 kernel
    hidden_layer_kernel_batch_vectorized4<<<blocks, THREADS_PER_BLOCK, HIDDEN_NODES * sizeof(float)>>>(
        d_batch_input, d_weights, d_bias, d_output_vectorized4, 
        INPUT_NODES, HIDDEN_NODES, batch_size
    );
    CUDA_CHECK(cudaGetLastError(), "Vectorized4 kernel execution");
    CUDA_CHECK(cudaDeviceSynchronize(), "Synchronize after vectorized4 kernel");
    
    // Run combined kernel
    hidden_layer_kernel_batch_combined<<<blocks, THREADS_PER_BLOCK, HIDDEN_NODES * sizeof(float)>>>(
        d_batch_input, d_weights, d_bias, d_output_combined, 
        INPUT_NODES, HIDDEN_NODES, batch_size
    );
    CUDA_CHECK(cudaGetLastError(), "Combined kernel execution");
    CUDA_CHECK(cudaDeviceSynchronize(), "Synchronize after combined kernel");
    
    // Run transposed kernel
    hidden_layer_kernel_batch_transposed<<<blocks, THREADS_PER_BLOCK, 0>>>(
        d_batch_input, d_weights_transposed, d_bias, d_output_transposed, 
        INPUT_NODES, HIDDEN_NODES, batch_size
    );
    CUDA_CHECK(cudaGetLastError(), "Transposed kernel execution");
    CUDA_CHECK(cudaDeviceSynchronize(), "Synchronize after transposed kernel");
    
    // Run shared input kernel
    size_t shared_mem_size_input = HIDDEN_NODES * sizeof(float) + 64 * sizeof(float); // Bias + input tile
    hidden_layer_kernel_batch_shared_input<<<blocks, THREADS_PER_BLOCK, shared_mem_size_input>>>(
        d_batch_input, d_weights, d_bias, d_output_shared_input, 
        INPUT_NODES, HIDDEN_NODES, batch_size
    );
    CUDA_CHECK(cudaGetLastError(), "Shared input kernel execution");
    CUDA_CHECK(cudaDeviceSynchronize(), "Synchronize after shared input kernel");
    
    // Run shared tiled kernel
    const int TILE_DIM_INPUT = 64;
    const int TILE_DIM_HIDDEN = 32;
    dim3 tiled_grid(1, (HIDDEN_NODES + TILE_DIM_HIDDEN - 1) / TILE_DIM_HIDDEN, batch_size);
    size_t shared_mem_size_tiled = (TILE_DIM_INPUT + TILE_DIM_INPUT * TILE_DIM_HIDDEN + TILE_DIM_HIDDEN) * sizeof(float);
    hidden_layer_kernel_batch_shared_tiled<<<tiled_grid, THREADS_PER_BLOCK, shared_mem_size_tiled>>>(
        d_batch_input, d_weights, d_bias, d_output_shared_tiled, 
        INPUT_NODES, HIDDEN_NODES, batch_size
    );
    CUDA_CHECK(cudaGetLastError(), "Shared tiled kernel execution");
    CUDA_CHECK(cudaDeviceSynchronize(), "Synchronize after shared tiled kernel");
    
    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_output_original, d_output_original, batch_hidden_size, cudaMemcpyDeviceToHost), 
              "Copy original output to host");
    CUDA_CHECK(cudaMemcpy(h_output_optimized, d_output_optimized, batch_hidden_size, cudaMemcpyDeviceToHost), 
              "Copy optimized output to host");
    CUDA_CHECK(cudaMemcpy(h_output_advanced, d_output_advanced, batch_hidden_size, cudaMemcpyDeviceToHost), 
              "Copy advanced output to host");
    CUDA_CHECK(cudaMemcpy(h_output_vectorized, d_output_vectorized, batch_hidden_size, cudaMemcpyDeviceToHost), 
              "Copy vectorized output to host");
    CUDA_CHECK(cudaMemcpy(h_output_vectorized4, d_output_vectorized4, batch_hidden_size, cudaMemcpyDeviceToHost), 
              "Copy vectorized4 output to host");
    CUDA_CHECK(cudaMemcpy(h_output_combined, d_output_combined, batch_hidden_size, cudaMemcpyDeviceToHost), 
              "Copy combined output to host");
    CUDA_CHECK(cudaMemcpy(h_output_transposed, d_output_transposed, batch_hidden_size, cudaMemcpyDeviceToHost), 
              "Copy transposed output to host");
    CUDA_CHECK(cudaMemcpy(h_output_shared_input, d_output_shared_input, batch_hidden_size, cudaMemcpyDeviceToHost), 
              "Copy shared input output to host");
    CUDA_CHECK(cudaMemcpy(h_output_shared_tiled, d_output_shared_tiled, batch_hidden_size, cudaMemcpyDeviceToHost), 
              "Copy shared tiled output to host");
    
    // Compare results
    bool all_equal = true;
    
    auto compare_outputs = [&all_equal, batch_size](
        const char* name, const float* reference, const float* test, float* max_diff, int* max_diff_idx) {
        bool equal = true;
        *max_diff = 0.0f;
        *max_diff_idx = -1;
        
        for (int i = 0; i < batch_size * HIDDEN_NODES; i++) {
            float diff = fabsf(reference[i] - test[i]);
            if (diff > 1e-5f) {  // Note: Using slightly less strict threshold for float comparison
                equal = false;
                all_equal = false;
                if (diff > *max_diff) {
                    *max_diff = diff;
                    *max_diff_idx = i;
                }
            }
        }
        
        printf("  %s kernel: %s", name, equal ? "PASS\n" : "FAIL - ");
        
        if (!equal) {
            printf("max diff: %e at index %d (batch=%d, neuron=%d)\n", 
                  *max_diff, *max_diff_idx,
                  *max_diff_idx / HIDDEN_NODES, *max_diff_idx % HIDDEN_NODES);
            
            // Print values at failure point
            int idx = *max_diff_idx;
            printf("    Values at failure point: original=%f, %s=%f\n", 
                   reference[idx], name, test[idx]);
        }
        
        return equal;
    };
    
    // Compare all kernels against original
    printf("Correctness Results:\n");
    
    float max_diff; 
    int max_diff_idx;
    
    compare_outputs("Optimized", h_output_original, h_output_optimized, &max_diff, &max_diff_idx);
    compare_outputs("Advanced", h_output_original, h_output_advanced, &max_diff, &max_diff_idx);
    compare_outputs("Vectorized", h_output_original, h_output_vectorized, &max_diff, &max_diff_idx);
    compare_outputs("Vectorized4", h_output_original, h_output_vectorized4, &max_diff, &max_diff_idx);
    compare_outputs("Combined", h_output_original, h_output_combined, &max_diff, &max_diff_idx);
    compare_outputs("Transposed", h_output_original, h_output_transposed, &max_diff, &max_diff_idx);
    compare_outputs("Shared-Input", h_output_original, h_output_shared_input, &max_diff, &max_diff_idx);
    compare_outputs("Shared-Tiled", h_output_original, h_output_shared_tiled, &max_diff, &max_diff_idx);
    
    // Free device memory
    cudaFree(d_batch_input);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_output_original);
    cudaFree(d_output_optimized);
    cudaFree(d_output_advanced);
    cudaFree(d_output_vectorized);
    cudaFree(d_output_vectorized4);
    cudaFree(d_output_combined);
    cudaFree(d_output_transposed);
    cudaFree(d_output_shared_input);
    cudaFree(d_output_shared_tiled);
    cudaFree(d_weights_transposed);
    
    // Free host memory
    free(h_batch_input);
    free(h_weights);
    free(h_bias);
    free(h_output_original);
    free(h_output_optimized);
    free(h_output_advanced);
    free(h_output_vectorized);
    free(h_output_vectorized4);
    free(h_output_combined);
    free(h_output_transposed);
    free(h_output_shared_input);
    free(h_output_shared_tiled);
    
    return all_equal;
}

int main(int argc, char **argv) {
    // Initialize random seed
    srand(time(NULL));
    
    // Process command line arguments
    int batch_size = 128;
    bool test_all_kernels = true;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc) {
            batch_size = atoi(argv[i+1]);
            i++;
        } else if (strcmp(argv[i], "--kernel") == 0 && i + 1 < argc) {
            int kernel = atoi(argv[i+1]);
            if (kernel >= 0 && kernel <= 8) {
                test_all_kernels = false;
                benchmark_batch_size(batch_size, kernel);
                i++;
            } else {
                printf("Invalid kernel type (0-8 allowed)\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  --batch-size N   Set batch size (default: 128)\n");
            printf("  --kernel K       Test only kernel K (0=original, 1=optimized, 2=advanced, 3=vectorized, 4=vectorized4, 5=combined, 6=transposed, 7=shared-input, 8=shared-tiled)\n");
            printf("  --help           Display this help message\n");
            return 0;
        }
    }
    
    // Test for correctness first
    bool correct = verify_correctness(32);
    if (!correct) {
        printf("ERROR: One or more optimized kernels produce incorrect results.\n");
        printf("Continuing with benchmarks, but results may not be accurate.\n");
    }
    
    if (test_all_kernels) {
        // Benchmark with different batch sizes
        int batch_sizes[] = {32, 64, 128, 256, 512, 1024};
        int num_batch_sizes = sizeof(batch_sizes) / sizeof(batch_sizes[0]);
        
        for (int kernel_type = 0; kernel_type < 9; kernel_type++) {
            printf("\n===== BENCHMARKING KERNEL TYPE %d =====\n", kernel_type);
            for (int i = 0; i < num_batch_sizes; i++) {
                benchmark_batch_size(batch_sizes[i], kernel_type);
            }
        }
    }
    
    printf("\nBenchmarking complete.\n");
    printf("\nCompile with: nvcc -O3 -arch=sm_XX --use_fast_math -lineinfo hidden_layer_kernel_batch.cu -o benchmark\n");
    printf("Run with profiling: nvprof --metrics achieved_occupancy,inst_executed,gld_throughput,gst_throughput,dram_read_throughput,dram_write_throughput,flop_count_sp ./benchmark\n");
    printf("\nKernel options:\n");
    printf("  0 = original - Basic implementation with shared memory for bias\n");
    printf("  1 = loop-unrolled - Basic implementation with 4-way loop unrolling\n");
    printf("  2 = advanced-optimized - Advanced kernel with 8-way loop unrolling and register caching\n");
    printf("  3 = vectorized - Uses float2 vectorized loads for better memory access\n");
    printf("  4 = vectorized4 - Uses float4 vectorized loads for better memory access\n");
    printf("  5 = combined - Combines float4 vectorization with aggressive loop unrolling\n");
    printf("  6 = transposed - Uses transposed weight matrix for coalesced memory access\n");
    printf("  7 = shared-input - Uses shared memory for input data and float4 vectorization\n");
    printf("  8 = shared-tiled - Uses tiled approach with both input and weight data in shared memory\n");
    printf("\nNOTE: The transposed kernel (6) should perform best as it addresses the fundamental memory access pattern issue\n");
    
    return 0;
} 