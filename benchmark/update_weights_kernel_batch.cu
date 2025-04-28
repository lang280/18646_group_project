#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_profiler_api.h>
#include <math.h>

// Constants for network dimensions
#define INPUT_NODES 784
#define HIDDEN_NODES 128
#define THREADS_PER_BLOCK 256
#define MAX_BLOCKS 65535

// Number of benchmark iterations
#define NUM_ITERATIONS 100
#define WARMUP_ITERATIONS 10

// Error checking macro
#define CUDA_CHECK(call, msg) \
    do { \
        cudaError_t status = call; \
        if (status != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s - %s\n", msg, cudaGetErrorString(status)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Device activation function derivatives for testing
__device__ inline double relu_derivative(double x) {
    return (x > 0.0) ? 1.0 : 0.0;
}

// Original kernel to update weights and biases for hidden layer
__global__ void update_input_hidden_weights_kernel(
    const double* __restrict__ batch_input,
    const double* __restrict__ batch_hidden_delta,
    double* __restrict__ weights,
    double* __restrict__ bias,
    int input_size,
    int hidden_size,
    int batch_size,
    double learning_rate
) {
    extern __shared__ double shared_mem[];
    double* s_gradient_accumulator = shared_mem;
    double* s_bias_gradient = shared_mem + blockDim.x;
    
    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + tid;
    
    // Initialize gradient accumulator in shared memory
    s_gradient_accumulator[tid] = 0.0;
    
    // Threads handling hidden neuron biases also initialize bias gradient
    if (tid < hidden_size) {
        s_bias_gradient[tid] = 0.0;
    }
    
    __syncthreads();
    
    if (global_tid < input_size * hidden_size) {
        int input_idx = global_tid / hidden_size;
        int hidden_idx = global_tid % hidden_size;
        
        // Accumulate gradients across batch directly into shared memory
        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            double input_val = batch_input[batch_idx * input_size + input_idx];
            double hidden_delta = batch_hidden_delta[batch_idx * hidden_size + hidden_idx];
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

// Advanced optimized kernel with tiled batch processing and register caching
__global__ void update_input_hidden_weights_kernel_advanced(
    const double* __restrict__ batch_input,
    const double* __restrict__ batch_hidden_delta,
    double* __restrict__ weights,
    double* __restrict__ bias,
    int input_size,
    int hidden_size,
    int batch_size,
    double learning_rate
) {
    extern __shared__ double shared_mem[];
    double* s_gradient_accumulator = shared_mem;
    double* s_bias_gradient = shared_mem + blockDim.x;
    
    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + tid;
    
    // Initialize accumulators in registers for better performance
    double gradient_sum = 0.0;
    double bias_gradient = 0.0;
    
    if (global_tid < input_size * hidden_size) {
        int input_idx = global_tid / hidden_size;
        int hidden_idx = global_tid % hidden_size;
        
        // Cache addresses for better memory access patterns
        int input_offset = input_idx;
        int hidden_offset = hidden_idx;
        
        // Use batch tiling approach for better memory coalescing
        // Each thread processes multiple batches with optimal memory access
        const int TILE_SIZE = 16; // Process 16 batches at a time
        
        for (int tile_start = 0; tile_start < batch_size; tile_start += TILE_SIZE) {
            int tile_end = min(tile_start + TILE_SIZE, batch_size);
            
            // Process batches in a tile with 8-way unrolling where possible
            int batch_idx = tile_start;
            
            #pragma unroll 8
            for (; batch_idx + 7 < tile_end; batch_idx += 8) {
                // Prefetch all input values
                double in0 = batch_input[(batch_idx+0) * input_size + input_offset];
                double in1 = batch_input[(batch_idx+1) * input_size + input_offset];
                double in2 = batch_input[(batch_idx+2) * input_size + input_offset];
                double in3 = batch_input[(batch_idx+3) * input_size + input_offset];
                double in4 = batch_input[(batch_idx+4) * input_size + input_offset];
                double in5 = batch_input[(batch_idx+5) * input_size + input_offset];
                double in6 = batch_input[(batch_idx+6) * input_size + input_offset];
                double in7 = batch_input[(batch_idx+7) * input_size + input_offset];
                
                // Prefetch all hidden deltas
                double hd0 = batch_hidden_delta[(batch_idx+0) * hidden_size + hidden_offset];
                double hd1 = batch_hidden_delta[(batch_idx+1) * hidden_size + hidden_offset];
                double hd2 = batch_hidden_delta[(batch_idx+2) * hidden_size + hidden_offset];
                double hd3 = batch_hidden_delta[(batch_idx+3) * hidden_size + hidden_offset];
                double hd4 = batch_hidden_delta[(batch_idx+4) * hidden_size + hidden_offset];
                double hd5 = batch_hidden_delta[(batch_idx+5) * hidden_size + hidden_offset];
                double hd6 = batch_hidden_delta[(batch_idx+6) * hidden_size + hidden_offset];
                double hd7 = batch_hidden_delta[(batch_idx+7) * hidden_size + hidden_offset];
                
                // Update gradient sum
                gradient_sum += in0 * hd0;
                gradient_sum += in1 * hd1;
                gradient_sum += in2 * hd2;
                gradient_sum += in3 * hd3;
                gradient_sum += in4 * hd4;
                gradient_sum += in5 * hd5;
                gradient_sum += in6 * hd6;
                gradient_sum += in7 * hd7;
                
                // Update bias gradient if this thread handles a bias
                if (input_idx == 0) {
                    bias_gradient += hd0;
                    bias_gradient += hd1;
                    bias_gradient += hd2;
                    bias_gradient += hd3;
                    bias_gradient += hd4;
                    bias_gradient += hd5;
                    bias_gradient += hd6;
                    bias_gradient += hd7;
                }
            }
            
            // Handle remaining batches in tile
            for (; batch_idx < tile_end; batch_idx++) {
                double input_val = batch_input[batch_idx * input_size + input_offset];
                double hidden_delta = batch_hidden_delta[batch_idx * hidden_size + hidden_offset];
                
                gradient_sum += input_val * hidden_delta;
                
                if (input_idx == 0) {
                    bias_gradient += hidden_delta;
                }
            }
        }
        
        // Directly update weight without using shared memory
        double weight_update = learning_rate * gradient_sum / batch_size;
        weights[input_idx * hidden_size + hidden_idx] -= weight_update;
        
        // Update bias directly if this thread is responsible for a bias
        if (input_idx == 0 && hidden_idx < hidden_size) {
            bias[hidden_idx] -= learning_rate * bias_gradient / batch_size;
        }
    }
}

// Vectorized kernel using double2 for more efficient memory access
__global__ void update_input_hidden_weights_kernel_vectorized2(
    const double* __restrict__ batch_input,
    const double* __restrict__ batch_hidden_delta,
    double* __restrict__ weights,
    double* __restrict__ bias,
    int input_size,
    int hidden_size,
    int batch_size,
    double learning_rate
) {
    extern __shared__ double shared_mem[];
    double* s_bias_gradient = shared_mem;
    
    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + tid;
    
    // Initialize accumulators in registers for better performance
    double gradient_sum = 0.0;
    double bias_gradient = 0.0;
    
    if (global_tid < input_size * hidden_size) {
        int input_idx = global_tid / hidden_size;
        int hidden_idx = global_tid % hidden_size;
        
        // Initialize bias gradients in shared memory
        if (tid < hidden_size) {
            s_bias_gradient[tid] = 0.0;
        }
        __syncthreads();
        
        // Only process even batch sizes with vectorized loads
        // Process in pairs using double2 for better memory throughput
        int vectorized_limit = batch_size & ~1; // Round down to even number
        
        for (int batch_idx = 0; batch_idx < vectorized_limit; batch_idx += 2) {
            // Use double2 to load two inputs at once
            double2 input_pair;
            double2 hidden_delta_pair;
            
            // Load two consecutive inputs
            input_pair.x = batch_input[(batch_idx) * input_size + input_idx];
            input_pair.y = batch_input[(batch_idx+1) * input_size + input_idx];
            
            // Load two consecutive hidden deltas
            hidden_delta_pair.x = batch_hidden_delta[(batch_idx) * hidden_size + hidden_idx];
            hidden_delta_pair.y = batch_hidden_delta[(batch_idx+1) * hidden_size + hidden_idx];
            
            // Compute products and accumulate
            gradient_sum += input_pair.x * hidden_delta_pair.x;
            gradient_sum += input_pair.y * hidden_delta_pair.y;
            
            // Accumulate bias gradients if this thread handles a bias
            if (input_idx == 0) {
                // Use atomicAdd for bias gradient accumulation to avoid race conditions
                atomicAdd(&s_bias_gradient[hidden_idx], hidden_delta_pair.x + hidden_delta_pair.y);
            }
        }
        
        // Handle the remaining odd batch if necessary
        if (vectorized_limit < batch_size) {
            int batch_idx = vectorized_limit;
            double input_val = batch_input[batch_idx * input_size + input_idx];
            double hidden_delta = batch_hidden_delta[batch_idx * hidden_size + hidden_idx];
            
            gradient_sum += input_val * hidden_delta;
            
            if (input_idx == 0) {
                atomicAdd(&s_bias_gradient[hidden_idx], hidden_delta);
            }
        }
        
        __syncthreads();
        
        // Update weight
        double weight_update = learning_rate * gradient_sum / batch_size;
        weights[input_idx * hidden_size + hidden_idx] -= weight_update;
        
        // Update bias if this thread is responsible for it
        if (input_idx == 0 && hidden_idx < hidden_size) {
            bias[hidden_idx] -= learning_rate * s_bias_gradient[hidden_idx] / batch_size;
        }
    }
}

// Improved vectorized kernel using double4 for even better memory throughput
__global__ void update_input_hidden_weights_kernel_vectorized4(
    const double* __restrict__ batch_input,
    const double* __restrict__ batch_hidden_delta,
    double* __restrict__ weights,
    double* __restrict__ bias,
    int input_size,
    int hidden_size,
    int batch_size,
    double learning_rate
) {
    extern __shared__ double shared_mem[];
    double* s_bias_gradient = shared_mem;
    
    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + tid;
    
    // Initialize accumulators in registers for better performance
    double gradient_sum = 0.0;
    double bias_gradient = 0.0;
    
    if (global_tid < input_size * hidden_size) {
        int input_idx = global_tid / hidden_size;
        int hidden_idx = global_tid % hidden_size;
        
        // Initialize bias gradients in shared memory
        if (tid < hidden_size) {
            s_bias_gradient[tid] = 0.0;
        }
        __syncthreads();
        
        // Use double4 to process 4 batches at once
        // This allows loading/processing 4 batch elements in a single operation
        int vectorized_limit = batch_size & ~3; // Round down to multiple of 4
        
        // Cache input offset and hidden offset for better memory access pattern
        int input_offset = input_idx;
        int hidden_offset = hidden_idx;
        
        // Process batches in groups of 4 using double4 loading pattern
        for (int batch_idx = 0; batch_idx < vectorized_limit; batch_idx += 4) {
            // Load 4 consecutive inputs
            double in0 = batch_input[(batch_idx+0) * input_size + input_offset];
            double in1 = batch_input[(batch_idx+1) * input_size + input_offset];
            double in2 = batch_input[(batch_idx+2) * input_size + input_offset];
            double in3 = batch_input[(batch_idx+3) * input_size + input_offset];
            
            // Load 4 consecutive hidden deltas
            double hd0 = batch_hidden_delta[(batch_idx+0) * hidden_size + hidden_offset];
            double hd1 = batch_hidden_delta[(batch_idx+1) * hidden_size + hidden_offset];
            double hd2 = batch_hidden_delta[(batch_idx+2) * hidden_size + hidden_offset];
            double hd3 = batch_hidden_delta[(batch_idx+3) * hidden_size + hidden_offset];
            
            // Compute 4 products at once and accumulate
            gradient_sum += in0 * hd0;
            gradient_sum += in1 * hd1;
            gradient_sum += in2 * hd2;
            gradient_sum += in3 * hd3;
            
            // Accumulate bias gradients if this thread handles a bias
            if (input_idx == 0) {
                // Use a single atomic operation for better performance
                double bias_delta_sum = hd0 + hd1 + hd2 + hd3;
                atomicAdd(&s_bias_gradient[hidden_idx], bias_delta_sum);
            }
        }
        
        // Handle the remaining items (less than 4)
        for (int batch_idx = vectorized_limit; batch_idx < batch_size; batch_idx++) {
            double input_val = batch_input[batch_idx * input_size + input_offset];
            double hidden_delta = batch_hidden_delta[batch_idx * hidden_size + hidden_offset];
            
            gradient_sum += input_val * hidden_delta;
            
            if (input_idx == 0) {
                atomicAdd(&s_bias_gradient[hidden_idx], hidden_delta);
            }
        }
        
        __syncthreads();
        
        // Update weight with a single write operation
        // Directly compute weight update to minimize register usage
        weights[input_idx * hidden_size + hidden_idx] -= learning_rate * gradient_sum / batch_size;
        
        // Update bias if this thread is responsible
        if (input_idx == 0 && hidden_idx < hidden_size) {
            bias[hidden_idx] -= learning_rate * s_bias_gradient[hidden_idx] / batch_size;
        }
    }
}

// Fill arrays with random values to simulate real data
void fill_random(double* array, size_t size) {
    for (size_t i = 0; i < size; i++) {
        array[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0; // Values between -1 and 1
    }
}

// Run benchmark for a specific batch size and kernel type
void benchmark_batch_size(int batch_size, int kernel_type) {
    const char* kernel_names[] = {"original", "advanced", "vectorized2", "vectorized4"};
    printf("\nBenchmarking %s kernel with batch size %d...\n", 
           kernel_names[kernel_type], batch_size);
    
    // Allocate host memory
    size_t batch_input_size = batch_size * INPUT_NODES * sizeof(double);
    size_t batch_hidden_delta_size = batch_size * HIDDEN_NODES * sizeof(double);
    size_t weight_size = INPUT_NODES * HIDDEN_NODES * sizeof(double);
    size_t bias_size = HIDDEN_NODES * sizeof(double);
    
    double* h_batch_input = (double*)malloc(batch_input_size);
    double* h_batch_hidden_delta = (double*)malloc(batch_hidden_delta_size);
    double* h_weights = (double*)malloc(weight_size);
    double* h_bias = (double*)malloc(bias_size);
    double* h_weights_reference = (double*)malloc(weight_size);
    double* h_bias_reference = (double*)malloc(bias_size);
    
    // Fill with random data
    fill_random(h_batch_input, batch_size * INPUT_NODES);
    fill_random(h_batch_hidden_delta, batch_size * HIDDEN_NODES);
    fill_random(h_weights, INPUT_NODES * HIDDEN_NODES);
    fill_random(h_bias, HIDDEN_NODES);
    
    // Make a copy for reference
    memcpy(h_weights_reference, h_weights, weight_size);
    memcpy(h_bias_reference, h_bias, bias_size);
    
    // Allocate device memory
    double *d_batch_input, *d_batch_hidden_delta, *d_weights, *d_bias;
    CUDA_CHECK(cudaMalloc(&d_batch_input, batch_input_size), "Allocate device input");
    CUDA_CHECK(cudaMalloc(&d_batch_hidden_delta, batch_hidden_delta_size), "Allocate device hidden delta");
    CUDA_CHECK(cudaMalloc(&d_weights, weight_size), "Allocate device weights");
    CUDA_CHECK(cudaMalloc(&d_bias, bias_size), "Allocate device bias");
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_batch_input, h_batch_input, batch_input_size, 
                        cudaMemcpyHostToDevice), "Copy input to device");
    CUDA_CHECK(cudaMemcpy(d_batch_hidden_delta, h_batch_hidden_delta, batch_hidden_delta_size, 
                        cudaMemcpyHostToDevice), "Copy hidden delta to device");
    
    // Learning rate
    double learning_rate = 0.01;
    
    // Calculate grid dimensions
    int total_weights = INPUT_NODES * HIDDEN_NODES;
    int blocks = (total_weights + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    blocks = blocks > MAX_BLOCKS ? MAX_BLOCKS : blocks;
    
    // Set shared memory size based on kernel type
    size_t shared_mem_size;
    switch (kernel_type) {
        case 0: // Original
            shared_mem_size = (THREADS_PER_BLOCK + HIDDEN_NODES) * sizeof(double);
            break;
        case 1: // Advanced
            shared_mem_size = (THREADS_PER_BLOCK + HIDDEN_NODES) * sizeof(double);
            break;
        case 2: // Vectorized2
        case 3: // Vectorized4
            shared_mem_size = HIDDEN_NODES * sizeof(double);
            break;
        default:
            shared_mem_size = (THREADS_PER_BLOCK + HIDDEN_NODES) * sizeof(double);
    }
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start), "Create start event");
    CUDA_CHECK(cudaEventCreate(&stop), "Create stop event");
    
    // First run original kernel to get reference results
    // Reset device weights and bias to initial values
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, weight_size, 
                        cudaMemcpyHostToDevice), "Copy weights to device");
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias, bias_size, 
                        cudaMemcpyHostToDevice), "Copy bias to device");
    
    // Run reference kernel
    update_input_hidden_weights_kernel<<<blocks, THREADS_PER_BLOCK, 
                                        (THREADS_PER_BLOCK + HIDDEN_NODES) * sizeof(double)>>>(
        d_batch_input, d_batch_hidden_delta, d_weights, d_bias, 
        INPUT_NODES, HIDDEN_NODES, batch_size, learning_rate
    );
    CUDA_CHECK(cudaGetLastError(), "Reference kernel execution");
    CUDA_CHECK(cudaDeviceSynchronize(), "Synchronize after reference kernel");
    
    // Copy reference results back
    CUDA_CHECK(cudaMemcpy(h_weights_reference, d_weights, weight_size, 
                        cudaMemcpyDeviceToHost), "Copy reference weights to host");
    CUDA_CHECK(cudaMemcpy(h_bias_reference, d_bias, bias_size, 
                        cudaMemcpyDeviceToHost), "Copy reference bias to host");
    
    // Now benchmark the selected kernel
    // Warm up
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        // Reset device weights and bias to initial values
        CUDA_CHECK(cudaMemcpy(d_weights, h_weights, weight_size, 
                          cudaMemcpyHostToDevice), "Copy weights to device for warmup");
        CUDA_CHECK(cudaMemcpy(d_bias, h_bias, bias_size, 
                          cudaMemcpyHostToDevice), "Copy bias to device for warmup");
        
        switch (kernel_type) {
            case 0: // Original
                update_input_hidden_weights_kernel<<<blocks, THREADS_PER_BLOCK, shared_mem_size>>>(
                    d_batch_input, d_batch_hidden_delta, d_weights, d_bias, 
                    INPUT_NODES, HIDDEN_NODES, batch_size, learning_rate
                );
                break;
            case 1: // Advanced
                update_input_hidden_weights_kernel_advanced<<<blocks, THREADS_PER_BLOCK, shared_mem_size>>>(
                    d_batch_input, d_batch_hidden_delta, d_weights, d_bias, 
                    INPUT_NODES, HIDDEN_NODES, batch_size, learning_rate
                );
                break;
            case 2: // Vectorized2
                update_input_hidden_weights_kernel_vectorized2<<<blocks, THREADS_PER_BLOCK, shared_mem_size>>>(
                    d_batch_input, d_batch_hidden_delta, d_weights, d_bias, 
                    INPUT_NODES, HIDDEN_NODES, batch_size, learning_rate
                );
                break;
            case 3: // Vectorized4
                update_input_hidden_weights_kernel_vectorized4<<<blocks, THREADS_PER_BLOCK, shared_mem_size>>>(
                    d_batch_input, d_batch_hidden_delta, d_weights, d_bias, 
                    INPUT_NODES, HIDDEN_NODES, batch_size, learning_rate
                );
                break;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize(), "Warmup synchronize");
    
    // Benchmark timing
    float times[NUM_ITERATIONS];
    float total_time = 0.0f;
    
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        // Reset device weights and bias to initial values
        CUDA_CHECK(cudaMemcpy(d_weights, h_weights, weight_size, 
                          cudaMemcpyHostToDevice), "Copy weights to device for benchmark");
        CUDA_CHECK(cudaMemcpy(d_bias, h_bias, bias_size, 
                          cudaMemcpyHostToDevice), "Copy bias to device for benchmark");
        
        float milliseconds = 0.0f;
        
        CUDA_CHECK(cudaEventRecord(start), "Record start event");
        
        switch (kernel_type) {
            case 0: // Original
                update_input_hidden_weights_kernel<<<blocks, THREADS_PER_BLOCK, shared_mem_size>>>(
                    d_batch_input, d_batch_hidden_delta, d_weights, d_bias, 
                    INPUT_NODES, HIDDEN_NODES, batch_size, learning_rate
                );
                break;
            case 1: // Advanced
                update_input_hidden_weights_kernel_advanced<<<blocks, THREADS_PER_BLOCK, shared_mem_size>>>(
                    d_batch_input, d_batch_hidden_delta, d_weights, d_bias, 
                    INPUT_NODES, HIDDEN_NODES, batch_size, learning_rate
                );
                break;
            case 2: // Vectorized2
                update_input_hidden_weights_kernel_vectorized2<<<blocks, THREADS_PER_BLOCK, shared_mem_size>>>(
                    d_batch_input, d_batch_hidden_delta, d_weights, d_bias, 
                    INPUT_NODES, HIDDEN_NODES, batch_size, learning_rate
                );
                break;
            case 3: // Vectorized4
                update_input_hidden_weights_kernel_vectorized4<<<blocks, THREADS_PER_BLOCK, shared_mem_size>>>(
                    d_batch_input, d_batch_hidden_delta, d_weights, d_bias, 
                    INPUT_NODES, HIDDEN_NODES, batch_size, learning_rate
                );
                break;
        }
        
        CUDA_CHECK(cudaEventRecord(stop), "Record stop event");
        CUDA_CHECK(cudaEventSynchronize(stop), "Synchronize stop event");
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop), "Calculate elapsed time");
        
        times[i] = milliseconds;
        total_time += milliseconds;
    }
    
    // Copy back final results for verification (from the last iteration)
    CUDA_CHECK(cudaMemcpy(h_weights, d_weights, weight_size, 
                       cudaMemcpyDeviceToHost), "Copy weights to host for verification");
    CUDA_CHECK(cudaMemcpy(h_bias, d_bias, bias_size, 
                       cudaMemcpyDeviceToHost), "Copy bias to host for verification");
    
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
    
    // Verify correctness
    bool is_correct = true;
    double max_diff_weights = 0.0;
    double max_diff_bias = 0.0;
    int max_diff_idx_weights = -1;
    int max_diff_idx_bias = -1;
    
    // Check weights
    for (int i = 0; i < INPUT_NODES * HIDDEN_NODES; i++) {
        double diff = fabs(h_weights[i] - h_weights_reference[i]);
        if (diff > 1e-6) {
            is_correct = false;
            if (diff > max_diff_weights) {
                max_diff_weights = diff;
                max_diff_idx_weights = i;
            }
        }
    }
    
    // Check bias
    for (int i = 0; i < HIDDEN_NODES; i++) {
        double diff = fabs(h_bias[i] - h_bias_reference[i]);
        if (diff > 1e-6) {
            is_correct = false;
            if (diff > max_diff_bias) {
                max_diff_bias = diff;
                max_diff_idx_bias = i;
            }
        }
    }
    
    // Print results
    printf("Results for %s kernel with batch size %d:\n", kernel_names[kernel_type], batch_size);
    printf("  Correctness: %s", is_correct ? "PASS\n" : "FAIL - ");
    
    if (!is_correct) {
        printf("Max weight diff: %e at index %d, Max bias diff: %e at index %d\n", 
               max_diff_weights, max_diff_idx_weights, max_diff_bias, max_diff_idx_bias);
        
        if (max_diff_idx_weights >= 0) {
            printf("    Weight values at failure point: ref=%f, test=%f\n", 
                   h_weights_reference[max_diff_idx_weights], h_weights[max_diff_idx_weights]);
        }
        
        if (max_diff_idx_bias >= 0) {
            printf("    Bias values at failure point: ref=%f, test=%f\n", 
                   h_bias_reference[max_diff_idx_bias], h_bias[max_diff_idx_bias]);
        }
    }
    
    printf("  Average time: %.4f ms\n", avg_time);
    printf("  Min time:     %.4f ms\n", min_time);
    printf("  Max time:     %.4f ms\n", max_time);
    printf("  Std dev:      %.4f ms\n", std_dev);
    
    // Calculate throughput
    double elements_processed = (batch_size * (INPUT_NODES * HIDDEN_NODES + HIDDEN_NODES));
    double gb_processed = elements_processed * sizeof(double) / 1e9;
    double elements_per_second = elements_processed / (avg_time / 1000.0);
    double gb_per_second = gb_processed / (avg_time / 1000.0);
    
    printf("  Performance metrics:\n");
    printf("    Elements/sec: %.2f million\n", elements_per_second / 1e6);
    printf("    Memory bandwidth: %.2f GB/s\n", gb_per_second);
    printf("    Batch size: %d\n", batch_size);
    
    // Free device memory
    cudaFree(d_batch_input);
    cudaFree(d_batch_hidden_delta);
    cudaFree(d_weights);
    cudaFree(d_bias);
    
    // Free host memory
    free(h_batch_input);
    free(h_batch_hidden_delta);
    free(h_weights);
    free(h_bias);
    free(h_weights_reference);
    free(h_bias_reference);
    
    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
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
            if (kernel >= 0 && kernel <= 3) {
                test_all_kernels = false;
                benchmark_batch_size(batch_size, kernel);
                i++;
            } else {
                printf("Invalid kernel type (0-3 allowed)\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  --batch-size N   Set batch size (default: 128)\n");
            printf("  --kernel K       Test only kernel K (0=original, 1=advanced, 2=vectorized2, 3=vectorized4)\n");
            printf("  --help           Display this help message\n");
            return 0;
        }
    }
    
    if (test_all_kernels) {
        // Benchmark with different batch sizes
        int batch_sizes[] = {32, 64, 128, 256, 512, 1024};
        int num_batch_sizes = sizeof(batch_sizes) / sizeof(batch_sizes[0]);
        
        for (int kernel_type = 0; kernel_type < 4; kernel_type++) {
            printf("\n===== BENCHMARKING KERNEL TYPE %d =====\n", kernel_type);
            for (int i = 0; i < num_batch_sizes; i++) {
                benchmark_batch_size(batch_sizes[i], kernel_type);
            }
        }
    }
    
    printf("\nBenchmarking complete.\n");
    
    return 0;
} 