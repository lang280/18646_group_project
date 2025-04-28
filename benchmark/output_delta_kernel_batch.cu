#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_profiler_api.h>

// Constants for network dimensions
#define OUTPUT_NODES 10
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

// Device activation functions
__device__ inline double sigmoid_derivative(double y) {
    return y * (1.0 - y);
}

// Original kernel to calculate output layer deltas
__global__ void calculate_output_delta_kernel(
    const double* __restrict__ batch_output,
    const double* __restrict__ batch_targets,
    double* __restrict__ batch_output_delta,
    int output_size,
    int batch_size
) {
    extern __shared__ double shared_mem[];
    double* s_output = shared_mem;
    double* s_targets = shared_mem + blockDim.x;
    
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
        double output_val = s_output[tid];
        double target_val = s_targets[tid];
        
        batch_output_delta[offset] = (output_val - target_val) * sigmoid_derivative(output_val);
    }
}

// Optimized kernel with loop unrolling for processing multiple outputs per thread
__global__ void calculate_output_delta_kernel_optimized(
    const double* __restrict__ batch_output,
    const double* __restrict__ batch_targets,
    double* __restrict__ batch_output_delta,
    int output_size,
    int batch_size
) {
    extern __shared__ double shared_mem[];
    double* s_output = shared_mem;
    double* s_targets = shared_mem + blockDim.x;
    
    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + tid;
    int total_elements = output_size * batch_size;
    
    // Calculate multiple elements per thread using loop unrolling
    // Process 4 elements per thread where possible
    #pragma unroll 4
    for (int i = global_tid; i < (total_elements / 4) * 4; i += blockDim.x * gridDim.x * 4) {
        // Process 4 elements at once
        for (int j = 0; j < 4; j++) {
            int idx = i + j * blockDim.x * gridDim.x;
            if (idx < total_elements) {
                double output_val = batch_output[idx];
                double target_val = batch_targets[idx];
                
                batch_output_delta[idx] = (output_val - target_val) * sigmoid_derivative(output_val);
            }
        }
    }
    
    // Handle remaining elements
    for (int idx = global_tid + (total_elements / 4) * 4; 
         idx < total_elements; 
         idx += blockDim.x * gridDim.x) {
        double output_val = batch_output[idx];
        double target_val = batch_targets[idx];
        
        batch_output_delta[idx] = (output_val - target_val) * sigmoid_derivative(output_val);
    }
}

// Advanced optimized kernel with register caching and coalesced memory access
__global__ void calculate_output_delta_kernel_advanced(
    const double* __restrict__ batch_output,
    const double* __restrict__ batch_targets,
    double* __restrict__ batch_output_delta,
    int output_size,
    int batch_size
) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_tid = bid * blockDim.x + tid;
    
    // Each thread processes a chunk of the batch
    int chunk_size = (output_size * batch_size + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x);
    int start_idx = global_tid * chunk_size;
    int end_idx = min(start_idx + chunk_size, output_size * batch_size);
    
    // Process elements in chunks of 8 with coalesced memory access
    int i = start_idx;
    
    // Main processing loop with 8-way unrolling
    #pragma unroll 8
    for (; i + 7 < end_idx; i += 8) {
        // Prefetch outputs and targets into registers
        double out0 = batch_output[i];
        double out1 = batch_output[i+1];
        double out2 = batch_output[i+2];
        double out3 = batch_output[i+3];
        double out4 = batch_output[i+4];
        double out5 = batch_output[i+5];
        double out6 = batch_output[i+6];
        double out7 = batch_output[i+7];
        
        double tgt0 = batch_targets[i];
        double tgt1 = batch_targets[i+1];
        double tgt2 = batch_targets[i+2];
        double tgt3 = batch_targets[i+3];
        double tgt4 = batch_targets[i+4];
        double tgt5 = batch_targets[i+5];
        double tgt6 = batch_targets[i+6];
        double tgt7 = batch_targets[i+7];
        
        // Compute deltas
        batch_output_delta[i]   = (out0 - tgt0) * sigmoid_derivative(out0);
        batch_output_delta[i+1] = (out1 - tgt1) * sigmoid_derivative(out1);
        batch_output_delta[i+2] = (out2 - tgt2) * sigmoid_derivative(out2);
        batch_output_delta[i+3] = (out3 - tgt3) * sigmoid_derivative(out3);
        batch_output_delta[i+4] = (out4 - tgt4) * sigmoid_derivative(out4);
        batch_output_delta[i+5] = (out5 - tgt5) * sigmoid_derivative(out5);
        batch_output_delta[i+6] = (out6 - tgt6) * sigmoid_derivative(out6);
        batch_output_delta[i+7] = (out7 - tgt7) * sigmoid_derivative(out7);
    }
    
    // Handle remaining elements with 4-way unrolling
    #pragma unroll 4
    for (; i + 3 < end_idx; i += 4) {
        double out0 = batch_output[i];
        double out1 = batch_output[i+1];
        double out2 = batch_output[i+2];
        double out3 = batch_output[i+3];
        
        double tgt0 = batch_targets[i];
        double tgt1 = batch_targets[i+1];
        double tgt2 = batch_targets[i+2];
        double tgt3 = batch_targets[i+3];
        
        batch_output_delta[i]   = (out0 - tgt0) * sigmoid_derivative(out0);
        batch_output_delta[i+1] = (out1 - tgt1) * sigmoid_derivative(out1);
        batch_output_delta[i+2] = (out2 - tgt2) * sigmoid_derivative(out2);
        batch_output_delta[i+3] = (out3 - tgt3) * sigmoid_derivative(out3);
    }
    
    // Handle final elements
    for (; i < end_idx; i++) {
        double output_val = batch_output[i];
        double target_val = batch_targets[i];
        
        batch_output_delta[i] = (output_val - target_val) * sigmoid_derivative(output_val);
    }
}

// Fill an array with random values between 0 and 1 (for outputs) or binary values (for targets)
void fill_random(double* array, size_t size, bool is_target) {
    for (size_t i = 0; i < size; i++) {
        if (is_target) {
            // Target arrays have binary values
            array[i] = (rand() % 2 == 0) ? 0.0 : 1.0;
        } else {
            // Output arrays have values between 0 and 1 (simulating sigmoid outputs)
            array[i] = (double)rand() / RAND_MAX;
        }
    }
}

// Run benchmark for a specific batch size and kernel type
void benchmark_batch_size(int batch_size, int kernel_type) {
    const char* kernel_names[] = {"original", "optimized", "advanced"};
    printf("\nBenchmarking %s kernel with batch size %d...\n", 
           kernel_names[kernel_type], batch_size);
    
    // Allocate host memory
    size_t batch_output_size = batch_size * OUTPUT_NODES * sizeof(double);
    
    double* h_batch_output = (double*)malloc(batch_output_size);
    double* h_batch_targets = (double*)malloc(batch_output_size);
    double* h_batch_output_delta = (double*)malloc(batch_output_size);
    double* h_batch_output_delta_ref = (double*)malloc(batch_output_size);
    
    // Fill with random data
    fill_random(h_batch_output, batch_size * OUTPUT_NODES, false);
    fill_random(h_batch_targets, batch_size * OUTPUT_NODES, true);
    
    // Allocate device memory
    double *d_batch_output, *d_batch_targets, *d_batch_output_delta;
    CUDA_CHECK(cudaMalloc(&d_batch_output, batch_output_size), "Allocate device output");
    CUDA_CHECK(cudaMalloc(&d_batch_targets, batch_output_size), "Allocate device targets");
    CUDA_CHECK(cudaMalloc(&d_batch_output_delta, batch_output_size), "Allocate device output delta");
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_batch_output, h_batch_output, batch_output_size, 
                        cudaMemcpyHostToDevice), "Copy output to device");
    CUDA_CHECK(cudaMemcpy(d_batch_targets, h_batch_targets, batch_output_size, 
                        cudaMemcpyHostToDevice), "Copy targets to device");
    
    // Calculate grid dimensions based on kernel type
    int total_outputs = batch_size * OUTPUT_NODES;
    int blocks, threads_per_block;
    size_t shared_mem_size;
    
    switch (kernel_type) {
        case 0: // Original
            threads_per_block = THREADS_PER_BLOCK;
            blocks = (total_outputs + threads_per_block - 1) / threads_per_block;
            blocks = blocks > MAX_BLOCKS ? MAX_BLOCKS : blocks;
            shared_mem_size = 2 * threads_per_block * sizeof(double);
            break;
            
        case 1: // Optimized
            threads_per_block = THREADS_PER_BLOCK;
            // Fewer blocks needed since each thread processes multiple outputs
            blocks = min(32, (total_outputs + 4 * threads_per_block - 1) / (4 * threads_per_block)); 
            shared_mem_size = 0; // No shared memory used
            break;
            
        case 2: // Advanced
            threads_per_block = THREADS_PER_BLOCK;
            // Even fewer blocks for the advanced version
            blocks = min(32, (total_outputs + 8 * threads_per_block - 1) / (8 * threads_per_block));
            shared_mem_size = 0; // No shared memory used
            break;
            
        default:
            printf("Invalid kernel type: %d\n", kernel_type);
            exit(EXIT_FAILURE);
    }
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start), "Create start event");
    CUDA_CHECK(cudaEventCreate(&stop), "Create stop event");
    
    // Setup grid dimensions
    dim3 grid(blocks);
    dim3 block(threads_per_block);
    
    // Run reference version to verify correctness
    calculate_output_delta_kernel<<<grid, block, 2 * threads_per_block * sizeof(double)>>>(
        d_batch_output, d_batch_targets, d_batch_output_delta, 
        OUTPUT_NODES, batch_size
    );
    CUDA_CHECK(cudaGetLastError(), "Reference kernel execution");
    CUDA_CHECK(cudaDeviceSynchronize(), "Synchronize after reference kernel");
    
    // Copy reference results
    CUDA_CHECK(cudaMemcpy(h_batch_output_delta_ref, d_batch_output_delta, batch_output_size, 
                          cudaMemcpyDeviceToHost), "Copy reference results");
    
    // Warm up
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        switch (kernel_type) {
            case 0: // Original
                calculate_output_delta_kernel<<<grid, block, shared_mem_size>>>(
                    d_batch_output, d_batch_targets, d_batch_output_delta, 
                    OUTPUT_NODES, batch_size
                );
                break;
            case 1: // Optimized
                calculate_output_delta_kernel_optimized<<<grid, block, shared_mem_size>>>(
                    d_batch_output, d_batch_targets, d_batch_output_delta, 
                    OUTPUT_NODES, batch_size
                );
                break;
            case 2: // Advanced
                calculate_output_delta_kernel_advanced<<<grid, block, shared_mem_size>>>(
                    d_batch_output, d_batch_targets, d_batch_output_delta, 
                    OUTPUT_NODES, batch_size
                );
                break;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize(), "Warmup synchronize");
    
    // Benchmark timing
    float times[NUM_ITERATIONS];
    float total_time = 0.0f;
    
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        float milliseconds = 0.0f;
        
        CUDA_CHECK(cudaEventRecord(start), "Record start event");
        
        switch (kernel_type) {
            case 0: // Original
                calculate_output_delta_kernel<<<grid, block, shared_mem_size>>>(
                    d_batch_output, d_batch_targets, d_batch_output_delta, 
                    OUTPUT_NODES, batch_size
                );
                break;
            case 1: // Optimized
                calculate_output_delta_kernel_optimized<<<grid, block, shared_mem_size>>>(
                    d_batch_output, d_batch_targets, d_batch_output_delta, 
                    OUTPUT_NODES, batch_size
                );
                break;
            case 2: // Advanced
                calculate_output_delta_kernel_advanced<<<grid, block, shared_mem_size>>>(
                    d_batch_output, d_batch_targets, d_batch_output_delta, 
                    OUTPUT_NODES, batch_size
                );
                break;
        }
        
        CUDA_CHECK(cudaEventRecord(stop), "Record stop event");
        CUDA_CHECK(cudaEventSynchronize(stop), "Synchronize stop event");
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop), "Calculate elapsed time");
        
        times[i] = milliseconds;
        total_time += milliseconds;
    }
    
    // Copy results for verification (only on the last iteration)
    CUDA_CHECK(cudaMemcpy(h_batch_output_delta, d_batch_output_delta, batch_output_size, 
                          cudaMemcpyDeviceToHost), "Copy results");
    
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
    double max_diff = 0.0;
    int max_diff_idx = -1;
    
    for (int i = 0; i < batch_size * OUTPUT_NODES; i++) {
        double diff = fabs(h_batch_output_delta[i] - h_batch_output_delta_ref[i]);
        if (diff > 1e-6) {
            is_correct = false;
            if (diff > max_diff) {
                max_diff = diff;
                max_diff_idx = i;
            }
        }
    }
    
    // Print results
    printf("Results for %s kernel with batch size %d:\n", kernel_names[kernel_type], batch_size);
    printf("  Correctness: %s", is_correct ? "PASS\n" : "FAIL - ");
    
    if (!is_correct) {
        printf("max diff: %e at index %d\n", max_diff, max_diff_idx);
        printf("    Values at failure point: ref=%f, test=%f\n", 
               h_batch_output_delta_ref[max_diff_idx], h_batch_output_delta[max_diff_idx]);
    }
    
    printf("  Average time: %.4f ms\n", avg_time);
    printf("  Min time:     %.4f ms\n", min_time);
    printf("  Max time:     %.4f ms\n", max_time);
    printf("  Std dev:      %.4f ms\n", std_dev);
    
    // Calculate throughput
    double elements_per_second = (batch_size * OUTPUT_NODES) / (avg_time * 1e-3);
    double gb_per_second = (3 * batch_size * OUTPUT_NODES * sizeof(double)) / (avg_time * 1e-3 * 1e9);
    printf("  Performance metrics:\n");
    printf("    Elements/sec: %.2f million\n", elements_per_second / 1e6);
    printf("    Memory bandwidth: %.2f GB/s\n", gb_per_second);
    
    // Free device memory
    cudaFree(d_batch_output);
    cudaFree(d_batch_targets);
    cudaFree(d_batch_output_delta);
    
    // Free host memory
    free(h_batch_output);
    free(h_batch_targets);
    free(h_batch_output_delta);
    free(h_batch_output_delta_ref);
    
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
            if (kernel >= 0 && kernel <= 2) {
                test_all_kernels = false;
                benchmark_batch_size(batch_size, kernel);
                i++;
            } else {
                printf("Invalid kernel type (0-2 allowed)\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  --batch-size N   Set batch size (default: 128)\n");
            printf("  --kernel K       Test only kernel K (0=original, 1=optimized, 2=advanced)\n");
            printf("  --help           Display this help message\n");
            return 0;
        }
    }
    
    if (test_all_kernels) {
        // Benchmark with different batch sizes
        int batch_sizes[] = {32, 64, 128, 256, 512, 1024};
        int num_batch_sizes = sizeof(batch_sizes) / sizeof(batch_sizes[0]);
        
        for (int kernel_type = 0; kernel_type < 3; kernel_type++) {
            printf("\n===== BENCHMARKING KERNEL TYPE %d =====\n", kernel_type);
            for (int i = 0; i < num_batch_sizes; i++) {
                benchmark_batch_size(batch_sizes[i], kernel_type);
            }
        }
    }
    
    printf("\nBenchmarking complete.\n");
    
    return 0;
} 