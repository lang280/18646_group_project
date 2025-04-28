#ifndef BACKWARD_H
#define BACKWARD_H

#include <cuda_runtime.h>

// Backward pass kernels
__global__ void calculate_output_delta_kernel(
    const float* __restrict__ batch_output,
    const float* __restrict__ batch_targets,
    float* __restrict__ batch_output_delta,
    int output_size,
    int batch_size
);

__global__ void calculate_hidden_delta_kernel(
    const float* __restrict__ batch_hidden,
    const float* __restrict__ batch_output_delta,
    const float* __restrict__ weights,
    float* __restrict__ batch_hidden_delta,
    int hidden_size,
    int output_size,
    int batch_size
);

// Weight update kernels
__global__ void update_input_hidden_weights_kernel_vectorized4(
    const float* __restrict__ batch_input,
    const float* __restrict__ batch_hidden_delta,
    float* __restrict__ weights,
    float* __restrict__ bias,
    int input_size,
    int hidden_size,
    int batch_size,
    float learning_rate
);

__global__ void update_hidden_output_weights_kernel(
    const float* __restrict__ batch_hidden,
    const float* __restrict__ batch_output_delta,
    float* __restrict__ weights,
    float* __restrict__ bias,
    int hidden_size,
    int output_size,
    int batch_size,
    float learning_rate
);

#endif // BACKWARD_H 