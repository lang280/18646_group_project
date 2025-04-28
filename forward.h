#ifndef FORWARD_H
#define FORWARD_H

#include <cuda_runtime.h>

// Forward pass kernels
__global__ void hidden_layer_kernel_batch_tiled(
    const float* __restrict__ batch_input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ batch_hidden,
    int input_size,
    int hidden_size,
    int batch_size
);

__global__ void hidden_layer_kernel_batch_vectorized4(
    const float* __restrict__ batch_input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ batch_hidden,
    int input_size,
    int hidden_size,
    int batch_size
);

__global__ void output_layer_kernel_batch(
    const float* __restrict__ batch_hidden,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ batch_output,
    int hidden_size,
    int output_size,
    int batch_size
);

__global__ void output_layer_kernel_batch_advanced(
    const float* __restrict__ batch_hidden,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ batch_output,
    int hidden_size,
    int output_size,
    int batch_size
);

__global__ void count_correct_predictions_kernel(
    const float* __restrict__ batch_output,
    const float* __restrict__ batch_targets,
    int* __restrict__ correct_count,
    int output_size,
    int batch_size
);

#endif // FORWARD_H 