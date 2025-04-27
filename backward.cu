#include <cuda_runtime.h>
#include <stdio.h>
#include "backward.h"


__device__ inline double activation_derivative(double y) {
    return y * (1.0 - y);
}

__device__ inline double relu_derivative(double y) {
    return (y > 0.0) ? 1.0 : 0.0;
}

__global__ void backprop_update_kernel(const double *inputs,
                                       const double *hidden_out,
                                       const double *outputs,
                                       const double *targets,
                                       double *W_in,
                                       double *W_out,
                                       double *bias1,
                                       double *bias2,
                                       int batch_size,
                                       double learning_rate) 
{
    int idx = blockIdx.x;

    extern __shared__ double sh_hidden_delta[];

    if (idx < HIDDEN_NODES) {
        int j = idx; // hidden neuron index

        for (int s = threadIdx.x; s < batch_size; s += blockDim.x) {
            double sum_output_error = 0.0;
            for (int k = 0; k < OUTPUT_NODES; ++k) {
                double output_val = outputs[s * OUTPUT_NODES + k];
                double target_val = targets[s * OUTPUT_NODES + k];
                double output_delta = (output_val - target_val) * activation_derivative(output_val);
                sum_output_error += output_delta * W_out[j * OUTPUT_NODES + k];
            }
            double hidden_val = hidden_out[s * HIDDEN_NODES + j];
            sh_hidden_delta[s] = sum_output_error * relu_derivative(hidden_val);
        }
        __syncthreads();

        for (int i = threadIdx.x; i < INPUT_NODES; i += blockDim.x) {
            double grad_sum = 0.0;
            for (int s = 0; s < batch_size; ++s) {
                grad_sum += sh_hidden_delta[s] * inputs[s * INPUT_NODES + i];
            }
            W_in[i * HIDDEN_NODES + j] -= learning_rate * grad_sum;
        }

        __shared__ double bias1_grad_sum_sh;
        if (threadIdx.x == 0) bias1_grad_sum_sh = 0.0;
        __syncthreads();

        for (int s = threadIdx.x; s < batch_size; s += blockDim.x) {
            atomicAdd(&bias1_grad_sum_sh, sh_hidden_delta[s]);
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            bias1[j] -= learning_rate * bias1_grad_sum_sh;
        }

        for (int k = threadIdx.x; k < OUTPUT_NODES; k += blockDim.x) {
            double grad_sum_out = 0.0;
            for (int s = 0; s < batch_size; ++s) {
                double output_val = outputs[s * OUTPUT_NODES + k];
                double target_val = targets[s * OUTPUT_NODES + k];
                double output_delta = (output_val - target_val) * activation_derivative(output_val);
                double hidden_out_val = hidden_out[s * HIDDEN_NODES + j];
                grad_sum_out += output_delta * hidden_out_val;
            }
            W_out[j * OUTPUT_NODES + k] -= learning_rate * grad_sum_out;
        }
    }
    else if (idx < HIDDEN_NODES + OUTPUT_NODES) {
        int k = idx - HIDDEN_NODES;
        __shared__ double bias2_grad_sum_sh; // Use shared memory for reduction
        if (threadIdx.x == 0) bias2_grad_sum_sh = 0.0;
        __syncthreads();

        for (int s = threadIdx.x; s < batch_size; s += blockDim.x) {
            double output_val = outputs[s * OUTPUT_NODES + k];
            double target_val = targets[s * OUTPUT_NODES + k];
            double output_delta = (output_val - target_val) * activation_derivative(output_val);
            atomicAdd(&bias2_grad_sum_sh, output_delta); 
        }
        __syncthreads(); // Ensure all atomicAdds are done

        if (threadIdx.x == 0) {
           bias2[k] -= learning_rate * bias2_grad_sum_sh;
        }
    }
}

// Host function to launch backpropagation kernels
extern "C" 
void backward_propagate(
    const double input[INPUT_NODES],
    const double target[OUTPUT_NODES],
    double weight1[INPUT_NODES * HIDDEN_NODES],
    double weight2[HIDDEN_NODES * OUTPUT_NODES],
    double bias1[HIDDEN_NODES],
    double bias2[OUTPUT_NODES],
    double hidden[HIDDEN_NODES],
    double output[OUTPUT_NODES],
    int num_threads,
    double learning_rate
    ) {
    // Define problem sizes
    double *d_input, *d_hidden, *d_output, *d_target;
    double *d_weight1, *d_weight2;
    double *d_bias1, *d_bias2;

    // Allocate GPU memory
    cudaMalloc(&d_input, INPUT_NODES * sizeof(double));
    cudaMalloc(&d_target, OUTPUT_NODES * sizeof(double));
    cudaMalloc(&d_weight1, INPUT_NODES * HIDDEN_NODES * sizeof(double));
    cudaMalloc(&d_bias1, HIDDEN_NODES * sizeof(double));
    cudaMalloc(&d_hidden, HIDDEN_NODES * sizeof(double));
    cudaMalloc(&d_weight2, HIDDEN_NODES * OUTPUT_NODES * sizeof(double));
    cudaMalloc(&d_bias2, OUTPUT_NODES * sizeof(double));
    cudaMalloc(&d_output, OUTPUT_NODES * sizeof(double));
    
    // Copy data from host to device
    cudaMemcpy(d_input, input, INPUT_NODES * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target, OUTPUT_NODES * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight1, weight1, INPUT_NODES * HIDDEN_NODES * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias1, bias1, HIDDEN_NODES * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight2, weight2, HIDDEN_NODES * OUTPUT_NODES * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias2, bias2, OUTPUT_NODES * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hidden, hidden, HIDDEN_NODES * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output, OUTPUT_NODES * sizeof(double), cudaMemcpyHostToDevice);

    // Launch the kernel
    int threads_per_block = 256;
    int blocks_per_grid = HIDDEN_NODES + OUTPUT_NODES; // one block per hidden neuron
    size_t shared_mem_size = num_threads * sizeof(double); // for sh_hidden_delta

    backprop_update_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
        d_input, d_hidden, d_output, d_target, 
        d_weight1, d_weight2, d_bias1, d_bias2, 
        num_threads, learning_rate
    );

    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy((void*)weight1, d_weight1, INPUT_NODES * HIDDEN_NODES * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)weight2, d_weight2, HIDDEN_NODES * OUTPUT_NODES * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)bias1, d_bias1, HIDDEN_NODES * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)bias2, d_bias2, OUTPUT_NODES * sizeof(double), cudaMemcpyDeviceToHost);


    // Free device memory
    cudaFree(d_input);
    cudaFree(d_target);
    cudaFree(d_weight1);
    cudaFree(d_bias1);
    cudaFree(d_hidden);
    cudaFree(d_weight2);
    cudaFree(d_bias2);
}
