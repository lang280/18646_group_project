#ifndef FORWARD_H
#define FORWARD_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Neural network dimensions - must match main.c
#define OLD_INPUT_SIZE 28
#define NEW_INPUT_SIZE 256
#define INPUT_NODES (NEW_INPUT_SIZE * NEW_INPUT_SIZE)
#define HIDDEN_NODES 256
#define OUTPUT_NODES 10

#ifdef __cplusplus
extern "C" {
#endif

// Activation functions
double relu(double x);
double sigmoid(double x);

/**
 * Forward propagation with CUDA for 256x256 input images
 * 
 * @param input Flattened input values [INPUT_NODES]
 * @param weight1 Flattened hidden weights [INPUT_NODES * HIDDEN_NODES]
 * @param weight2 Flattened output weights [HIDDEN_NODES * OUTPUT_NODES]
 * @param bias1 Hidden bias [HIDDEN_NODES]
 * @param bias2 Output bias [OUTPUT_NODES]
 * @param hidden Hidden layer output [HIDDEN_NODES]
 * @param output Output layer [OUTPUT_NODES]
 * @param num_threads Thread count (currently unused)
 */
void forward_propagate(
    const double* input,
    const double* weight1,
    const double* weight2,
    const double* bias1,
    const double* bias2,
    double* hidden,
    double* output,
    int num_threads
);

#ifdef __cplusplus
}
#endif

#endif /* FORWARD_H */
