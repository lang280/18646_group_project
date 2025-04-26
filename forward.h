#ifndef FORWARD_H
#define FORWARD_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Neural network dimensions - must match main.c
#define INPUT_NODES 784
#define HIDDEN_NODES 256
#define OUTPUT_NODES 10

#ifdef __cplusplus
extern "C" {
#endif

// Activation functions
double relu(double x);
double sigmoid(double x);

/**
 * forward propagation with cuda
 * 
 * @param input values [INPUT_NODES]
 * @param weight1 hidden weights [INPUT_NODES][HIDDEN_NODES]
 * @param weight2 output weights [HIDDEN_NODES][OUTPUT_NODES]
 * @param bias1 hidden bias [HIDDEN_NODES]
 * @param bias2 output bias [OUTPUT_NODES]
 * @param hidden hidden layer [HIDDEN_NODES]
 * @param output output layer [OUTPUT_NODES]
 * @param num_threads thread count
 */
void forward_propagate(
    const double input[INPUT_NODES],
    const double weight1[INPUT_NODES][HIDDEN_NODES],
    const double weight2[HIDDEN_NODES][OUTPUT_NODES],
    const double bias1[HIDDEN_NODES],
    const double bias2[OUTPUT_NODES],
    double hidden[HIDDEN_NODES],
    double output[OUTPUT_NODES],
    int num_threads
);

#ifdef __cplusplus
}
#endif

#endif /* FORWARD_H */
