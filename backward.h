#ifndef BACKWARD_H
#define BACKWARD_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Neural network dimensions - must match main.c
#define NEW_INPUT_SIZE 256
#define INPUT_NODES (NEW_INPUT_SIZE * NEW_INPUT_SIZE)
#define HIDDEN_NODES 256
#define OUTPUT_NODES 10

#ifdef __cplusplus
extern "C" {
#endif


/**
 * backward propagation with cuda
 * 
 * @param input values [INPUT_NODES]
 * @param target values [OUTPUT_NODES]
 * @param weight1 hidden weights [INPUT_NODES][HIDDEN_NODES]
 * @param weight2 output weights [HIDDEN_NODES][OUTPUT_NODES]
 * @param bias1 hidden bias [HIDDEN_NODES]
 * @param bias2 output bias [OUTPUT_NODES]
 * @param hidden hidden layer [HIDDEN_NODES]
 * @param output output layer [OUTPUT_NODES]
 * @param num_threads thread count
 * @param learning_rate learning rate
 */
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
);

#ifdef __cplusplus
}
#endif

#endif /* BACKWARD_H */
