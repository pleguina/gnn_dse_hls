/**
 * GraphSAGE Layer for FPGA Implementation
 * Header file with type definitions and function declarations
 */

#ifndef GRAPHSAGE_LAYER_H
#define GRAPHSAGE_LAYER_H

#include <stdint.h>
#include <ap_int.h>

// ============================================================================
// Configuration Parameters
// ============================================================================

// Maximum dimensions (from trained model)
#define MAX_NODES 32
#define MAX_FEATURES_IN 16
#define MAX_FEATURES_HIDDEN 24
#define MAX_FEATURES_OUT 7  // Number of output classes

// Quantization parameters
#define INT8_MIN -128
#define INT8_MAX 127
#define ACCUMULATOR_BITS 32

// ============================================================================
// Type Definitions
// ============================================================================

typedef int8_t data_t;       // Input/output/weight data type
typedef int32_t acc_t;       // Accumulator type
typedef float scale_t;       // Scale factor type

// ============================================================================
// Function Declarations
// ============================================================================

/**
 * Matrix multiplication: C = A * B
 * A: [M x K], B: [K x N], C: [M x N]
 */
void matmul_int8(
    const data_t A[MAX_NODES][MAX_FEATURES_IN],
    const data_t B[MAX_FEATURES_IN][MAX_FEATURES_OUT],
    acc_t C[MAX_NODES][MAX_FEATURES_OUT],
    int M, int K, int N
);

/**
 * Aggregation: Aggregate neighbor features using adjacency matrix
 * agg = adj_matrix * features
 * Uses MAX_FEATURES_HIDDEN for maximum flexibility
 */
void aggregate(
    const scale_t adj_matrix[MAX_NODES][MAX_NODES],
    const data_t features[MAX_NODES][MAX_FEATURES_HIDDEN],
    data_t agg_out[MAX_NODES][MAX_FEATURES_HIDDEN],
    int num_nodes,
    int num_features,
    scale_t scale_in,
    scale_t scale_out
);

/**
 * Linear transformation: out = in * W^T + bias
 * Uses MAX_FEATURES_HIDDEN for maximum buffer size
 */
void linear_transform(
    const data_t input[MAX_NODES][MAX_FEATURES_HIDDEN],
    const data_t weights[MAX_FEATURES_HIDDEN][MAX_FEATURES_HIDDEN],
    const acc_t bias[MAX_FEATURES_HIDDEN],
    data_t output[MAX_NODES][MAX_FEATURES_HIDDEN],
    int num_nodes,
    int in_features,
    int out_features,
    scale_t scale_in,
    scale_t scale_weight,
    scale_t scale_out
);

/**
 * ReLU activation function (INT8 quantized)
 * Uses MAX_FEATURES_HIDDEN for maximum buffer size
 */
void relu_int8(
    data_t data[MAX_NODES][MAX_FEATURES_HIDDEN],
    int num_nodes,
    int num_features
);

/**
 * Two-layer GraphSAGE network (simplified implementation)
 */
void graphsage_network(
    const scale_t adj_matrix[MAX_NODES][MAX_NODES],
    const data_t input[MAX_NODES][MAX_FEATURES_IN],

    // Layer 1 parameters
    const data_t weights1[MAX_FEATURES_HIDDEN][MAX_FEATURES_IN],
    const acc_t bias1[MAX_FEATURES_HIDDEN],
    scale_t scale_w1,

    // Layer 2 parameters
    const data_t weights2[MAX_FEATURES_OUT][MAX_FEATURES_HIDDEN],
    const acc_t bias2[MAX_FEATURES_OUT],
    scale_t scale_w2,

    data_t output[MAX_NODES][MAX_FEATURES_OUT],
    int num_nodes,
    scale_t scale_in,
    scale_t scale_hidden,
    scale_t scale_out
);

#endif // GRAPHSAGE_LAYER_H
