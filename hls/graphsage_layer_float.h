/**
 * GraphSAGE Layer for FPGA Implementation - PHASE 1: FLOAT VERSION
 * Header file with type definitions and function declarations
 * 
 * This version uses FLOAT arithmetic to match PyG exactly.
 * No quantization errors - used for validating graph structure and SAGEConv logic.
 */

#ifndef GRAPHSAGE_LAYER_FLOAT_H
#define GRAPHSAGE_LAYER_FLOAT_H

#include <stdint.h>

// ============================================================================
// Configuration Parameters
// ============================================================================

// Maximum dimensions (from trained model)
#define MAX_NODES 32
#define MAX_FEATURES_IN 16
#define MAX_FEATURES_HIDDEN 24
#define MAX_FEATURES_OUT 7  // Number of output classes

// ============================================================================
// Type Definitions - PHASE 1: FLOAT
// ============================================================================

typedef float data_t;        // Activations (FLOAT for Phase 1)
typedef float weight_t;      // Weights (FLOAT for Phase 1)
typedef float acc_t;         // Accumulator (FLOAT for Phase 1)
typedef float scale_t;       // Adjacency matrix values

// ============================================================================
// Function Declarations
// ============================================================================

/**
 * Aggregation: mean aggregation using adjacency matrix
 * agg[i] = sum_j (adj[i,j] * features[j])
 * Adjacency matrix is already row-normalized for mean aggregation
 */
void aggregate_neighbors(
    const scale_t adj_matrix[MAX_NODES][MAX_NODES],
    const data_t features[MAX_NODES][MAX_FEATURES_HIDDEN],
    data_t agg_out[MAX_NODES][MAX_FEATURES_HIDDEN],
    int num_nodes,
    int num_features
);

/**
 * Linear transformation: out = features * W^T + bias
 * W is [out_features x in_features]
 * features is [num_nodes x in_features]
 * out is [num_nodes x out_features]
 */
void linear_transform(
    const data_t features[MAX_NODES][MAX_FEATURES_HIDDEN],
    const weight_t weights[MAX_FEATURES_HIDDEN][MAX_FEATURES_HIDDEN],
    const weight_t bias[MAX_FEATURES_HIDDEN],
    data_t output[MAX_NODES][MAX_FEATURES_HIDDEN],
    int num_nodes,
    int in_features,
    int out_features
);

/**
 * ReLU activation function (element-wise)
 */
void relu_float(
    data_t data[MAX_NODES][MAX_FEATURES_HIDDEN],
    int num_nodes,
    int num_features
);

/**
 * Two-layer GraphSAGE network - PHASE 1 FLOAT VERSION
 * Top-level function for HLS synthesis
 * 
 * Implements:
 *   h1 = ReLU(W1 @ mean_agg(input) + b1)
 *   out = W2 @ mean_agg(h1) + b2
 */
void graphsage_network(
    const scale_t adj_matrix[MAX_NODES][MAX_NODES],
    const data_t input[MAX_NODES][MAX_FEATURES_IN],

    // Layer 1 parameters
    const weight_t weights1[MAX_FEATURES_HIDDEN][MAX_FEATURES_IN],
    const weight_t bias1[MAX_FEATURES_HIDDEN],

    // Layer 2 parameters
    const weight_t weights2[MAX_FEATURES_OUT][MAX_FEATURES_HIDDEN],
    const weight_t bias2[MAX_FEATURES_OUT],

    data_t output[MAX_NODES][MAX_FEATURES_OUT],
    int num_nodes
);

#endif // GRAPHSAGE_LAYER_FLOAT_H
