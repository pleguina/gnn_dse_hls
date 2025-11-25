/**
 * GraphSAGE Layer for FPGA Implementation - PHASE 1: FLOAT VERSION
 * C++ implementation with HLS pragmas
 * 
 * This version uses FLOAT arithmetic throughout to match PyG exactly.
 * Goal: Validate graph structure, aggregation, and SAGEConv logic.
 * No quantization errors.
 */

#include "graphsage_layer_float.h"
#include <cmath>

// ============================================================================
// Core Functions
// ============================================================================

/**
 * Aggregation: mean aggregation using adjacency matrix
 * Adjacency matrix is already row-normalized, so this computes mean aggregation
 * 
 * agg[i,f] = sum_j (adj[i,j] * features[j,f])
 */
void aggregate_neighbors(
    const scale_t adj_matrix[MAX_NODES][MAX_NODES],
    const data_t features[MAX_NODES][MAX_FEATURES_HIDDEN],
    data_t agg_out[MAX_NODES][MAX_FEATURES_HIDDEN],
    int num_nodes,
    int num_features
) {
    AGG_I: for (int i = 0; i < num_nodes; i++) {
        AGG_F: for (int f = 0; f < num_features; f++) {
            #pragma HLS PIPELINE II=1
            
            acc_t sum = 0.0f;
            
            AGG_J: for (int j = 0; j < num_nodes; j++) {
                #pragma HLS UNROLL factor=8
                if (adj_matrix[i][j] != 0.0f) {
                    sum += adj_matrix[i][j] * features[j][f];
                }
            }
            
            agg_out[i][f] = sum;
        }
    }
}

/**
 * Linear transformation: out = features * W^T + bias
 * 
 * features: [num_nodes x in_features]
 * weights: [out_features x in_features] (stored as W^T for efficiency)
 * bias: [out_features]
 * output: [num_nodes x out_features]
 */
void linear_transform(
    const data_t features[MAX_NODES][MAX_FEATURES_HIDDEN],
    const weight_t weights[MAX_FEATURES_HIDDEN][MAX_FEATURES_HIDDEN],
    const weight_t bias[MAX_FEATURES_HIDDEN],
    data_t output[MAX_NODES][MAX_FEATURES_HIDDEN],
    int num_nodes,
    int in_features,
    int out_features
) {
    LIN_N: for (int n = 0; n < num_nodes; n++) {
        LIN_OUT: for (int o = 0; o < out_features; o++) {
            #pragma HLS PIPELINE II=1
            
            acc_t acc = bias[o];
            
            LIN_IN: for (int i = 0; i < in_features; i++) {
                #pragma HLS UNROLL factor=8
                acc += features[n][i] * weights[o][i];
            }
            
            output[n][o] = acc;
        }
    }
}

/**
 * ReLU activation function (element-wise)
 */
void relu_float(
    data_t data[MAX_NODES][MAX_FEATURES_HIDDEN],
    int num_nodes,
    int num_features
) {
    RELU_N: for (int n = 0; n < num_nodes; n++) {
        RELU_F: for (int f = 0; f < num_features; f++) {
            #pragma HLS PIPELINE
            if (data[n][f] < 0.0f) {
                data[n][f] = 0.0f;
            }
        }
    }
}

/**
 * Two-layer GraphSAGE network - PHASE 1 FLOAT VERSION
 * Top-level function for HLS synthesis
 * 
 * Implements exactly:
 *   h_agg1 = aggregate(input)          # mean aggregation
 *   h1 = ReLU(W1 @ h_agg1 + b1)       # linear + activation
 *   h_agg2 = aggregate(h1)             # mean aggregation  
 *   out = W2 @ h_agg2 + b2             # final linear (no activation)
 * 
 * This matches PyG SAGEConv with root_weight=False, normalize=False
 */
void graphsage_network(
    const scale_t adj_matrix[MAX_NODES][MAX_NODES],
    const data_t input[MAX_NODES][MAX_FEATURES_IN],

    // Layer 1 parameters (conv1.lin_l)
    const weight_t weights1[MAX_FEATURES_HIDDEN][MAX_FEATURES_IN],
    const weight_t bias1[MAX_FEATURES_HIDDEN],

    // Layer 2 parameters (conv2.lin_l)
    const weight_t weights2[MAX_FEATURES_OUT][MAX_FEATURES_HIDDEN],
    const weight_t bias2[MAX_FEATURES_OUT],

    data_t output[MAX_NODES][MAX_FEATURES_OUT],
    int num_nodes
) {
    // HLS interface pragmas
    #pragma HLS INTERFACE mode=s_axilite port=return
    #pragma HLS INTERFACE mode=bram port=adj_matrix
    #pragma HLS INTERFACE mode=bram port=input
    #pragma HLS INTERFACE mode=bram port=weights1
    #pragma HLS INTERFACE mode=bram port=bias1
    #pragma HLS INTERFACE mode=bram port=weights2
    #pragma HLS INTERFACE mode=bram port=bias2
    #pragma HLS INTERFACE mode=bram port=output

    // Intermediate buffers
    static data_t agg1[MAX_NODES][MAX_FEATURES_IN];       // Aggregated input
    static data_t hidden[MAX_NODES][MAX_FEATURES_HIDDEN]; // Layer 1 output (after linear+ReLU)
    static data_t agg2[MAX_NODES][MAX_FEATURES_HIDDEN];   // Aggregated hidden

    // Array partitioning for better memory bandwidth
    #pragma HLS ARRAY_PARTITION variable=agg1 cyclic factor=4 dim=2
    #pragma HLS ARRAY_PARTITION variable=hidden cyclic factor=4 dim=2
    #pragma HLS ARRAY_PARTITION variable=agg2 cyclic factor=4 dim=2

    // ========== Layer 1: Aggregate ==========
    // Compute: agg1 = adj_matrix @ input
    // This is mean aggregation because adj_matrix is row-normalized
    aggregate_neighbors(
        adj_matrix,
        input,
        agg1,
        num_nodes,
        MAX_FEATURES_IN
    );

    // ========== Layer 1: Linear + ReLU ==========
    // Compute: hidden = ReLU(W1 @ agg1 + b1)
    linear_transform(
        agg1,
        weights1,
        bias1,
        hidden,
        num_nodes,
        MAX_FEATURES_IN,
        MAX_FEATURES_HIDDEN
    );
    
    relu_float(hidden, num_nodes, MAX_FEATURES_HIDDEN);

    // ========== Layer 2: Aggregate ==========
    // Compute: agg2 = adj_matrix @ hidden
    aggregate_neighbors(
        adj_matrix,
        hidden,
        agg2,
        num_nodes,
        MAX_FEATURES_HIDDEN
    );

    // ========== Layer 2: Linear (no ReLU) ==========
    // Compute: output = W2 @ agg2 + b2
    linear_transform(
        agg2,
        weights2,
        bias2,
        output,
        num_nodes,
        MAX_FEATURES_HIDDEN,
        MAX_FEATURES_OUT
    );
}
