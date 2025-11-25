/**
 * GraphSAGE Layer for FPGA Implementation - FLOAT VERSION (Templated)
 * Header file with type definitions and function declarations
 * 
 * This version uses FLOAT arithmetic to match PyG exactly.
 * No quantization errors - used for validating graph structure and SAGEConv logic.
 * Fully templated to support different model configurations.
 */

#ifndef GRAPHSAGE_LAYER_FLOAT_H
#define GRAPHSAGE_LAYER_FLOAT_H

#include <stdint.h>

// ============================================================================
// Configuration Parameters (Compile-time configurable)
// ============================================================================

// Default configuration for reduced model (matches model_config.yaml)
#ifndef NUM_NODES
#define NUM_NODES 8  // Subgraph size for HLS testing
#endif

#ifndef IN_FEATURES
#define IN_FEATURES 16  // reduced_model.in_channels_reduced
#endif

#ifndef HIDDEN_FEATURES
#define HIDDEN_FEATURES 24  // reduced_model.hidden_channels
#endif

#ifndef OUT_FEATURES
#define OUT_FEATURES 7  // reduced_model.out_channels (num_classes)
#endif

// Legacy compatibility defines
#define MAX_NODES NUM_NODES
#define MAX_FEATURES_IN IN_FEATURES
#define MAX_FEATURES_HIDDEN HIDDEN_FEATURES
#define MAX_FEATURES_OUT OUT_FEATURES

// ============================================================================
// Type Definitions - FLOAT VERSION
// ============================================================================

typedef float data_t;        // Activations (FLOAT)
typedef float weight_t;      // Weights (FLOAT)
typedef float acc_t;         // Accumulator (FLOAT)
typedef float scale_t;       // Adjacency matrix values

// ============================================================================
// Templated Function Implementations (must be in header for templates)
// ============================================================================

/**
 * Aggregation: mean aggregation using adjacency matrix
 * agg[i] = sum_j (adj[i,j] * features[j])
 * Adjacency matrix is already row-normalized for mean aggregation
 * 
 * Template parameters:
 *   N_NODES: Number of nodes in the graph
 *   N_FEATURES: Number of features per node
 */
template<int N_NODES, int N_FEATURES>
void aggregate_neighbors(
    const scale_t adj_matrix[N_NODES][N_NODES],
    const data_t features[N_NODES][N_FEATURES],
    data_t agg_out[N_NODES][N_FEATURES]
) {
    #pragma HLS INLINE

    AGG_I: for (int i = 0; i < N_NODES; i++) {
        #pragma HLS UNROLL
        AGG_F: for (int f = 0; f < N_FEATURES; f++) {
            #pragma HLS UNROLL
            
            acc_t sum = 0.0f;
            
            AGG_J: for (int j = 0; j < N_NODES; j++) {
                #pragma HLS UNROLL
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
 * W is [OUT_FEATURES x IN_FEATURES]
 * features is [N_NODES x IN_FEATURES]
 * out is [N_NODES x OUT_FEATURES]
 * 
 * Template parameters:
 *   N_NODES: Number of nodes
 *   IN_FEAT: Input feature dimension
 *   OUT_FEAT: Output feature dimension
 */
template<int N_NODES, int IN_FEAT, int OUT_FEAT>
void linear_transform(
    const data_t features[N_NODES][IN_FEAT],
    const weight_t weights[OUT_FEAT][IN_FEAT],
    const weight_t bias[OUT_FEAT],
    data_t output[N_NODES][OUT_FEAT]
) {
    #pragma HLS INLINE
    LIN_N: for (int n = 0; n < N_NODES; n++) {
        #pragma HLS UNROLL
        LIN_OUT: for (int o = 0; o < OUT_FEAT; o++) {
            #pragma HLS UNROLL
            
            acc_t acc = bias[o];
            
            LIN_IN: for (int i = 0; i < IN_FEAT; i++) {
                #pragma HLS UNROLL
                acc += features[n][i] * weights[o][i];
            }
            
            output[n][o] = acc;
        }
    }
}

/**
 * ReLU activation function (element-wise)
 * 
 * Template parameters:
 *   N_NODES: Number of nodes
 *   N_FEATURES: Number of features per node
 */
template<int N_NODES, int N_FEATURES>
void relu_float(
    data_t data[N_NODES][N_FEATURES]
) {
    #pragma HLS INLINE

    RELU_N: for (int n = 0; n < N_NODES; n++) {
        #pragma HLS UNROLL
        RELU_F: for (int f = 0; f < N_FEATURES; f++) {
            #pragma HLS UNROLL
            if (data[n][f] < 0.0f) {
                data[n][f] = 0.0f;
            }
        }
    }
}

/**
 * Two-layer GraphSAGE network - FLOAT VERSION (Templated)
 * Top-level function for HLS synthesis
 * 
 * Implements:
 *   h_agg1 = aggregate(input)          # mean aggregation
 *   h1 = ReLU(W1 @ h_agg1 + b1)       # linear + activation
 *   h_agg2 = aggregate(h1)             # mean aggregation  
 *   out = W2 @ h_agg2 + b2             # final linear (no activation)
 * 
 * This matches PyG SAGEConv with root_weight=False, normalize=False
 * 
 * Template parameters:
 *   N_NODES: Number of nodes in subgraph
 *   IN_FEAT: Input feature dimension (after projection)
 *   HIDDEN_FEAT: Hidden layer dimension
 *   OUT_FEAT: Output dimension (num_classes)
 */
template<int N_NODES, int IN_FEAT, int HIDDEN_FEAT, int OUT_FEAT>
void graphsage_network_template(
    const scale_t adj_matrix[N_NODES][N_NODES],
    const data_t input[N_NODES][IN_FEAT],
    const weight_t weights1[HIDDEN_FEAT][IN_FEAT],
    const weight_t bias1[HIDDEN_FEAT],
    const weight_t weights2[OUT_FEAT][HIDDEN_FEAT],
    const weight_t bias2[OUT_FEAT],
    data_t output[N_NODES][OUT_FEAT]
) {
    #pragma HLS PIPELINE II=1
    
    // Intermediate buffers - fully partitioned for parallel access
    data_t agg1[N_NODES][IN_FEAT];                 // Aggregated input
    data_t linear1_out[N_NODES][HIDDEN_FEAT];      // Layer 1 linear output (before ReLU)
    data_t hidden[N_NODES][HIDDEN_FEAT];           // Layer 1 output (after ReLU)
    data_t agg2[N_NODES][HIDDEN_FEAT];             // Aggregated hidden

    // Complete array partitioning for fully parallel implementation
    #pragma HLS ARRAY_PARTITION variable=agg1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=agg1 complete dim=2
    #pragma HLS ARRAY_PARTITION variable=linear1_out complete dim=1
    #pragma HLS ARRAY_PARTITION variable=linear1_out complete dim=2
    #pragma HLS ARRAY_PARTITION variable=hidden complete dim=1
    #pragma HLS ARRAY_PARTITION variable=hidden complete dim=2
    #pragma HLS ARRAY_PARTITION variable=agg2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=agg2 complete dim=2

    // ========== Layer 1: Aggregate ==========
    // Compute: agg1 = adj_matrix @ input
    // This is mean aggregation because adj_matrix is row-normalized
    aggregate_neighbors<N_NODES, IN_FEAT>(
        adj_matrix,
        input,
        agg1
    );

    // ========== Layer 1: Linear ==========
    // Compute: linear1_out = W1 @ agg1 + b1
    linear_transform<N_NODES, IN_FEAT, HIDDEN_FEAT>(
        agg1,
        weights1,
        bias1,
        linear1_out
    );
    
    // ========== Layer 1: ReLU ==========
    // Copy to hidden buffer and apply ReLU
    COPY_RELU_N: for (int n = 0; n < N_NODES; n++) {
        #pragma HLS UNROLL
        COPY_RELU_F: for (int f = 0; f < HIDDEN_FEAT; f++) {
            #pragma HLS UNROLL
            hidden[n][f] = linear1_out[n][f];
        }
    }
    relu_float<N_NODES, HIDDEN_FEAT>(hidden);

    // ========== Layer 2: Aggregate ==========
    // Compute: agg2 = adj_matrix @ hidden
    aggregate_neighbors<N_NODES, HIDDEN_FEAT>(
        adj_matrix,
        hidden,
        agg2
    );

    // ========== Layer 2: Linear (no ReLU) ==========
    // Compute: output = W2 @ agg2 + b2
    linear_transform<N_NODES, HIDDEN_FEAT, OUT_FEAT>(
        agg2,
        weights2,
        bias2,
        output
    );
}

/**
 * Non-templated wrapper for HLS top function (uses default config)
 * Declaration only - implementation in .cpp file
 */
void graphsage_network(
    const scale_t adj_matrix[NUM_NODES][NUM_NODES],
    const data_t input[NUM_NODES][IN_FEATURES],
    const weight_t weights1[HIDDEN_FEATURES][IN_FEATURES],
    const weight_t bias1[HIDDEN_FEATURES],
    const weight_t weights2[OUT_FEATURES][HIDDEN_FEATURES],
    const weight_t bias2[OUT_FEATURES],
    data_t output[NUM_NODES][OUT_FEATURES]
);

#endif // GRAPHSAGE_LAYER_FLOAT_H



