/**
 * GraphSAGE Layer for FPGA Implementation - FIXED-POINT VERSION
 * C++ implementation with HLS pragmas
 * 
 * This version uses ap_fixed for configurable fixed-point arithmetic.
 * Template parameters allow experimenting with different Q formats.
 * 
 * Fully templated to support different model configurations and bit widths.
 */

#ifndef GRAPHSAGE_LAYER_FIXED_H
#define GRAPHSAGE_LAYER_FIXED_H

#include <ap_fixed.h>
#include <cmath>
#include <algorithm>

// ============================================================================
// CONFIGURABLE FIXED-POINT TYPE PARAMETERS
// ============================================================================

// Data type (features, activations)
// Format: ap_fixed<W, I> where W=total bits, I=integer bits, F=W-I fractional bits
#ifndef DATA_W
#define DATA_W 16  // Total width
#endif
#ifndef DATA_I
#define DATA_I 8   // Integer bits (rest are fractional)
#endif

// Weight type
#ifndef WEIGHT_W
#define WEIGHT_W 16
#endif
#ifndef WEIGHT_I
#define WEIGHT_I 4
#endif

// Accumulator type (needs more bits to avoid overflow)
#ifndef ACC_W
#define ACC_W 32
#endif
#ifndef ACC_I
#define ACC_I 16
#endif

// Scale type (adjacency matrix - can be float or fixed)
#ifndef SCALE_W
#define SCALE_W 16
#endif
#ifndef SCALE_I
#define SCALE_I 2
#endif

// Type definitions
typedef ap_fixed<DATA_W, DATA_I> data_t;
typedef ap_fixed<WEIGHT_W, WEIGHT_I> weight_t;
typedef ap_fixed<ACC_W, ACC_I> acc_t;
typedef ap_fixed<SCALE_W, SCALE_I> scale_t;

// ============================================================================
// MODEL CONFIGURATION
// ============================================================================

#ifndef NUM_NODES
#define NUM_NODES 8
#endif

#ifndef IN_FEATURES
#define IN_FEATURES 16
#endif

#ifndef HIDDEN_FEATURES
#define HIDDEN_FEATURES 24
#endif

#ifndef OUT_FEATURES
#define OUT_FEATURES 7
#endif

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * ReLU activation function (element-wise)
 */
template<int N_NODES, int N_FEATURES>
void relu_fixed(data_t data[N_NODES][N_FEATURES]) {
    #pragma HLS INLINE
    
    RELU_N: for (int n = 0; n < N_NODES; n++) {
        #pragma HLS UNROLL
        RELU_F: for (int f = 0; f < N_FEATURES; f++) {
            #pragma HLS UNROLL
            if (data[n][f] < 0) {
                data[n][f] = 0;
            }
        }
    }
}

// ============================================================================
// GRAPHSAGE LAYER OPERATIONS
// ============================================================================

/**
 * Aggregate neighbor features: agg = adj_matrix @ features
 * 
 * For each node, compute weighted sum of neighbor features.
 * Uses fixed-point arithmetic.
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
            
            acc_t sum = 0;
            
            AGG_J: for (int j = 0; j < N_NODES; j++) {
                #pragma HLS UNROLL
                if (adj_matrix[i][j] != 0) {
                    sum += adj_matrix[i][j] * features[j][f];
                }
            }
            
            // Cast accumulator back to data type
            agg_out[i][f] = (data_t)sum;
        }
    }
}

/**
 * Linear transformation: out = features * W^T + bias
 * 
 * Fixed-point matrix multiplication with bias addition.
 * Uses wider accumulator to prevent overflow.
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
            
            // Use wider accumulator for MAC operations
            acc_t acc = 0;
            
            // Matrix multiply: acc = sum(features[i] * weights[o][i])
            LIN_IN: for (int i = 0; i < IN_FEAT; i++) {
                #pragma HLS UNROLL
                acc += features[n][i] * weights[o][i];
            }
            
            // Add bias
            acc += bias[o];
            
            // Cast back to output data type
            output[n][o] = (data_t)acc;
        }
    }
}

// ============================================================================
// FULL GRAPHSAGE NETWORK
// ============================================================================

/**
 * Full GraphSAGE network with 2 layers.
 * 
 * Architecture:
 *   Input -> Aggregate -> Linear -> ReLU -> Aggregate -> Linear -> Output
 * 
 * Uses fixed-point arithmetic throughout.
 * 
 * Template parameters:
 *   N_NODES: Number of nodes in subgraph
 *   IN_FEAT: Input feature dimension
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
    aggregate_neighbors<N_NODES, IN_FEAT>(
        adj_matrix,
        input,
        agg1
    );

    // ========== Layer 1: Linear ==========
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
    relu_fixed<N_NODES, HIDDEN_FEAT>(hidden);

    // ========== Layer 2: Aggregate ==========
    aggregate_neighbors<N_NODES, HIDDEN_FEAT>(
        adj_matrix,
        hidden,
        agg2
    );

    // ========== Layer 2: Linear (no ReLU) ==========
    linear_transform<N_NODES, HIDDEN_FEAT, OUT_FEAT>(
        agg2,
        weights2,
        bias2,
        output
    );
}

#endif // GRAPHSAGE_LAYER_FIXED_H
