/**
 * GraphSAGE Layer for FPGA Implementation - INT8 PTQ VERSION
 * Header file with type definitions and function declarations
 * 
 * This version uses INT8 quantized arithmetic for weights and activations.
 * Post-Training Quantization (PTQ) with symmetric quantization (zero_point = 0).
 * 
 * Quantization scheme:
 *   - Weights: INT8 symmetric quantization
 *   - Activations: INT8 symmetric quantization  
 *   - Accumulators: INT32 for intermediate sums
 *   - No zero-point (symmetric quantization)
 *   - Per-tensor quantization (single scale per layer)
 */

#ifndef GRAPHSAGE_LAYER_PTQ_H
#define GRAPHSAGE_LAYER_PTQ_H

#include <stdint.h>
#ifndef __SYNTHESIS__
#include <cstdio>
#endif

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

// ============================================================================
// Type Definitions - QUANTIZED VERSION
// ============================================================================

typedef int8_t data_t;        // Quantized activations (INT8)
typedef int8_t weight_t;      // Quantized weights (INT8)
typedef int32_t acc_t;        // Accumulator (INT32 for MAC results)
typedef float scale_t;        // Adjacency matrix values (still float for precision)
typedef float quant_scale_t;  // Quantization scales (float)

// ============================================================================
// Quantization Helper Functions
// ============================================================================

/**
 * Quantize floating-point value to INT8
 * q = round(x / scale)
 * Clamp to [-128, 127]
 */
inline int8_t quantize(float x, quant_scale_t scale) {
    #pragma HLS INLINE
    float q = x / scale;
    if (q > 127.0f) return 127;
    if (q < -128.0f) return -128;
    return (int8_t)(q + (q >= 0 ? 0.5f : -0.5f));  // Round to nearest
}

/**
 * Dequantize INT8 value to floating-point
 * x = q * scale
 */
inline float dequantize(int8_t q, quant_scale_t scale) {
    #pragma HLS INLINE
    return (float)q * scale;
}

/**
 * Requantize: convert from one quantization scale to another
 * Used after layer operations to rescale
 * out_q = round((in_q * in_scale) / out_scale)
 */
inline int8_t requantize(acc_t in_acc, quant_scale_t in_scale, quant_scale_t out_scale) {
    #pragma HLS INLINE
    // Compute effective scale factor
    float scale_factor = in_scale / out_scale;
    float result = (float)in_acc * scale_factor;
    
    // Clamp and round
    if (result > 127.0f) return 127;
    if (result < -128.0f) return -128;
    
    int8_t quantized = (int8_t)(result + (result >= 0 ? 0.5f : -0.5f));
    
    return quantized;
}

// ============================================================================
// Templated Function Implementations (must be in header for templates)
// ============================================================================

/**
 * Aggregation: mean aggregation using adjacency matrix (QUANTIZED)
 * agg[i] = sum_j (adj[i,j] * features[j])
 * 
 * Note: adj_matrix is still float for precision in graph operations
 * Features are INT8 quantized
 * 
 * Template parameters:
 *   N_NODES: Number of nodes in the graph
 *   N_FEATURES: Number of features per node
 */
template<int N_NODES, int N_FEATURES>
void aggregate_neighbors(
    const scale_t adj_matrix[N_NODES][N_NODES],
    const data_t features[N_NODES][N_FEATURES],
    data_t agg_out[N_NODES][N_FEATURES],
    quant_scale_t scale_in,   // Scale of input features
    quant_scale_t scale_out   // Scale of output aggregation
) {
    #pragma HLS INLINE

    const int DEBUG_NODE = 6;
    
    AGG_I: for (int i = 0; i < N_NODES; i++) {
        #pragma HLS UNROLL
        AGG_F: for (int f = 0; f < N_FEATURES; f++) {
            #pragma HLS UNROLL
            
            float sum = 0.0f;
            
            AGG_J: for (int j = 0; j < N_NODES; j++) {
                #pragma HLS UNROLL
                if (adj_matrix[i][j] != 0.0f) {
                    // Dequantize input, multiply by adjacency, accumulate
                    sum += adj_matrix[i][j] * dequantize(features[j][f], scale_in);
                }
            }
            
            // Debug for node 6, features of interest
            #ifndef __SYNTHESIS__
            if (i == DEBUG_NODE && (f < 3 || f == 8 || f == 13 || f == 16 || f == 19)) {
                printf("  AGG: node=%d feat=%d sum_float=%.6f\n", i, f, sum);
            }
            #endif
            
            // Quantize result to output scale
            agg_out[i][f] = quantize(sum, scale_out);
            
            #ifndef __SYNTHESIS__
            if (i == DEBUG_NODE && (f < 3 || f == 8 || f == 13 || f == 16 || f == 19)) {
                printf("  AGG: node=%d feat=%d before_quant=%.6f after_quant=%d\n", 
                       i, f, sum/scale_out, (int)agg_out[i][f]);
            }
            #endif
        }
    }
}

/**
 * Linear transformation: out = features * W^T + bias (QUANTIZED)
 * 
 * Quantized matrix multiplication with INT8 inputs and INT32 accumulator
 * Result is requantized to INT8 output
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
    const acc_t bias[OUT_FEAT],                // Bias is INT32 in accumulator scale
    data_t output[N_NODES][OUT_FEAT],
    quant_scale_t scale_in,
    quant_scale_t scale_w,
    quant_scale_t scale_out
) {
    #pragma HLS INLINE
    
    const int DEBUG_NODE = 6;
    
    LIN_N: for (int n = 0; n < N_NODES; n++) {
        #pragma HLS UNROLL
        LIN_OUT: for (int o = 0; o < OUT_FEAT; o++) {
            #pragma HLS UNROLL
            
            // INT32 accumulator for MAC operations
            acc_t acc = 0;
            
            // Quantized MAC: acc = sum(features[i] * weights[i])
            LIN_IN: for (int i = 0; i < IN_FEAT; i++) {
                #pragma HLS UNROLL
                acc += (acc_t)features[n][i] * (acc_t)weights[o][i];
            }
            
            // Add quantized bias directly (already in accumulator scale)
            acc += bias[o];
            
            #ifndef __SYNTHESIS__
            if (n == DEBUG_NODE && o < 3) {
                printf("  LINEAR: node=%d out=%d acc_int32=%d\n", n, o, acc);
            }
            #endif
            
            // Requantize: acc is in scale (scale_in * scale_w), convert to scale_out
            float scale_factor = (scale_in * scale_w) / scale_out;
            float before_round = (float)acc * scale_factor;
            
            #ifndef __SYNTHESIS__
            if (n == DEBUG_NODE && o < 3) {
                printf("  LINEAR: node=%d out=%d scale_factor=%.10f before_round=%.6f\n", 
                       n, o, scale_factor, before_round);
            }
            #endif
            
            output[n][o] = requantize(acc, scale_in * scale_w, scale_out);
            
            #ifndef __SYNTHESIS__
            if (n == DEBUG_NODE && o < 3) {
                printf("  LINEAR: node=%d out=%d after_quant=%d\n", n, o, (int)output[n][o]);
            }
            #endif
        }
    }
}

/**
 * ReLU activation function (element-wise) - PTQ
 * 
 * For symmetric quantization with zero_point=0:
 *   ReLU(x) = max(0, x) in quantized space is just max(0, q)
 * 
 * Template parameters:
 *   N_NODES: Number of nodes
 *   N_FEATURES: Number of features per node
 */
template<int N_NODES, int N_FEATURES>
void relu_ptq(
    data_t data[N_NODES][N_FEATURES]
) {
    #pragma HLS INLINE

    RELU_N: for (int n = 0; n < N_NODES; n++) {
        #pragma HLS UNROLL
        RELU_F: for (int f = 0; f < N_FEATURES; f++) {
            #pragma HLS UNROLL
            // Simple comparison: if negative, set to 0
            if (data[n][f] < 0) {
                data[n][f] = 0;
            }
        }
    }
}

/**
 * Two-layer GraphSAGE network - QUANTIZED VERSION (PTQ)
 * Top-level function for HLS synthesis
 * 
 * Implements:
 *   h_agg1 = aggregate(input)          # quantized aggregation
 *   h1 = ReLU(W1 @ h_agg1 + b1)       # quantized linear + activation
 *   h_agg2 = aggregate(h1)             # quantized aggregation  
 *   out = W2 @ h_agg2 + b2             # quantized final linear (no activation)
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
    const acc_t bias1[HIDDEN_FEAT],            // INT32 bias in accumulator scale
    const weight_t weights2[OUT_FEAT][HIDDEN_FEAT],
    const acc_t bias2[OUT_FEAT],               // INT32 bias in accumulator scale
    data_t output[N_NODES][OUT_FEAT],
    quant_scale_t scale_in,
    quant_scale_t scale_w1,
    quant_scale_t scale_hidden,
    quant_scale_t scale_w2,
    quant_scale_t scale_out
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
    #ifndef __SYNTHESIS__
    printf("\n=== HLS DEBUG: LAYER 1 AGGREGATION ===\n");
    #endif
    
    aggregate_neighbors<N_NODES, IN_FEAT>(
        adj_matrix,
        input,
        agg1,
        scale_in,
        scale_hidden   // Output is quantized to hidden scale
    );

    #ifndef __SYNTHESIS__
    printf("Agg1[6,:] = [");
    for (int f = 0; f < IN_FEAT; f++) {
        printf("%d", (int)agg1[6][f]);
        if (f < IN_FEAT-1) printf(", ");
    }
    printf("]\n");
    #endif

    // ========== Layer 1: Linear ==========
    #ifndef __SYNTHESIS__
    printf("\n=== HLS DEBUG: LAYER 1 LINEAR ===\n");
    #endif
    
    linear_transform<N_NODES, IN_FEAT, HIDDEN_FEAT>(
        agg1,
        weights1,
        bias1,
        linear1_out,
        scale_hidden,   // Input is now in hidden scale (not scale_in!)
        scale_w1,
        scale_hidden
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
    relu_ptq<N_NODES, HIDDEN_FEAT>(hidden);

    #ifndef __SYNTHESIS__
    printf("Hidden[6,:] = [");
    for (int f = 0; f < HIDDEN_FEAT; f++) {
        printf("%d", (int)hidden[6][f]);
        if (f < HIDDEN_FEAT-1) printf(", ");
    }
    printf("]\n");
    #endif

    // ========== Layer 2: Aggregate ==========
    #ifndef __SYNTHESIS__
    printf("\n=== HLS DEBUG: LAYER 2 AGGREGATION ===\n");
    #endif
    
    aggregate_neighbors<N_NODES, HIDDEN_FEAT>(
        adj_matrix,
        hidden,
        agg2,
        scale_hidden,   // Input in hidden scale
        scale_hidden    // Output also in hidden scale
    );

    #ifndef __SYNTHESIS__
    printf("Agg2[6,:] = [");
    for (int f = 0; f < HIDDEN_FEAT; f++) {
        printf("%d", (int)agg2[6][f]);
        if (f < HIDDEN_FEAT-1) printf(", ");
    }
    printf("]\n");
    #endif

    // ========== Layer 2: Linear (no ReLU) ==========
    #ifndef __SYNTHESIS__
    printf("\n=== HLS DEBUG: LAYER 2 LINEAR ===\n");
    #endif
    
    linear_transform<N_NODES, HIDDEN_FEAT, OUT_FEAT>(
        agg2,
        weights2,
        bias2,
        output,
        scale_hidden,
        scale_w2,
        scale_out
    );

    #ifndef __SYNTHESIS__
    printf("Output[6,:] = [");
    for (int f = 0; f < OUT_FEAT; f++) {
        printf("%d", (int)output[6][f]);
        if (f < OUT_FEAT-1) printf(", ");
    }
    printf("]\n");
    printf("\n=== HLS DEBUG COMPLETE ===\n\n");
    #endif
}

/**
 * Non-templated wrapper for HLS top function (uses default config)
 * Declaration only - implementation in .cpp file
 */
void graphsage_network_ptq(
    const scale_t adj_matrix[NUM_NODES][NUM_NODES],
    const data_t input[NUM_NODES][IN_FEATURES],
    const weight_t weights1[HIDDEN_FEATURES][IN_FEATURES],
    const acc_t bias1[HIDDEN_FEATURES],
    const weight_t weights2[OUT_FEATURES][HIDDEN_FEATURES],
    const acc_t bias2[OUT_FEATURES],
    data_t output[NUM_NODES][OUT_FEATURES],
    quant_scale_t scale_in,
    quant_scale_t scale_w1,
    quant_scale_t scale_hidden,
    quant_scale_t scale_w2,
    quant_scale_t scale_out
);

#endif // GRAPHSAGE_LAYER_PTQ_H
