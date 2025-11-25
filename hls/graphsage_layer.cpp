/**
 * GraphSAGE Layer for FPGA Implementation
 * C++ implementation with HLS pragmas
 */

#include "graphsage_layer.h"
#include <cmath>

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Quantize float value to int8
 */
inline data_t quantize(float value, scale_t scale) {
    int32_t quantized = (int32_t)(value / scale + 0.5f);
    if (quantized > INT8_MAX) return INT8_MAX;
    if (quantized < INT8_MIN) return INT8_MIN;
    return (data_t)quantized;
}

/**
 * Dequantize int8 to float
 */
inline float dequantize(data_t value, scale_t scale) {
    return (float)value * scale;
}

// ============================================================================
// Core Functions
// ============================================================================

/**
 * Matrix multiplication: C = A * B
 * Optimized for FPGA with pipelining
 */
void matmul_int8(
    const data_t A[MAX_NODES][MAX_FEATURES_IN],
    const data_t B[MAX_FEATURES_IN][MAX_FEATURES_OUT],
    acc_t C[MAX_NODES][MAX_FEATURES_OUT],
    int M, int K, int N
) {
    // Initialize output
    INIT_LOOP_M: for (int i = 0; i < M; i++) {
        #pragma HLS PIPELINE
        INIT_LOOP_N: for (int j = 0; j < N; j++) {
            C[i][j] = 0;
        }
    }

    // Matrix multiplication
    MM_LOOP_M: for (int i = 0; i < M; i++) {
        MM_LOOP_N: for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            acc_t sum = 0;
            MM_LOOP_K: for (int k = 0; k < K; k++) {
                #pragma HLS UNROLL factor=4
                sum += (acc_t)A[i][k] * (acc_t)B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

/**
 * Aggregation: Aggregate neighbor features using adjacency matrix (for input layer)
 * For normalized adjacency matrix with float values
 */
void aggregate_input(
    const scale_t adj_matrix[MAX_NODES][MAX_NODES],
    const data_t features[MAX_NODES][MAX_FEATURES_IN],
    data_t agg_out[MAX_NODES][MAX_FEATURES_IN],
    int num_nodes,
    int num_features,
    scale_t scale_in,   // scale of `features`
    scale_t scale_out   // desired scale of `agg_out`
) {
    AGG_LOOP_I: for (int i = 0; i < num_nodes; i++) {
        AGG_LOOP_F: for (int f = 0; f < num_features; f++) {
            #pragma HLS PIPELINE II=1
            float sum = 0.0f;

            AGG_LOOP_J: for (int j = 0; j < num_nodes; j++) {
                float adj_val = adj_matrix[i][j];
                if (adj_val != 0.0f) {
                    float feat_val = dequantize(features[j][f], scale_in);
                    sum += adj_val * feat_val;
                }
            }

            agg_out[i][f] = quantize(sum, scale_out);
        }
    }
}

/**
 * Aggregation: Aggregate neighbor features using adjacency matrix (for hidden layer)
 * For normalized adjacency matrix with float values
 */
void aggregate_hidden(
    const scale_t adj_matrix[MAX_NODES][MAX_NODES],
    const data_t features[MAX_NODES][MAX_FEATURES_HIDDEN],
    data_t agg_out[MAX_NODES][MAX_FEATURES_HIDDEN],
    int num_nodes,
    int num_features,
    scale_t scale_in,   // scale of `features`
    scale_t scale_out   // desired scale of `agg_out`
) {
    AGG_LOOP_I: for (int i = 0; i < num_nodes; i++) {
        AGG_LOOP_F: for (int f = 0; f < num_features; f++) {
            #pragma HLS PIPELINE II=1

            float sum = 0.0f;

            AGG_LOOP_J: for (int j = 0; j < num_nodes; j++) {
                #pragma HLS UNROLL  // full unroll over neighbors (small N)
                scale_t a_ij = adj_matrix[i][j];
                if (a_ij != 0.0f) {
                    float feat_val = dequantize(features[j][f], scale_in);
                    sum += a_ij * feat_val;
                }
            }

            // Quantize aggregated value to output scale:
            agg_out[i][f] = quantize(sum, scale_out);
        }
    }
}


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
) {
    LIN_LOOP_N: for (int n = 0; n < num_nodes; n++) {
        LIN_LOOP_OUT: for (int o = 0; o < out_features; o++) {
            #pragma HLS PIPELINE II=1

            acc_t sum = bias[o];

            LIN_LOOP_IN: for (int i = 0; i < in_features; i++) {
                #pragma HLS UNROLL factor=4
                sum += (acc_t)input[n][i] * (acc_t)weights[o][i];
            }

            // Dequantize, then requantize to output scale
            float result = (float)sum * scale_in * scale_weight;
            output[n][o] = quantize(result, scale_out);
        }
    }
}

/**
 * ReLU activation function (INT8 quantized)
 * Uses MAX_FEATURES_HIDDEN for maximum buffer size
 */
void relu_int8(
    data_t data[MAX_NODES][MAX_FEATURES_HIDDEN],
    int num_nodes,
    int num_features
) {
    RELU_LOOP_N: for (int n = 0; n < num_nodes; n++) {
        RELU_LOOP_F: for (int f = 0; f < num_features; f++) {
            #pragma HLS PIPELINE
            if (data[n][f] < 0) {
                data[n][f] = 0;
            }
        }
    }
}

// Generic graphsage_layer removed for simplicity
// Network is implemented directly in graphsage_network function below

/**
 * Two-layer GraphSAGE network
 * Top-level function for HLS synthesis
 */
void graphsage_network(
    const scale_t adj_matrix[MAX_NODES][MAX_NODES],
    const data_t input[MAX_NODES][MAX_FEATURES_IN],

    // Layer 1 parameters (conv1.lin_l)
    const data_t weights1[MAX_FEATURES_HIDDEN][MAX_FEATURES_IN],
    const acc_t  bias1[MAX_FEATURES_HIDDEN],
    scale_t      scale_w1, // s_w1

    // Layer 2 parameters (conv2.lin_l)
    const data_t weights2[MAX_FEATURES_OUT][MAX_FEATURES_HIDDEN],
    const acc_t  bias2[MAX_FEATURES_OUT],
    scale_t      scale_w2, // s_w2

    data_t output[MAX_NODES][MAX_FEATURES_OUT],
    int num_nodes,

    // Activation scales:
    scale_t scale_in,      // s_x0
    scale_t scale_hidden,  // s_h1
    scale_t scale_out      // s_y
) {
    #pragma HLS INTERFACE mode=s_axilite port=return
    #pragma HLS INTERFACE mode=bram port=adj_matrix
    #pragma HLS INTERFACE mode=bram port=input
    #pragma HLS INTERFACE mode=bram port=weights1
    #pragma HLS INTERFACE mode=bram port=bias1
    #pragma HLS INTERFACE mode=bram port=weights2
    #pragma HLS INTERFACE mode=bram port=bias2
    #pragma HLS INTERFACE mode=bram port=output

    // Intermediate buffers with correct sizes
    static data_t agg1[MAX_NODES][MAX_FEATURES_IN];      // Aggregated input
    static data_t hidden[MAX_NODES][MAX_FEATURES_HIDDEN]; // Layer 1 output
    static data_t agg2[MAX_NODES][MAX_FEATURES_HIDDEN];   // Aggregated hidden
    static data_t output_buf[MAX_NODES][MAX_FEATURES_OUT]; // Final output

    #pragma HLS ARRAY_PARTITION variable=agg1   cyclic factor=4 dim=2
    #pragma HLS ARRAY_PARTITION variable=hidden cyclic factor=4 dim=2
    #pragma HLS ARRAY_PARTITION variable=agg2   cyclic factor=4 dim=2
    #pragma HLS ARRAY_PARTITION variable=output_buf cyclic factor=4 dim=2

    // ========== Layer 1: Aggregate (mean) ==========
    // PyG: h1_agg = mean_{j in N(i)} x_j
    // HLS: agg1 = A * input (with dequantize/quantize inside `aggregate`)
    aggregate_input(
        adj_matrix,
        input,
        agg1,
        num_nodes,
        MAX_FEATURES_IN,
        scale_in,     // input is in scale_in
        scale_hidden  // agg1 will be in scale_hidden
    );

    // ========== Layer 1: Linear transform + ReLU ==========
    // PyG: h1 = ReLU( W1 * h1_agg + b1 )

    L1_N: for (int n = 0; n < num_nodes; n++) {
        L1_O: for (int o = 0; o < MAX_FEATURES_HIDDEN; o++) {
            #pragma HLS PIPELINE II=1

            // bias1 was pre-quantized as: b1_real / (scale_hidden * scale_w1)
            acc_t acc = bias1[o];

            // Integer MAC: sum_i agg1[n][i] * weights1[o][i]
            L1_I: for (int i = 0; i < MAX_FEATURES_IN; i++) {
                #pragma HLS UNROLL factor=4
                acc += (acc_t)agg1[n][i] * (acc_t)weights1[o][i];
            }

            // Real value: y_real = acc * (scale_hidden * scale_w1)
            float y_real = (float)acc * (scale_hidden * scale_w1);

            // Quantize back to int8 in scale_hidden (keep same activation scale)
            data_t q = quantize(y_real, scale_hidden);

            // ReLU in int8
            if (q < 0) q = 0;

            hidden[n][o] = q;
        }
    }

    // ========== Layer 2: Aggregate (mean) ==========
    // PyG: h2_agg = mean_{j in N(i)} h1_j

    aggregate_hidden(
        adj_matrix,
        hidden,
        agg2,
        num_nodes,
        MAX_FEATURES_HIDDEN,
        scale_hidden, // hidden in scale_hidden
        scale_hidden  // keep same scale for agg2
    );

    // ========== Layer 2: Linear transform (no ReLU on final) ==========
    // PyG: out = W2 * h2_agg + b2

    L2_N: for (int n = 0; n < num_nodes; n++) {
        L2_O: for (int o = 0; o < MAX_FEATURES_OUT; o++) {
            #pragma HLS PIPELINE II=1

            // bias2 was pre-quantized as: b2_real / (scale_hidden * scale_w2)
            acc_t acc = bias2[o];

            L2_I: for (int i = 0; i < MAX_FEATURES_HIDDEN; i++) {
                #pragma HLS UNROLL factor=4
                acc += (acc_t)agg2[n][i] * (acc_t)weights2[o][i];
            }

            // Real value: y_real = acc * (scale_hidden * scale_w2)
            float y_real = (float)acc * (scale_hidden * scale_w2);

            // Quantize to final output scale
            output_buf[n][o] = quantize(y_real, scale_out);
        }
    }

    // Copy output
    COPY_OUTPUT: for (int n = 0; n < num_nodes; n++) {
        for (int f = 0; f < MAX_FEATURES_OUT; f++) {
            #pragma HLS PIPELINE
            output[n][f] = output_buf[n][f];
        }
    }
}
