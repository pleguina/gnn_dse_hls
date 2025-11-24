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
 * Aggregation: Aggregate neighbor features using adjacency matrix
 * For normalized adjacency matrix with float values
 * Uses MAX_FEATURES_HIDDEN for buffer size
 */
void aggregate(
    const scale_t adj_matrix[MAX_NODES][MAX_NODES],
    const data_t features[MAX_NODES][MAX_FEATURES_HIDDEN],
    data_t agg_out[MAX_NODES][MAX_FEATURES_HIDDEN],
    int num_nodes,
    int num_features,
    scale_t scale_in,
    scale_t scale_out
) {
    AGG_LOOP_I: for (int i = 0; i < num_nodes; i++) {
        AGG_LOOP_F: for (int f = 0; f < num_features; f++) {
            #pragma HLS PIPELINE II=1

            float sum = 0.0f;

            AGG_LOOP_J: for (int j = 0; j < num_nodes; j++) {
                #pragma HLS UNROLL factor=4
                if (adj_matrix[i][j] != 0.0f) {
                    float feat_val = dequantize(features[j][f], scale_in);
                    sum += adj_matrix[i][j] * feat_val;
                }
            }

            // Quantize output
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
) {
    #pragma HLS INTERFACE mode=s_axilite port=return
    #pragma HLS INTERFACE mode=bram port=adj_matrix
    #pragma HLS INTERFACE mode=bram port=input
    #pragma HLS INTERFACE mode=bram port=weights1
    #pragma HLS INTERFACE mode=bram port=bias1
    #pragma HLS INTERFACE mode=bram port=weights2
    #pragma HLS INTERFACE mode=bram port=bias2
    #pragma HLS INTERFACE mode=bram port=output

    // All intermediate buffers use MAX_FEATURES_HIDDEN for maximum size
    static data_t input_buf[MAX_NODES][MAX_FEATURES_HIDDEN];
    static data_t hidden[MAX_NODES][MAX_FEATURES_HIDDEN];
    static data_t agg1[MAX_NODES][MAX_FEATURES_HIDDEN];
    static data_t agg2[MAX_NODES][MAX_FEATURES_HIDDEN];
    static data_t output_buf[MAX_NODES][MAX_FEATURES_HIDDEN];

    #pragma HLS ARRAY_PARTITION variable=input_buf cyclic factor=4 dim=2
    #pragma HLS ARRAY_PARTITION variable=hidden cyclic factor=4 dim=2
    #pragma HLS ARRAY_PARTITION variable=agg1 cyclic factor=4 dim=2
    #pragma HLS ARRAY_PARTITION variable=agg2 cyclic factor=4 dim=2
    #pragma HLS ARRAY_PARTITION variable=output_buf cyclic factor=4 dim=2

    // ========== Layer 1: Fused Aggregate + Linear (like PyG SAGEConv) ==========
    // PyG does: out = W * (aggregate(x) || x)
    // We do it in one pass to avoid double quantization

    LAYER1_N: for (int n = 0; n < num_nodes; n++) {
        LAYER1_O: for (int o = 0; o < MAX_FEATURES_HIDDEN; o++) {
            #pragma HLS PIPELINE II=1

            // Accumulate in INT32 to match bias scale
            // bias was computed as: bias_fp32 / (scale_in * scale_w1)
            acc_t acc = bias1[o];

            // Do aggregation with INT8 math (to match bias scale)
            LAYER1_NEIGHBOR: for (int j = 0; j < num_nodes; j++) {
                if (adj_matrix[n][j] != 0.0f) {
                    LAYER1_FEAT: for (int i = 0; i < MAX_FEATURES_IN; i++) {
                        // INT8 x INT8 multiply, then scale by adjacency
                        acc_t product = (acc_t)input[j][i] * (acc_t)weights1[o][i];
                        acc += (acc_t)(adj_matrix[n][j] * (float)product);
                    }
                }
            }

            // Now acc is in units of: (scale_in * scale_w1) per multiplication
            // We want output in scale_hidden units
            // So requantize: acc * (scale_in * scale_w1) / scale_hidden
            float requant = (scale_in * scale_w1) / scale_hidden;
            acc_t scaled = (acc_t)(acc * requant);

            // DEBUG: Print first few values
            if (n == 0 && o == 0) {
                printf("DEBUG Layer1 [%d][%d]: acc=%d, requant=%f, scaled=%d\n", n, o, (int)acc, requant, (int)scaled);
            }

            // Clamp to INT8 and apply ReLU
            if (scaled < 0) scaled = 0;
            hidden[n][o] = (data_t)((scaled > INT8_MAX) ? INT8_MAX : ((scaled < INT8_MIN) ? INT8_MIN : scaled));
        }
    }

    // ========== Layer 2: Fused Aggregate + Linear ==========
    LAYER2_N: for (int n = 0; n < num_nodes; n++) {
        LAYER2_O: for (int o = 0; o < MAX_FEATURES_OUT; o++) {
            #pragma HLS PIPELINE II=1

            // Accumulate in INT32 to match bias scale
            acc_t acc = bias2[o];

            // Do aggregation with INT8 math (to match bias scale)
            LAYER2_NEIGHBOR: for (int j = 0; j < num_nodes; j++) {
                if (adj_matrix[n][j] != 0.0f) {
                    LAYER2_FEAT: for (int i = 0; i < MAX_FEATURES_HIDDEN; i++) {
                        // INT8 x INT8 multiply, then scale by adjacency
                        acc_t product = (acc_t)hidden[j][i] * (acc_t)weights2[o][i];
                        acc += (acc_t)(adj_matrix[n][j] * (float)product);
                    }
                }
            }

            // Requantize from (scale_hidden * scale_w2) to scale_out
            float requant = (scale_hidden * scale_w2) / scale_out;
            acc_t scaled = (acc_t)(acc * requant);

            // DEBUG: Print first few values
            if (n == 0 && o < 3) {
                printf("DEBUG Layer2 [%d][%d]: acc=%d, requant=%f, scaled=%d\n", n, o, (int)acc, requant, (int)scaled);
            }

            // Clamp to INT8 (no ReLU on final output)
            output_buf[n][o] = (data_t)((scaled > INT8_MAX) ? INT8_MAX : ((scaled < INT8_MIN) ? INT8_MIN : scaled));
        }
    }

    // Copy output from buffer
    COPY_OUTPUT: for (int n = 0; n < num_nodes; n++) {
        for (int f = 0; f < MAX_FEATURES_OUT; f++) {
            #pragma HLS PIPELINE
            output[n][f] = output_buf[n][f];
        }
    }
}
