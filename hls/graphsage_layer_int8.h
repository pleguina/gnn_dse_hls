/**
 * GraphSAGE Layer for FPGA Implementation - PURE INT8 VERSION
 * Header file with type definitions and function declarations
 * 
 * This version uses PURE INTEGER arithmetic - NO floating point operations!
 * 
 * Based on analysis from tests/compare_ptq_float_vs_int8_detailed.py:
 *   - M=20: 13 LSB max error vs PTQ-Float (smaller multipliers)
 *   - M=24: 0 LSB error vs PTQ-Float (exact match, recommended)
 * 
 * Quantization scheme:
 *   - Weights: INT8 symmetric quantization
 *   - Activations: INT8 symmetric quantization  
 *   - Accumulators: INT32 for intermediate sums
 *   - Adjacency: INT16 (scaled by K=4096)
 *   - Scale factors: Fixed-point with M fractional bits
 *   - Rounding: (x + (1 << (M-1))) >> M (round half up)
 * 
 * Key formulas:
 *   Aggregation: q_agg = (sum(A_int16 * q_in) * beta_fp + (1<<(M-1))) >> M
 *   Linear:      q_out = (acc * eff_scale_fp + (1<<(M-1))) >> M
 * 
 * Where:
 *   beta_fp = round(scale_in / (K * scale_out) * 2^M)
 *   eff_scale_fp = round((scale_in * scale_w / scale_out) * 2^M)
 */

#ifndef GRAPHSAGE_LAYER_INT8_H
#define GRAPHSAGE_LAYER_INT8_H

#include <stdint.h>
#ifndef __SYNTHESIS__
#include <cstdio>
#endif

// ============================================================================
// Configuration Parameters
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
// Fixed-Point Configuration
// ============================================================================

// M = number of fractional bits for fixed-point scales
// M=20: 13 LSB max error, smaller multipliers (20-bit scales)
// M=24: 0 LSB error (exact match), larger multipliers (24-bit scales)
#ifndef M_BITS
#define M_BITS 24  // Recommended for exact match with PTQ-Float
#endif

// K = adjacency matrix scaling factor (2^K_BITS)
#ifndef K_BITS
#define K_BITS 12
#endif
#define K_VALUE (1 << K_BITS)  // 4096

// Rounding constant: (1 << (M-1))
#define ROUND_CONST (1LL << (M_BITS - 1))

// ============================================================================
// Type Definitions - PURE INTEGER VERSION
// ============================================================================

typedef int8_t   data_t;      // Quantized activations (INT8)
typedef int8_t   weight_t;    // Quantized weights (INT8)
typedef int16_t  adj_t;       // Adjacency matrix (INT16, scaled by K)
typedef int32_t  acc_t;       // Accumulator for MAC (INT32)
typedef int32_t  bias_t;      // Bias (INT32, in accumulator domain)
typedef int32_t  scale_fp_t;  // Fixed-point scale factors (INT32)
typedef int64_t  mult_t;      // For 32x32 multiplication results (INT64)

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Clamp INT64 result to INT8 range [-128, 127]
 */
inline int8_t int8_clamp(int64_t x) {
    #pragma HLS INLINE
    if (x > 127) return 127;
    if (x < -128) return -128;
    return (int8_t)x;
}

/**
 * Fixed-point multiply and shift with rounding
 * result = (a * b + ROUND) >> M
 */
inline int64_t fp_mult_shift(int64_t a, int32_t b) {
    #pragma HLS INLINE
    int64_t product = a * (int64_t)b;
    int64_t rounded = product + ROUND_CONST;
    return rounded >> M_BITS;
}

// ============================================================================
// Pure Integer Aggregation
// ============================================================================

/**
 * Integer-only mean aggregation
 * 
 * Formula: q_agg[i,f] = clamp((sum_j(A_int16[i,j] * q_in[j,f]) * beta_fp + ROUND) >> M)
 * 
 * Where:
 *   A_int16[i,j] = round(adj_float[i,j] * K)  (precomputed)
 *   beta_fp = round(scale_in / (K * scale_out) * 2^M)
 * 
 * Template parameters:
 *   N_NODES: Number of nodes
 *   N_FEAT: Number of features
 */
template<int N_NODES, int N_FEAT>
void aggregate_int8(
    const adj_t adj_matrix[N_NODES][N_NODES],
    const data_t features[N_NODES][N_FEAT],
    data_t agg_out[N_NODES][N_FEAT],
    scale_fp_t beta_fp
) {
    #pragma HLS INLINE

    AGG_I: for (int i = 0; i < N_NODES; i++) {
        #pragma HLS UNROLL
        AGG_F: for (int f = 0; f < N_FEAT; f++) {
            #pragma HLS UNROLL
            
            // INT32 accumulator for weighted sum
            acc_t tmp = 0;
            
            AGG_J: for (int j = 0; j < N_NODES; j++) {
                #pragma HLS UNROLL
                // INT16 * INT8 -> INT32 (no overflow for reasonable N_NODES)
                tmp += (acc_t)adj_matrix[i][j] * (acc_t)features[j][f];
            }
            
            // Apply fixed-point scale: (tmp * beta_fp + ROUND) >> M
            mult_t scaled = (mult_t)tmp * (mult_t)beta_fp;
            mult_t rounded = scaled + ROUND_CONST;
            int64_t result = rounded >> M_BITS;
            
            // Clamp to INT8
            agg_out[i][f] = int8_clamp(result);
            
            #ifndef __SYNTHESIS__
            if (i == 6 && f < 3) {
                printf("  AGG_INT8: node=%d feat=%d tmp=%d scaled=%lld result=%lld out=%d\n",
                       i, f, tmp, (long long)scaled, (long long)result, (int)agg_out[i][f]);
            }
            #endif
        }
    }
}

// ============================================================================
// Pure Integer Linear Transform
// ============================================================================

/**
 * Integer-only linear transformation
 * 
 * Formula: q_out[n,o] = clamp((acc * eff_scale_fp + ROUND) >> M)
 * Where: acc = bias_int32[o] + sum_f(q_in[n,f] * w_int8[o,f])
 * 
 * The bias is pre-scaled to accumulator domain:
 *   bias_int32 = round(bias_float / (scale_in * scale_w))
 * 
 * eff_scale_fp = round((scale_in * scale_w / scale_out) * 2^M)
 * 
 * Template parameters:
 *   N_NODES: Number of nodes
 *   IN_FEAT: Input features
 *   OUT_FEAT: Output features
 */
template<int N_NODES, int IN_FEAT, int OUT_FEAT>
void linear_int8(
    const data_t features[N_NODES][IN_FEAT],
    const weight_t weights[OUT_FEAT][IN_FEAT],
    const bias_t bias[OUT_FEAT],
    data_t output[N_NODES][OUT_FEAT],
    scale_fp_t eff_scale_fp
) {
    #pragma HLS INLINE

    LIN_N: for (int n = 0; n < N_NODES; n++) {
        #pragma HLS UNROLL
        LIN_O: for (int o = 0; o < OUT_FEAT; o++) {
            #pragma HLS UNROLL
            
            // INT32 accumulator: start with bias
            acc_t acc = bias[o];
            
            // MAC: acc += sum(x * w)
            LIN_F: for (int f = 0; f < IN_FEAT; f++) {
                #pragma HLS UNROLL
                // INT8 * INT8 -> INT32
                acc += (acc_t)features[n][f] * (acc_t)weights[o][f];
            }
            
            // Apply fixed-point scale: (acc * eff_scale_fp + ROUND) >> M
            mult_t scaled = (mult_t)acc * (mult_t)eff_scale_fp;
            mult_t rounded = scaled + ROUND_CONST;
            int64_t result = rounded >> M_BITS;
            
            // Clamp to INT8
            output[n][o] = int8_clamp(result);
            
            #ifndef __SYNTHESIS__
            if (n == 6 && o < 3) {
                printf("  LIN_INT8: node=%d out=%d acc=%d scaled=%lld result=%lld out=%d\n",
                       n, o, acc, (long long)scaled, (long long)result, (int)output[n][o]);
            }
            #endif
        }
    }
}

// ============================================================================
// Pure Integer ReLU
// ============================================================================

/**
 * ReLU for INT8 with symmetric quantization (zero_point = 0)
 * Simply: max(0, x)
 */
template<int N_NODES, int N_FEAT>
void relu_int8(
    data_t data[N_NODES][N_FEAT]
) {
    #pragma HLS INLINE

    RELU_N: for (int n = 0; n < N_NODES; n++) {
        #pragma HLS UNROLL
        RELU_F: for (int f = 0; f < N_FEAT; f++) {
            #pragma HLS UNROLL
            if (data[n][f] < 0) {
                data[n][f] = 0;
            }
        }
    }
}

// ============================================================================
// Full Network - Pure Integer
// ============================================================================

/**
 * Two-layer GraphSAGE network - PURE INT8 VERSION
 * 
 * NO FLOATING POINT OPERATIONS IN THE DATAPATH!
 * 
 * Layer 1: agg1 = aggregate(input, beta1_fp)
 *          hidden = ReLU(linear(agg1, W1, b1, eff_scale1_fp))
 * Layer 2: agg2 = aggregate(hidden, beta2_fp)
 *          output = linear(agg2, W2, b2, eff_scale2_fp)
 * 
 * Fixed-point parameters (precomputed):
 *   beta1_fp: Aggregation scale for layer 1
 *   beta2_fp: Aggregation scale for layer 2 (= K for hidden->hidden)
 *   eff_scale1_fp: Linear requantization scale for layer 1
 *   eff_scale2_fp: Linear requantization scale for layer 2
 */
template<int N_NODES, int IN_FEAT, int HIDDEN_FEAT, int OUT_FEAT>
void graphsage_int8_template(
    const adj_t adj_matrix[N_NODES][N_NODES],
    const data_t input[N_NODES][IN_FEAT],
    const weight_t weights1[HIDDEN_FEAT][IN_FEAT],
    const bias_t bias1[HIDDEN_FEAT],
    const weight_t weights2[OUT_FEAT][HIDDEN_FEAT],
    const bias_t bias2[OUT_FEAT],
    data_t output[N_NODES][OUT_FEAT],
    scale_fp_t beta1_fp,
    scale_fp_t beta2_fp,
    scale_fp_t eff_scale1_fp,
    scale_fp_t eff_scale2_fp
) {
    #pragma HLS PIPELINE II=1
    
    // Intermediate buffers
    data_t agg1[N_NODES][IN_FEAT];
    data_t hidden[N_NODES][HIDDEN_FEAT];
    data_t agg2[N_NODES][HIDDEN_FEAT];

    #pragma HLS ARRAY_PARTITION variable=agg1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=agg1 complete dim=2
    #pragma HLS ARRAY_PARTITION variable=hidden complete dim=1
    #pragma HLS ARRAY_PARTITION variable=hidden complete dim=2
    #pragma HLS ARRAY_PARTITION variable=agg2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=agg2 complete dim=2

    // ========== Layer 1: Aggregate ==========
    #ifndef __SYNTHESIS__
    printf("\n=== INT8 HLS: LAYER 1 AGGREGATION (beta1_fp=%d) ===\n", beta1_fp);
    #endif
    
    aggregate_int8<N_NODES, IN_FEAT>(adj_matrix, input, agg1, beta1_fp);

    #ifndef __SYNTHESIS__
    printf("Agg1[6,:8] = [");
    for (int f = 0; f < 8 && f < IN_FEAT; f++) {
        printf("%d%s", (int)agg1[6][f], f < 7 ? ", " : "");
    }
    printf("]\n");
    #endif

    // ========== Layer 1: Linear + ReLU ==========
    #ifndef __SYNTHESIS__
    printf("\n=== INT8 HLS: LAYER 1 LINEAR (eff_scale1_fp=%d) ===\n", eff_scale1_fp);
    #endif
    
    linear_int8<N_NODES, IN_FEAT, HIDDEN_FEAT>(agg1, weights1, bias1, hidden, eff_scale1_fp);
    relu_int8<N_NODES, HIDDEN_FEAT>(hidden);

    #ifndef __SYNTHESIS__
    printf("Hidden[6,:8] = [");
    for (int f = 0; f < 8 && f < HIDDEN_FEAT; f++) {
        printf("%d%s", (int)hidden[6][f], f < 7 ? ", " : "");
    }
    printf("]\n");
    #endif

    // ========== Layer 2: Aggregate ==========
    #ifndef __SYNTHESIS__
    printf("\n=== INT8 HLS: LAYER 2 AGGREGATION (beta2_fp=%d) ===\n", beta2_fp);
    #endif
    
    aggregate_int8<N_NODES, HIDDEN_FEAT>(adj_matrix, hidden, agg2, beta2_fp);

    #ifndef __SYNTHESIS__
    printf("Agg2[6,:8] = [");
    for (int f = 0; f < 8 && f < HIDDEN_FEAT; f++) {
        printf("%d%s", (int)agg2[6][f], f < 7 ? ", " : "");
    }
    printf("]\n");
    #endif

    // ========== Layer 2: Linear (no ReLU) ==========
    #ifndef __SYNTHESIS__
    printf("\n=== INT8 HLS: LAYER 2 LINEAR (eff_scale2_fp=%d) ===\n", eff_scale2_fp);
    #endif
    
    linear_int8<N_NODES, HIDDEN_FEAT, OUT_FEAT>(agg2, weights2, bias2, output, eff_scale2_fp);

    #ifndef __SYNTHESIS__
    printf("Output[6,:] = [");
    for (int f = 0; f < OUT_FEAT; f++) {
        printf("%d%s", (int)output[6][f], f < OUT_FEAT-1 ? ", " : "");
    }
    printf("]\n");
    printf("\n=== INT8 HLS COMPLETE ===\n\n");
    #endif
}

// ============================================================================
// Top-level function declaration
// ============================================================================

/**
 * Non-templated wrapper for HLS synthesis
 */
void graphsage_int8(
    const adj_t adj_matrix[NUM_NODES][NUM_NODES],
    const data_t input[NUM_NODES][IN_FEATURES],
    const weight_t weights1[HIDDEN_FEATURES][IN_FEATURES],
    const bias_t bias1[HIDDEN_FEATURES],
    const weight_t weights2[OUT_FEATURES][HIDDEN_FEATURES],
    const bias_t bias2[OUT_FEATURES],
    data_t output[NUM_NODES][OUT_FEATURES],
    scale_fp_t beta1_fp,
    scale_fp_t beta2_fp,
    scale_fp_t eff_scale1_fp,
    scale_fp_t eff_scale2_fp
);

#endif // GRAPHSAGE_LAYER_INT8_H
