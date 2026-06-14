/**
 * GraphSAGE Layer for FPGA — QAT v2 (Quantization-Aware Training v2)
 *
 * Pure-integer arithmetic (INT8 activations/weights, INT32 accumulators).
 * Fully unrolled and pipelined — all loops have #pragma HLS UNROLL,
 * all arrays are completely partitioned, matching the float reference style.
 *
 * Quantization scheme:
 *   Input (HLS kernel):  INT8, scale = scale_proj_out
 *   Aggregation:         adj_int16 (mean-normalized * K) × INT8 features
 *                        → INT32 acc → beta_fp scale → INT8 agg_out
 *   Linear:              INT8 × INT8 → INT32 acc → eff_scale_fp → INT8 out
 *
 * Key formulas (derived from QAT fake-quantizer scales):
 *   beta_fp      = round(scale_in / (K * scale_agg_out) * 2^M)
 *   eff_scale_fp = round((scale_agg * scale_w / scale_layer_out) * 2^M)
 *   bias_int32   = round(bias_float / (scale_agg * scale_w))
 *
 * See build/weights_qat_v2/qat_v2_params.json for parameter values.
 */

#pragma once

#include <stdint.h>
#ifndef __SYNTHESIS__
#include <cstdio>
#endif
#include <ap_int.h>

// ============================================================================
// Network dimensions
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
// Fixed-point configuration
// ============================================================================

#ifndef M_BITS
#define M_BITS 24
#endif

#ifndef K_BITS
#define K_BITS 12
#endif

// ============================================================================
// Bit-widths
// ============================================================================

#define ADJ_BITS    (K_BITS + 4)
#define ACC_BITS    32
#define SCALE_BITS  (M_BITS + 8)
#define MULT_BITS   (ACC_BITS + SCALE_BITS)

#define QAT_ROUND_CONST ((ap_int<MULT_BITS>)(1) << (M_BITS - 1))

// ============================================================================
// Type definitions
// ============================================================================

typedef ap_int<8>          qv2_data_t;
typedef ap_int<8>          qv2_weight_t;
typedef ap_int<ADJ_BITS>   qv2_adj_t;
typedef ap_int<ACC_BITS>   qv2_acc_t;
typedef ap_int<ACC_BITS>   qv2_bias_t;
typedef ap_int<SCALE_BITS> qv2_scale_fp_t;
typedef ap_int<MULT_BITS>  qv2_mult_t;

// ============================================================================
// Helper: saturate to INT8
// ============================================================================

inline qv2_data_t qv2_int8_clamp(qv2_mult_t x) {
#pragma HLS INLINE
    if (x > 127)  return qv2_data_t(127);
    if (x < -128) return qv2_data_t(-128);
    return qv2_data_t(x);
}

// ============================================================================
// Aggregation — fully unrolled
//
//   q_agg[i,f] = clamp( (Σ_j adj[i,j]*q_in[j,f]) * beta_fp + ROUND >> M )
// ============================================================================

template<int N_NODES, int N_FEAT>
void qv2_aggregate(
    const qv2_adj_t  adj_matrix[N_NODES][N_NODES],
    const qv2_data_t features[N_NODES][N_FEAT],
    qv2_data_t       agg_out[N_NODES][N_FEAT],
    qv2_scale_fp_t   beta_fp
) {
#pragma HLS INLINE

QV2_AGG_I:
    for (int i = 0; i < N_NODES; i++) {
#pragma HLS UNROLL
    QV2_AGG_F:
        for (int f = 0; f < N_FEAT; f++) {
#pragma HLS UNROLL
            qv2_acc_t tmp = 0;

        QV2_AGG_J:
            for (int j = 0; j < N_NODES; j++) {
#pragma HLS UNROLL
                tmp += (qv2_acc_t)adj_matrix[i][j] * (qv2_acc_t)features[j][f];
            }

            qv2_mult_t scaled  = (qv2_mult_t)tmp * (qv2_mult_t)beta_fp;
            qv2_mult_t rounded = scaled + QAT_ROUND_CONST;
            agg_out[i][f] = qv2_int8_clamp(rounded >> M_BITS);
        }
    }
}

// ============================================================================
// Linear transform + requantisation — fully unrolled
//
//   acc = bias[o] + Σ_f q_in[n,f] * w[o,f]
//   q_out[n,o] = clamp( acc * eff_scale_fp + ROUND >> M )
// ============================================================================

template<int N_NODES, int IN_FEAT, int OUT_FEAT>
void qv2_linear(
    const qv2_data_t   features[N_NODES][IN_FEAT],
    const qv2_weight_t weights[OUT_FEAT][IN_FEAT],
    const qv2_bias_t   bias[OUT_FEAT],
    qv2_data_t         output[N_NODES][OUT_FEAT],
    qv2_scale_fp_t     eff_scale_fp
) {
#pragma HLS INLINE

QV2_LIN_N:
    for (int n = 0; n < N_NODES; n++) {
#pragma HLS UNROLL
    QV2_LIN_O:
        for (int o = 0; o < OUT_FEAT; o++) {
#pragma HLS UNROLL
            qv2_acc_t acc = bias[o];

        QV2_LIN_F:
            for (int f = 0; f < IN_FEAT; f++) {
#pragma HLS UNROLL
                acc += (qv2_acc_t)features[n][f] * (qv2_acc_t)weights[o][f];
            }

            qv2_mult_t scaled  = (qv2_mult_t)acc * (qv2_mult_t)eff_scale_fp;
            qv2_mult_t rounded = scaled + QAT_ROUND_CONST;
            output[n][o] = qv2_int8_clamp(rounded >> M_BITS);
        }
    }
}

// ============================================================================
// ReLU — fully unrolled
// ============================================================================

template<int N_NODES, int N_FEAT>
void qv2_relu(qv2_data_t data[N_NODES][N_FEAT]) {
#pragma HLS INLINE

QV2_RELU_N:
    for (int n = 0; n < N_NODES; n++) {
#pragma HLS UNROLL
    QV2_RELU_F:
        for (int f = 0; f < N_FEAT; f++) {
#pragma HLS UNROLL
            if (data[n][f] < 0) data[n][f] = 0;
        }
    }
}

// ============================================================================
// Full two-layer network — fully unrolled, pipelined
// ============================================================================

template<int N_NODES, int IN_FEAT, int HIDDEN_FEAT, int OUT_FEAT>
void graphsage_qat_v2_template(
    const qv2_adj_t    adj_matrix[N_NODES][N_NODES],
    const qv2_data_t   input[N_NODES][IN_FEAT],
    const qv2_weight_t weights1[HIDDEN_FEAT][IN_FEAT],
    const qv2_bias_t   bias1[HIDDEN_FEAT],
    const qv2_weight_t weights2[OUT_FEAT][HIDDEN_FEAT],
    const qv2_bias_t   bias2[OUT_FEAT],
    qv2_data_t         output[N_NODES][OUT_FEAT],
    qv2_scale_fp_t     beta1_fp,
    qv2_scale_fp_t     beta2_fp,
    qv2_scale_fp_t     eff_scale1_fp,
    qv2_scale_fp_t     eff_scale2_fp
) {
#pragma HLS PIPELINE II=1

    qv2_data_t agg1  [N_NODES][IN_FEAT];
    qv2_data_t hidden[N_NODES][HIDDEN_FEAT];
    qv2_data_t agg2  [N_NODES][HIDDEN_FEAT];

#pragma HLS ARRAY_PARTITION variable=agg1   complete dim=1
#pragma HLS ARRAY_PARTITION variable=agg1   complete dim=2
#pragma HLS ARRAY_PARTITION variable=hidden complete dim=1
#pragma HLS ARRAY_PARTITION variable=hidden complete dim=2
#pragma HLS ARRAY_PARTITION variable=agg2   complete dim=1
#pragma HLS ARRAY_PARTITION variable=agg2   complete dim=2

    qv2_aggregate<N_NODES, IN_FEAT>   (adj_matrix, input,   agg1,   beta1_fp);
    qv2_linear   <N_NODES, IN_FEAT,   HIDDEN_FEAT>(agg1,    weights1, bias1, hidden, eff_scale1_fp);
    qv2_relu     <N_NODES, HIDDEN_FEAT>(hidden);
    qv2_aggregate<N_NODES, HIDDEN_FEAT>(adj_matrix, hidden,  agg2,   beta2_fp);
    qv2_linear   <N_NODES, HIDDEN_FEAT, OUT_FEAT>  (agg2,   weights2, bias2, output, eff_scale2_fp);
}

// ============================================================================
// Top-level declaration
// ============================================================================

void graphsage_qat_v2(
    const qv2_adj_t    adj_matrix[NUM_NODES][NUM_NODES],
    const qv2_data_t   input    [NUM_NODES][IN_FEATURES],
    const qv2_weight_t weights1 [HIDDEN_FEATURES][IN_FEATURES],
    const qv2_bias_t   bias1    [HIDDEN_FEATURES],
    const qv2_weight_t weights2 [OUT_FEATURES][HIDDEN_FEATURES],
    const qv2_bias_t   bias2    [OUT_FEATURES],
    qv2_data_t         output   [NUM_NODES][OUT_FEATURES],
    qv2_scale_fp_t     beta1_fp,
    qv2_scale_fp_t     beta2_fp,
    qv2_scale_fp_t     eff_scale1_fp,
    qv2_scale_fp_t     eff_scale2_fp
);
