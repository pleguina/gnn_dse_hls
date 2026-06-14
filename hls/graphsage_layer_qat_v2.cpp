/**
 * GraphSAGE QAT v2 — Top-level HLS implementation
 *
 * Fully unrolled and pipelined (II=1) to match the float/fixed references.
 * All input arrays are completely partitioned for parallel access.
 */

#include "graphsage_layer_qat_v2.h"

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
) {
#pragma HLS PIPELINE II=1

#pragma HLS INTERFACE mode=ap_none port=adj_matrix
#pragma HLS INTERFACE mode=ap_none port=input
#pragma HLS INTERFACE mode=ap_none port=weights1
#pragma HLS INTERFACE mode=ap_none port=bias1
#pragma HLS INTERFACE mode=ap_none port=weights2
#pragma HLS INTERFACE mode=ap_none port=bias2
#pragma HLS INTERFACE mode=ap_none port=output
#pragma HLS INTERFACE mode=ap_none port=beta1_fp
#pragma HLS INTERFACE mode=ap_none port=beta2_fp
#pragma HLS INTERFACE mode=ap_none port=eff_scale1_fp
#pragma HLS INTERFACE mode=ap_none port=eff_scale2_fp
#pragma HLS INTERFACE mode=ap_ctrl_none port=return

    // Complete partitioning: every array element accessible in a single cycle
#pragma HLS ARRAY_PARTITION variable=adj_matrix complete dim=1
#pragma HLS ARRAY_PARTITION variable=adj_matrix complete dim=2
#pragma HLS ARRAY_PARTITION variable=input      complete dim=1
#pragma HLS ARRAY_PARTITION variable=input      complete dim=2
#pragma HLS ARRAY_PARTITION variable=weights1   complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights1   complete dim=2
#pragma HLS ARRAY_PARTITION variable=bias1      complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights2   complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights2   complete dim=2
#pragma HLS ARRAY_PARTITION variable=bias2      complete dim=1
#pragma HLS ARRAY_PARTITION variable=output     complete dim=1
#pragma HLS ARRAY_PARTITION variable=output     complete dim=2

    graphsage_qat_v2_template<NUM_NODES, IN_FEATURES, HIDDEN_FEATURES, OUT_FEATURES>(
        adj_matrix, input,
        weights1, bias1,
        weights2, bias2,
        output,
        beta1_fp, beta2_fp,
        eff_scale1_fp, eff_scale2_fp
    );
}
