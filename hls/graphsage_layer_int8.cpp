/**
 * GraphSAGE Layer for FPGA Implementation - PURE INT8 VERSION
 * C++ implementation file
 * 
 * This version uses PURE INTEGER arithmetic - NO floating point in datapath!
 * All computations use INT8/INT16/INT32/INT64 with fixed-point scaling.
 * 
 * Configuration options:
 *   M_BITS=20: Smaller multipliers, 13 LSB max error vs PTQ-Float
 *   M_BITS=24: Larger multipliers, 0 LSB error (exact match) - RECOMMENDED
 */

#include "graphsage_layer_int8.h"

/**
 * Non-templated wrapper for HLS top function
 * 
 * Interface: All arrays use ap_none (direct wire, no handshake)
 * This allows for minimum latency combinatorial implementation.
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
) {

    #pragma HLS PIPELINE II=1
    // ========== HLS Interface Pragmas ==========
    // ap_none: Direct wire connections, no handshake protocol
    // This gives minimum latency but requires all inputs stable
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
    
    // ========== Array Partitioning ==========
    // Complete partitioning enables fully parallel access
    // Required for single-cycle operation with UNROLL
    #pragma HLS ARRAY_PARTITION variable=adj_matrix complete dim=1
    #pragma HLS ARRAY_PARTITION variable=adj_matrix complete dim=2
    #pragma HLS ARRAY_PARTITION variable=input complete dim=1
    #pragma HLS ARRAY_PARTITION variable=input complete dim=2
    #pragma HLS ARRAY_PARTITION variable=weights1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=weights1 complete dim=2
    #pragma HLS ARRAY_PARTITION variable=bias1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=weights2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=weights2 complete dim=2
    #pragma HLS ARRAY_PARTITION variable=bias2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=output complete dim=1
    #pragma HLS ARRAY_PARTITION variable=output complete dim=2
    
    // Call templated implementation
    graphsage_int8_template<NUM_NODES, IN_FEATURES, HIDDEN_FEATURES, OUT_FEATURES>(
        adj_matrix,
        input,
        weights1,
        bias1,
        weights2,
        bias2,
        output,
        beta1_fp,
        beta2_fp,
        eff_scale1_fp,
        eff_scale2_fp
    );
}
