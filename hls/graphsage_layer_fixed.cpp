/**
 * GraphSAGE Layer for FPGA Implementation - FIXED-POINT VERSION
 * C++ implementation with HLS pragmas
 * 
 * This version uses ap_fixed for configurable fixed-point arithmetic.
 * Wrapper function for HLS top synthesis.
 */

#include "graphsage_layer_fixed.h"

/**
 * Non-templated wrapper function for HLS top function
 * Uses the default configuration from defines
 */
void graphsage_network_fixed(
    const scale_t adj_matrix[NUM_NODES][NUM_NODES],
    const data_t input[NUM_NODES][IN_FEATURES],
    const weight_t weights1[HIDDEN_FEATURES][IN_FEATURES],
    const weight_t bias1[HIDDEN_FEATURES],
    const weight_t weights2[OUT_FEATURES][HIDDEN_FEATURES],
    const weight_t bias2[OUT_FEATURES],
    data_t output[NUM_NODES][OUT_FEATURES]
) {

    #pragma HLS PIPELINE II=1
    // Note: No #pragma HLS INLINE here so C simulation can compile
    // HLS will inline this during synthesis anyway since it just calls the template
    
    // Interface pragmas
    #pragma HLS INTERFACE mode=ap_none port=adj_matrix
    #pragma HLS INTERFACE mode=ap_none port=input
    #pragma HLS INTERFACE mode=ap_none port=weights1
    #pragma HLS INTERFACE mode=ap_none port=bias1
    #pragma HLS INTERFACE mode=ap_none port=weights2
    #pragma HLS INTERFACE mode=ap_none port=bias2
    #pragma HLS INTERFACE mode=ap_none port=output
    #pragma HLS INTERFACE mode=ap_ctrl_none port=return

    // Complete array partitioning on all ports for fully parallel access
    #pragma HLS ARRAY_PARTITION variable=adj_matrix complete dim=0
    #pragma HLS ARRAY_PARTITION variable=input complete dim=0
    #pragma HLS ARRAY_PARTITION variable=weights1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=bias1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=weights2 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=bias2 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=output complete dim=0

    // Call templated implementation
    graphsage_network_template<NUM_NODES, IN_FEATURES, HIDDEN_FEATURES, OUT_FEATURES>(
        adj_matrix,
        input,
        weights1,
        bias1,
        weights2,
        bias2,
        output
    );
}
