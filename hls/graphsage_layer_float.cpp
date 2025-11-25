/**
 * GraphSAGE Layer for FPGA Implementation - FLOAT VERSION (Templated)
 * C++ implementation with HLS pragmas
 * 
 * This version uses FLOAT arithmetic throughout to match PyG exactly.
 * Goal: Validate graph structure, aggregation, and SAGEConv logic.
 * No quantization errors.
 * 
 * Fully templated to support different model configurations.
 * Template implementations are in the header file.
 */

#include "graphsage_layer_float.h"
#include <cmath>

/**
 * Non-templated wrapper function for HLS top function
 * Uses the default configuration from defines
 */
void graphsage_network(
    const scale_t adj_matrix[NUM_NODES][NUM_NODES],
    const data_t input[NUM_NODES][IN_FEATURES],
    const weight_t weights1[HIDDEN_FEATURES][IN_FEATURES],
    const weight_t bias1[HIDDEN_FEATURES],
    const weight_t weights2[OUT_FEATURES][HIDDEN_FEATURES],
    const weight_t bias2[OUT_FEATURES],
    data_t output[NUM_NODES][OUT_FEATURES]
) {
    // Note: No #pragma HLS INLINE here so C simulation can compile
    // HLS will inline this during synthesis anyway since it just calls the template
    
    // HLS interface pragmas - ap_none for direct wire connections (no handshake)
    #pragma HLS INTERFACE mode=ap_none port=adj_matrix
    #pragma HLS INTERFACE mode=ap_none port=input
    #pragma HLS INTERFACE mode=ap_none port=weights1
    #pragma HLS INTERFACE mode=ap_none port=bias1
    #pragma HLS INTERFACE mode=ap_none port=weights2
    #pragma HLS INTERFACE mode=ap_none port=bias2
    #pragma HLS INTERFACE mode=ap_none port=output
    #pragma HLS INTERFACE mode=ap_ctrl_none port=return
    
    // Top-level pipeline with II=1 for fully parallel execution
    
    // Complete array partitioning for input arrays (fully parallel access)
    #pragma HLS ARRAY_PARTITION variable=adj_matrix complete dim=2
    #pragma HLS ARRAY_PARTITION variable=adj_matrix complete dim=1
    #pragma HLS ARRAY_PARTITION variable=input complete dim=2
    #pragma HLS ARRAY_PARTITION variable=input complete dim=1
    #pragma HLS ARRAY_PARTITION variable=weights1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=weights1 complete dim=2
    #pragma HLS ARRAY_PARTITION variable=bias1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=weights2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=weights2 complete dim=2
    #pragma HLS ARRAY_PARTITION variable=bias2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=output complete dim=2
    #pragma HLS ARRAY_PARTITION variable=output complete dim=1
    // Call templated implementation with default configuration
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


