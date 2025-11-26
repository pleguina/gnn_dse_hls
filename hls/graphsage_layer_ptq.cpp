/**
 * GraphSAGE Layer for FPGA Implementation - INT8 PTQ VERSION
 * C++ implementation with HLS pragmas
 * 
 * This version uses INT8 quantized arithmetic for weights and activations.
 * Post-Training Quantization with symmetric quantization (zero_point = 0).
 * 
 * Fully templated to support different model configurations.
 * Template implementations are in the header file.
 */

#include "graphsage_layer_ptq.h"

/**
 * Non-templated wrapper function for HLS top function
 * Uses the default configuration from defines
 */
void graphsage_network_ptq(
    const scale_t adj_matrix[NUM_NODES][NUM_NODES],
    const data_t input[NUM_NODES][IN_FEATURES],
    const weight_t weights1[HIDDEN_FEATURES][IN_FEATURES],
    const acc_t bias1[HIDDEN_FEATURES],        // INT32 bias
    const weight_t weights2[OUT_FEATURES][HIDDEN_FEATURES],
    const acc_t bias2[OUT_FEATURES],           // INT32 bias
    data_t output[NUM_NODES][OUT_FEATURES],
    quant_scale_t scale_in,
    quant_scale_t scale_w1,
    quant_scale_t scale_hidden,
    quant_scale_t scale_w2,
    quant_scale_t scale_out
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
    #pragma HLS INTERFACE mode=ap_none port=scale_in
    #pragma HLS INTERFACE mode=ap_none port=scale_w1
    #pragma HLS INTERFACE mode=ap_none port=scale_hidden
    #pragma HLS INTERFACE mode=ap_none port=scale_w2
    #pragma HLS INTERFACE mode=ap_none port=scale_out
    #pragma HLS INTERFACE mode=ap_ctrl_none port=return
    
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
        output,
        scale_in,
        scale_w1,
        scale_hidden,
        scale_w2,
        scale_out
    );
}
