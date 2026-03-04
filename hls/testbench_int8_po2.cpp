/**
 * Testbench for Pure INT8 with Power-of-Two Scales GraphSAGE HLS Implementation
 * 
 * Loads integer-only test vectors and compares against Python reference.
 * Uses the same parameters as INT8, but with PO2 shift-based scaling.
 * 
 * Key differences from testbench_int8.cpp:
 *   - No scale_fp parameters needed (shift amounts are compile-time constants)
 *   - Should produce very similar results to INT8 (with small PO2 quantization error)
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include "graphsage_layer_int8_po2.h"
#include "graphsage_layer_int8_po2.cpp"  // Include implementation for C simulation

// If K_VALUE isn't provided via build flags, compute it from K_BITS
#ifndef K_VALUE
const int K_VALUE = (1 << K_BITS);
#endif

// ============================================================================
// Stringify macros - convert macro values to string literals
// ============================================================================
#define STR_IMPL(x) #x
#define STR(x) STR_IMPL(x)

// ============================================================================
// Test vector paths - can be overridden via compiler flags for DSE
// Default paths are relative to csim build dir: build/hls/graphsage_int8_po2/solution1/csim/build/
// Note: No trailing slash - the code adds "/" when building full paths
// ============================================================================
#ifndef INT8_PARAMS_DIR
#define INT8_PARAMS_DIR ../../../../../../build/weights_ptq_int8
#endif

#ifndef TEST_VECTORS_DIR
#define TEST_VECTORS_DIR ../../../../../../build/test_vectors_ptq_float
#endif

#ifndef INT8_REFERENCE_DIR
#define INT8_REFERENCE_DIR ../../../../../../build/test_vectors_ptq_int8
#endif

#ifndef PO2_REFERENCE_DIR
#define PO2_REFERENCE_DIR ../../../../../../build/test_vectors_ptq_int8_po2
#endif

// ============================================================================
// Helper functions
// ============================================================================

template<typename T>
void load_1d_array(const char* filename, T* arr, int size) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot open " << filename << std::endl;
        exit(1);
    }
    for (int i = 0; i < size; i++) {
        int val;
        file >> val;
        arr[i] = (T)val;
    }
    file.close();
    std::cout << "  Loaded " << filename << " (" << size << " elements)" << std::endl;
}

template<typename T>
void load_2d_array(const char* filename, T* arr, int rows, int cols) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot open " << filename << std::endl;
        exit(1);
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int val;
            file >> val;
            arr[i * cols + j] = (T)val;
        }
    }
    file.close();
    std::cout << "  Loaded " << filename << " (" << rows << "x" << cols << ")" << std::endl;
}

void load_adj_matrix(const char* filename, adj_t adj[NUM_NODES][NUM_NODES]) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot open " << filename << std::endl;
        exit(1);
    }
    for (int i = 0; i < NUM_NODES; i++) {
        for (int j = 0; j < NUM_NODES; j++) {
            int val;
            file >> val;
            adj[i][j] = (adj_t)val;
        }
    }
    file.close();
    std::cout << "  Loaded " << filename << " (" << NUM_NODES << "x" << NUM_NODES << ")" << std::endl;
}

// ============================================================================
// Main testbench
// ============================================================================

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "     PURE INT8 PO2 (Power-of-Two Scales) GraphSAGE HLS TB" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  NUM_NODES = " << NUM_NODES << std::endl;
    std::cout << "  IN_FEATURES = " << IN_FEATURES << std::endl;
    std::cout << "  HIDDEN_FEATURES = " << HIDDEN_FEATURES << std::endl;
    std::cout << "  OUT_FEATURES = " << OUT_FEATURES << std::endl;
    std::cout << "  M_BITS = " << M_BITS << " (fractional bits)" << std::endl;
    std::cout << "  K_BITS = " << K_BITS << " (adjacency scale = " << (1 << K_BITS) << ")" << std::endl;
    std::cout << std::endl;
    std::cout << "PO2 Shift Amounts (compile-time constants):" << std::endl;
    std::cout << "  BETA1_SHIFT = " << BETA1_SHIFT << std::endl;
    std::cout << "  BETA2_SHIFT = " << BETA2_SHIFT << std::endl;
    std::cout << "  EFF_SCALE1_SHIFT = " << EFF_SCALE1_SHIFT << std::endl;
    std::cout << "  EFF_SCALE2_SHIFT = " << EFF_SCALE2_SHIFT << std::endl;
    std::cout << std::endl;

    // ========== Declare arrays ==========
    adj_t adj_matrix[NUM_NODES][NUM_NODES];
    data_t input[NUM_NODES][IN_FEATURES];
    weight_t weights1[HIDDEN_FEATURES][IN_FEATURES];
    bias_t bias1[HIDDEN_FEATURES];
    weight_t weights2[OUT_FEATURES][HIDDEN_FEATURES];
    bias_t bias2[OUT_FEATURES];
    data_t output[NUM_NODES][OUT_FEATURES];
    data_t reference[NUM_NODES][OUT_FEATURES];

    // ========== Load test vectors ==========
    std::cout << "Loading test vectors..." << std::endl;
    
    // Load INT16 adjacency matrix
    std::string adj_path = std::string(STR(INT8_PARAMS_DIR)) + "/adj_matrix_int16.txt";
    load_adj_matrix(adj_path.c_str(), adj_matrix);
    
    // Load INT8 input
    std::string input_path = std::string(STR(TEST_VECTORS_DIR)) + "/network_input.txt";
    load_2d_array(input_path.c_str(), (data_t*)input, NUM_NODES, IN_FEATURES);
    
    // Load INT8 weights
    std::string w1_path = std::string(STR(TEST_VECTORS_DIR)) + "/weights_layer1.txt";
    load_2d_array(w1_path.c_str(), (weight_t*)weights1, HIDDEN_FEATURES, IN_FEATURES);
    
    std::string w2_path = std::string(STR(TEST_VECTORS_DIR)) + "/weights_layer2.txt";
    load_2d_array(w2_path.c_str(), (weight_t*)weights2, OUT_FEATURES, HIDDEN_FEATURES);
    
    // Load INT32 biases
    std::string b1_path = std::string(STR(INT8_PARAMS_DIR)) + "/bias_layer1_int32.txt";
    load_1d_array(b1_path.c_str(), bias1, HIDDEN_FEATURES);
    
    std::string b2_path = std::string(STR(INT8_PARAMS_DIR)) + "/bias_layer2_int32.txt";
    load_1d_array(b2_path.c_str(), bias2, OUT_FEATURES);
    
    // Load reference output (from Python PO2 emulator - should match exactly!)
    std::string ref_path = std::string(STR(PO2_REFERENCE_DIR)) + "/network_output_int8_po2_reference.txt";
    load_2d_array(ref_path.c_str(), (data_t*)reference, NUM_NODES, OUT_FEATURES);

    // ========== Print input sample ==========
    std::cout << std::endl << "Input sample (node 6):" << std::endl << "  [";
    for (int f = 0; f < IN_FEATURES; f++) {
        std::cout << (int)input[6][f];
        if (f < IN_FEATURES - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // ========== Run HLS function ==========
    std::cout << std::endl << "============================================================" << std::endl;
    std::cout << "Running Pure INT8 PO2 GraphSAGE..." << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "(Note: No scale_fp parameters - using compile-time shift constants)" << std::endl;
    
    // PO2 version has fewer parameters - no scale_fp values needed!
    graphsage_int8_po2(
        adj_matrix,
        input,
        weights1,
        bias1,
        weights2,
        bias2,
        output
    );

    // ========== Compare results ==========
    std::cout << std::endl << "============================================================" << std::endl;
    std::cout << "Comparing HLS PO2 output vs Python INT8 reference..." << std::endl;
    std::cout << "(Note: Some difference expected due to PO2 scale approximation)" << std::endl;
    std::cout << "============================================================" << std::endl;
    
    int max_error = 0;
    int total_errors = 0;
    int total_elements = NUM_NODES * OUT_FEATURES;
    long long sum_abs_error = 0;
    
    for (int n = 0; n < NUM_NODES; n++) {
        std::cout << "Node " << n << ":" << std::endl;
        std::cout << "  HLS:    [";
        for (int o = 0; o < OUT_FEATURES; o++) {
            std::cout << std::setw(4) << (int)output[n][o];
            if (o < OUT_FEATURES - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "  Ref:    [";
        for (int o = 0; o < OUT_FEATURES; o++) {
            std::cout << std::setw(4) << (int)reference[n][o];
            if (o < OUT_FEATURES - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "  Diff:   [";
        for (int o = 0; o < OUT_FEATURES; o++) {
            int diff = (int)output[n][o] - (int)reference[n][o];
            std::cout << std::setw(4) << diff;
            if (o < OUT_FEATURES - 1) std::cout << ", ";
            
            if (diff != 0) total_errors++;
            if (abs(diff) > max_error) max_error = abs(diff);
            sum_abs_error += abs(diff);
        }
        std::cout << "]" << std::endl;
    }

    // ========== Summary ==========
    double mean_abs_error = (double)sum_abs_error / total_elements;
    
    std::cout << std::endl << "============================================================" << std::endl;
    std::cout << "SUMMARY" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Max error: " << max_error << " LSB" << std::endl;
    std::cout << "Mean absolute error: " << std::fixed << std::setprecision(2) << mean_abs_error << " LSB" << std::endl;
    std::cout << "Total differences: " << total_errors << "/" << total_elements << std::endl;
    
    std::cout << std::endl;
    std::cout << "HLS PO2 vs Python PO2 Emulator Comparison:" << std::endl;
    std::cout << "  - Using PO2 reference from generate_test_vectors_ptq_int8_po2.py" << std::endl;
    std::cout << "  - Both HLS and Python use same PO2 bit-shifts (should match exactly)" << std::endl;
    std::cout << "  - Expected: 0-5 LSB error (bit-exact or rounding differences)" << std::endl;
    
    // Now that we're comparing PO2 vs PO2, we expect very small errors
    if (max_error == 0) {
        std::cout << std::endl << "*** PERFECT MATCH! HLS PO2 matches Python PO2 exactly. ***" << std::endl;
        return 0;
    } else if (max_error <= 5) {
        std::cout << std::endl << "*** GOOD: Max " << max_error << " LSB error (acceptable rounding differences) ***" << std::endl;
        return 0;
    } else if (max_error <= 15) {
        std::cout << std::endl << "*** WARNING: Max " << max_error << " LSB error - investigate implementation difference ***" << std::endl;
        return 1;
    } else {
        std::cout << std::endl << "*** ERROR: Errors exceed 15 LSB - HLS PO2 does not match Python PO2 ***" << std::endl;
        return 1;
    }
}
