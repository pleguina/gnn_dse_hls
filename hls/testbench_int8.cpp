/**
 * Testbench for Pure INT8 GraphSAGE HLS Implementation
 * 
 * Loads integer-only test vectors and compares against Python reference.
 * Uses fixed-point parameters from build/weights_ptq_int8/int8_params.json
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include "graphsage_layer_int8.h"
#include "graphsage_layer_int8.cpp"  // Include implementation for C simulation

// ============================================================================
// Test vector paths (relative to csim build dir: build/hls/graphsage_int8/solution1/csim/build/)
// ============================================================================
const char* INT8_PARAMS_DIR = "../../../../../../build/weights_ptq_int8/";
const char* TEST_VECTORS_DIR = "../../../../../../build/test_vectors_ptq_float/";
const char* INT8_REFERENCE_DIR = "../../../../../../build/test_vectors_ptq_int8/";

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
    std::cout << "     PURE INT8 GraphSAGE HLS Testbench" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  NUM_NODES = " << NUM_NODES << std::endl;
    std::cout << "  IN_FEATURES = " << IN_FEATURES << std::endl;
    std::cout << "  HIDDEN_FEATURES = " << HIDDEN_FEATURES << std::endl;
    std::cout << "  OUT_FEATURES = " << OUT_FEATURES << std::endl;
    std::cout << "  M_BITS = " << M_BITS << " (fractional bits)" << std::endl;
    std::cout << "  K_BITS = " << K_BITS << " (adjacency scale = " << K_VALUE << ")" << std::endl;
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

    // Fixed-point scale parameters
    scale_fp_t beta1_fp, beta2_fp, eff_scale1_fp, eff_scale2_fp;

    // ========== Load test vectors ==========
    std::cout << "Loading test vectors..." << std::endl;
    
    // Load INT16 adjacency matrix
    std::string adj_path = std::string(INT8_PARAMS_DIR) + "adj_matrix_int16.txt";
    load_adj_matrix(adj_path.c_str(), adj_matrix);
    
    // Load INT8 input
    std::string input_path = std::string(TEST_VECTORS_DIR) + "network_input.txt";
    load_2d_array(input_path.c_str(), (data_t*)input, NUM_NODES, IN_FEATURES);
    
    // Load INT8 weights
    std::string w1_path = std::string(TEST_VECTORS_DIR) + "weights_layer1.txt";
    load_2d_array(w1_path.c_str(), (weight_t*)weights1, HIDDEN_FEATURES, IN_FEATURES);
    
    std::string w2_path = std::string(TEST_VECTORS_DIR) + "weights_layer2.txt";
    load_2d_array(w2_path.c_str(), (weight_t*)weights2, OUT_FEATURES, HIDDEN_FEATURES);
    
    // Load INT32 biases
    std::string b1_path = std::string(INT8_PARAMS_DIR) + "bias_layer1_int32.txt";
    load_1d_array(b1_path.c_str(), bias1, HIDDEN_FEATURES);
    
    std::string b2_path = std::string(INT8_PARAMS_DIR) + "bias_layer2_int32.txt";
    load_1d_array(b2_path.c_str(), bias2, OUT_FEATURES);
    
    // Load reference output
    std::string ref_path = std::string(INT8_REFERENCE_DIR) + "network_output_int8_reference.txt";
    load_2d_array(ref_path.c_str(), (data_t*)reference, NUM_NODES, OUT_FEATURES);

    // ========== Load fixed-point parameters ==========
    std::cout << std::endl << "Loading fixed-point parameters..." << std::endl;
    
    // These values should match int8_params.json
    // For M=20: beta1_fp=14, beta2_fp=4096, eff_scale1_fp=11206, eff_scale2_fp=11061
    // For M=24: beta1_fp=218, beta2_fp=4096, eff_scale1_fp=179294, eff_scale2_fp=176976
    
    #if M_BITS == 20
        beta1_fp = 14;
        beta2_fp = 4096;       // = K (since scale_hidden/scale_hidden = 1)
        eff_scale1_fp = 11206;
        eff_scale2_fp = 11061;
        std::cout << "  Using M=20 parameters (13 LSB max error expected)" << std::endl;
    #elif M_BITS == 24
        beta1_fp = 218;
        beta2_fp = 4096;       // = K
        eff_scale1_fp = 179294;
        eff_scale2_fp = 176976;
        std::cout << "  Using M=24 parameters (0 LSB error expected)" << std::endl;
    #else
        #error "Unsupported M_BITS value. Use 20 or 24."
    #endif
    
    std::cout << "  beta1_fp = " << beta1_fp << std::endl;
    std::cout << "  beta2_fp = " << beta2_fp << std::endl;
    std::cout << "  eff_scale1_fp = " << eff_scale1_fp << std::endl;
    std::cout << "  eff_scale2_fp = " << eff_scale2_fp << std::endl;

    // ========== Print input sample ==========
    std::cout << std::endl << "Input sample (node 6):" << std::endl << "  [";
    for (int f = 0; f < IN_FEATURES; f++) {
        std::cout << (int)input[6][f];
        if (f < IN_FEATURES - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // ========== Run HLS function ==========
    std::cout << std::endl << "============================================================" << std::endl;
    std::cout << "Running Pure INT8 GraphSAGE..." << std::endl;
    std::cout << "============================================================" << std::endl;
    
    graphsage_int8(
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

    // ========== Compare results ==========
    std::cout << std::endl << "============================================================" << std::endl;
    std::cout << "Comparing HLS output vs Python INT8 reference..." << std::endl;
    std::cout << "============================================================" << std::endl;
    
    int max_error = 0;
    int total_errors = 0;
    int total_elements = NUM_NODES * OUT_FEATURES;
    
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
        }
        std::cout << "]" << std::endl;
    }

    // ========== Summary ==========
    std::cout << std::endl << "============================================================" << std::endl;
    std::cout << "SUMMARY" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Max error: " << max_error << " LSB" << std::endl;
    std::cout << "Total errors: " << total_errors << "/" << total_elements << std::endl;
    
    if (max_error == 0) {
        std::cout << std::endl << "*** PERFECT MATCH! HLS matches Python INT8 exactly. ***" << std::endl;
        return 0;
    } else if (max_error <= 1) {
        std::cout << std::endl << "*** EXCELLENT: Max 1 LSB error (rounding difference) ***" << std::endl;
        return 0;
    } else {
        std::cout << std::endl << "*** WARNING: Errors exceed expected tolerance ***" << std::endl;
        return 1;
    }
}
