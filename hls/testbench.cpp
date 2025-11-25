/**
 * Testbench for GraphSAGE HLS implementation
 * Loads test vectors from Python and validates HLS implementation
 */

#include "graphsage_layer.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>

// ============================================================================
// Helper Functions for File I/O
// ============================================================================

/**
 * Load matrix from text file
 */
template<typename T>
bool load_matrix(const char* filename, T* matrix, int rows, int cols) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (!(file >> matrix[i * cols + j])) {
                std::cerr << "Error reading from " << filename << std::endl;
                return false;
            }
        }
    }

    file.close();
    return true;
}

/**
 * Load vector from text file
 */
template<typename T>
bool load_vector(const char* filename, T* vec, int size) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }

    for (int i = 0; i < size; i++) {
        if (!(file >> vec[i])) {
            std::cerr << "Error reading from " << filename << std::endl;
            return false;
        }
    }

    file.close();
    return true;
}

/**
 * Compare two matrices with tolerance
 */
bool compare_matrices(
    const data_t* output,
    const data_t* reference,
    int rows, int cols,
    int tolerance = 2  // Allow up to 2 quantization levels difference
) {
    int errors = 0;
    int max_error = 0;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int idx = i * cols + j;
            int diff = abs((int)output[idx] - (int)reference[idx]);
            if (diff > tolerance) {
                errors++;
                if (diff > max_error) {
                    max_error = diff;
                }
                if (errors <= 10) {  // Print first 10 errors
                    std::cout << "Mismatch at [" << i << "][" << j << "]: "
                             << "HLS=" << (int)output[idx]
                             << ", Reference=" << (int)reference[idx]
                             << ", Diff=" << diff << std::endl;
                }
            }
        }
    }

    std::cout << "Comparison results:" << std::endl;
    std::cout << "  Total elements: " << rows * cols << std::endl;
    std::cout << "  Errors: " << errors << std::endl;
    std::cout << "  Max error: " << max_error << std::endl;
    std::cout << "  Error rate: " << (float)errors / (rows * cols) * 100.0f << "%" << std::endl;

    return errors == 0;
}

// ============================================================================
// Test Functions
// ============================================================================

/**
 * Test single GraphSAGE layer - DISABLED (function removed for simplicity)
 */
/*
bool test_single_layer() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Testing Single GraphSAGE Layer" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    const int num_nodes = 32;
    const int in_features = 16;
    const int out_features = 16;

    // Allocate arrays
    static scale_t adj_matrix[MAX_NODES][MAX_NODES];
    static data_t input[MAX_NODES][MAX_FEATURES_IN];
    static data_t weights[MAX_FEATURES_OUT][MAX_FEATURES_IN];
    static acc_t bias[MAX_FEATURES_OUT];
    static data_t output[MAX_NODES][MAX_FEATURES_OUT];
    static data_t reference[MAX_NODES][MAX_FEATURES_OUT];

    // Load test vectors
    std::cout << "Loading test vectors..." << std::endl;

    if (!load_matrix("../../../../../test_vectors/adj_matrix.txt",
                     (scale_t*)adj_matrix, num_nodes, num_nodes)) {
        return false;
    }

    if (!load_matrix("../../../../../test_vectors/input_features.txt",
                     (data_t*)input, num_nodes, in_features)) {
        return false;
    }

    if (!load_matrix("../../../../../test_vectors/weights.txt",
                     (data_t*)weights, out_features, in_features)) {
        return false;
    }

    if (!load_vector("../../../../../test_vectors/bias.txt",
                     (acc_t*)bias, out_features)) {
        return false;
    }

    if (!load_matrix("../../../../../test_vectors/output_reference.txt",
                     (data_t*)reference, num_nodes, out_features)) {
        return false;
    }

    std::cout << "Test vectors loaded successfully!" << std::endl;

    // Run HLS function
    std::cout << "Running HLS implementation..." << std::endl;

    scale_t scale_in = 0.1f;
    scale_t scale_weight = 0.05f;
    scale_t scale_out = 0.1f;

    graphsage_layer(
        adj_matrix, input,
        weights, bias, output,
        num_nodes, in_features, out_features,
        scale_in, scale_weight, scale_out
    );

    std::cout << "HLS execution complete!" << std::endl;

    // Compare results
    std::cout << "\nComparing results with reference..." << std::endl;
    bool passed = compare_matrices((data_t*)output, (data_t*)reference,
                                   num_nodes, out_features, 2);

    if (passed) {
        std::cout << "\n[PASS] Single layer test passed!" << std::endl;
    } else {
        std::cout << "\n[FAIL] Single layer test failed!" << std::endl;
    }

    return passed;
}
*/

/**
 * Test full two-layer GraphSAGE network
 */
bool test_two_layer_network() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Testing Two-Layer GraphSAGE Network" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    const int num_nodes = 4;  // Number of nodes in test vectors

    // Allocate arrays (zero-initialize for safety)
    static scale_t adj_matrix[MAX_NODES][MAX_NODES] = {0};
    static data_t  input[MAX_NODES][MAX_FEATURES_IN] = {0};

    static data_t  weights1[MAX_FEATURES_HIDDEN][MAX_FEATURES_IN] = {0};
    static acc_t   bias1[MAX_FEATURES_HIDDEN] = {0};

    static data_t  weights2[MAX_FEATURES_OUT][MAX_FEATURES_HIDDEN] = {0};
    static acc_t   bias2[MAX_FEATURES_OUT] = {0};

    static data_t  output[MAX_NODES][MAX_FEATURES_OUT] = {0};
    static data_t  reference[MAX_NODES][MAX_FEATURES_OUT] = {0};

    // Load test vectors
    std::cout << "Loading test vectors..." << std::endl;

    if (!load_matrix("../../../../../test_vectors/adj_matrix.txt",
                     (scale_t*)adj_matrix, num_nodes, num_nodes)) {
        return false;
    }

    if (!load_matrix("../../../../../test_vectors/network_input.txt",
                     (data_t*)input, num_nodes, MAX_FEATURES_IN)) {
        return false;
    }

    if (!load_matrix("../../../../../test_vectors/weights_layer1.txt",
                     (data_t*)weights1, MAX_FEATURES_HIDDEN, MAX_FEATURES_IN)) {
        return false;
    }

    if (!load_vector("../../../../../test_vectors/bias_layer1.txt",
                     (acc_t*)bias1, MAX_FEATURES_HIDDEN)) {
        return false;
    }

    if (!load_matrix("../../../../../test_vectors/weights_layer2.txt",
                     (data_t*)weights2, MAX_FEATURES_OUT, MAX_FEATURES_HIDDEN)) {
        return false;
    }

    if (!load_vector("../../../../../test_vectors/bias_layer2.txt",
                     (acc_t*)bias2, MAX_FEATURES_OUT)) {
        return false;
    }

    if (!load_matrix("../../../../../test_vectors/network_output_reference.txt",
                     (data_t*)reference, num_nodes, MAX_FEATURES_OUT)) {
        return false;
    }

    std::cout << "Test vectors loaded successfully!" << std::endl;

    // Load scales
    scale_t scale_in, scale_w1, scale_w2, scale_hidden, scale_out;
    std::ifstream scales_file("../../../../../test_vectors/scales.txt");
    if (!scales_file.is_open()) {
        std::cerr << "Error: Cannot open scales.txt" << std::endl;
        return false;
    }

    std::string label;
    scales_file >> label >> scale_in;
    scales_file >> label >> scale_w1;
    scales_file >> label >> scale_w2;
    scales_file >> label >> scale_hidden;
    scales_file >> label >> scale_out;
    scales_file.close();

    std::cout << "Loaded scales:" << std::endl;
    std::cout << "  scale_in    : " << scale_in << std::endl;
    std::cout << "  scale_w1    : " << scale_w1 << std::endl;
    std::cout << "  scale_w2    : " << scale_w2 << std::endl;
    std::cout << "  scale_hidden: " << scale_hidden << std::endl;
    std::cout << "  scale_out   : " << scale_out << std::endl;

    // Print sample input data
    std::cout << "\n" << std::string(70, '-') << std::endl;
    std::cout << "INPUT DATA SAMPLES" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "Input [0,:5]: ";
    for (int i = 0; i < 5; i++) {
        std::cout << (int)input[0][i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Weights1 [0,:5]: ";
    for (int i = 0; i < 5; i++) {
        std::cout << (int)weights1[0][i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Bias1 [0:5]: ";
    for (int i = 0; i < 5; i++) {
        std::cout << bias1[i] << " ";
    }
    std::cout << std::endl;

    // Run HLS function
    std::cout << "\n" << std::string(70, '-') << std::endl;
    std::cout << "Running HLS implementation..." << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    graphsage_network(
        adj_matrix, input,
        weights1, bias1, scale_w1,
        weights2, bias2, scale_w2,
        output, num_nodes,
        scale_in, scale_hidden, scale_out
    );

    std::cout << "HLS execution complete!" << std::endl;
    
    // Print sample output data
    std::cout << "\n" << std::string(70, '-') << std::endl;
    std::cout << "OUTPUT DATA SAMPLES" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "HLS output [0,:]: ";
    for (int i = 0; i < MAX_FEATURES_OUT; i++) {
        std::cout << (int)output[0][i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Reference  [0,:]: ";
    for (int i = 0; i < MAX_FEATURES_OUT; i++) {
        std::cout << (int)reference[0][i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "HLS output [1,:]: ";
    for (int i = 0; i < MAX_FEATURES_OUT; i++) {
        std::cout << (int)output[1][i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Reference  [1,:]: ";
    for (int i = 0; i < MAX_FEATURES_OUT; i++) {
        std::cout << (int)reference[1][i] << " ";
    }
    std::cout << std::endl;

    // Compare results
    std::cout << "\nComparing results with reference..." << std::endl;
    bool passed = compare_matrices(
        (data_t*)output, (data_t*)reference,
        num_nodes, MAX_FEATURES_OUT, 3
    );

    if (passed) {
        std::cout << "\n[PASS] Two-layer network test passed!" << std::endl;
    } else {
        std::cout << "\n[FAIL] Two-layer network test failed!" << std::endl;
    }

    return passed;
}


// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "\n";
    std::cout << "================================================" << std::endl;
    std::cout << "  GraphSAGE HLS Implementation Testbench" << std::endl;
    std::cout << "================================================" << std::endl;

    bool all_passed = true;

    // Run tests
    // all_passed &= test_single_layer();
    all_passed &= test_two_layer_network();

    // Final result
    std::cout << "\n" << std::string(60, '=') << std::endl;
    if (all_passed) {
        std::cout << "[PASS] All tests passed!" << std::endl;
    } else {
        std::cout << "[FAIL] Some tests failed!" << std::endl;
    }
    std::cout << std::string(60, '=') << std::endl;

    return all_passed ? 0 : 1;
}
