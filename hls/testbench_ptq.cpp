/**
 * Testbench for GraphSAGE INT8 PTQ Implementation
 * 
 * This testbench loads PTQ quantized test vectors and runs the INT8 GraphSAGE.
 * It compares against Python PTQ reference outputs.
 * 
 * Configuration matches reduced model:
 *   - IN_FEATURES = 16 (after projection)
 *   - HIDDEN_FEATURES = 24
 *   - OUT_FEATURES = 7
 *   - NUM_NODES = 8 (test subgraph)
 *   - INT8 quantized weights/activations
 *   - Symmetric quantization (zero_point = 0)
 */

#include "graphsage_layer_ptq.h"
#include "graphsage_layer_ptq.cpp"  // Include implementation for C simulation
#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <cstdio>

using namespace std;

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Load matrix from text file (float format for adjacency matrix)
 */
template<int ROWS, int COLS>
bool load_float_matrix(const char* filename, float matrix[ROWS][COLS]) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "ERROR: Cannot open file: " << filename << endl;
        return false;
    }

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            if (!(file >> matrix[i][j])) {
                cout << "ERROR: Failed to read element [" << i << "," << j << "] from " << filename << endl;
                return false;
            }
        }
    }

    file.close();
    cout << "Loaded " << filename << " (" << ROWS << "x" << COLS << ")" << endl;
    return true;
}

/**
 * Load INT8 matrix from text file
 */
template<int ROWS, int COLS>
bool load_int8_matrix(const char* filename, int8_t matrix[ROWS][COLS]) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "ERROR: Cannot open file: " << filename << endl;
        return false;
    }

    int value;
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            if (!(file >> value)) {
                cout << "ERROR: Failed to read element [" << i << "," << j << "] from " << filename << endl;
                return false;
            }
            matrix[i][j] = (int8_t)value;
        }
    }

    file.close();
    cout << "Loaded " << filename << " (" << ROWS << "x" << COLS << ")" << endl;
    return true;
}

/**
 * Load INT8 vector from text file
 */
template<int SIZE>
bool load_int8_vector(const char* filename, int8_t vec[SIZE]) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "ERROR: Cannot open file: " << filename << endl;
        return false;
    }

    int value;
    for (int i = 0; i < SIZE; i++) {
        if (!(file >> value)) {
            cout << "ERROR: Failed to read element [" << i << "] from " << filename << endl;
            return false;
        }
        vec[i] = (int8_t)value;
    }

    file.close();
    cout << "Loaded " << filename << " (" << SIZE << " elements)" << endl;
    return true;
}

/**
 * Load INT32 vector from text file (for bias in accumulator scale)
 */
template<int SIZE>
bool load_int32_vector(const char* filename, int32_t vec[SIZE]) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "ERROR: Cannot open file: " << filename << endl;
        return false;
    }

    for (int i = 0; i < SIZE; i++) {
        if (!(file >> vec[i])) {
            cout << "ERROR: Failed to read element [" << i << "] from " << filename << endl;
            return false;
        }
    }

    file.close();
    cout << "Loaded " << filename << " (" << SIZE << " elements)" << endl;
    return true;
}

/**
 * Load quantization scales from file
 */
bool load_scales(const char* filename,
                float &scale_in,
                float &scale_w1,
                float &scale_w2,
                float &scale_hidden,
                float &scale_out) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "ERROR: Cannot open file: " << filename << endl;
        return false;
    }

    string label;
    // Format: "scale_in: 0.00486..."
    // File order: scale_in, scale_w1, scale_w2, scale_hidden, scale_out
    file >> label >> scale_in;
    file >> label >> scale_w1;
    file >> label >> scale_w2;
    file >> label >> scale_hidden;
    file >> label >> scale_out;

    file.close();
    
    cout << "Loaded quantization scales:" << endl;
    cout << "  scale_in:     " << scale_in << endl;
    cout << "  scale_hidden: " << scale_hidden << endl;
    cout << "  scale_w1:     " << scale_w1 << endl;
    cout << "  scale_w2:     " << scale_w2 << endl;
    cout << "  scale_out:    " << scale_out << endl;
    
    return true;
}

/**
 * Print INT8 matrix (for debugging)
 */
template<int ROWS, int COLS>
void print_int8_matrix(const char* name, int8_t matrix[ROWS][COLS], int max_rows = 3, int max_cols = 8) {
    cout << name << " (showing " << max_rows << "x" << max_cols << "):" << endl;
    for (int i = 0; i < max_rows && i < ROWS; i++) {
        cout << "  ";
        for (int j = 0; j < max_cols && j < COLS; j++) {
            cout << setw(5) << (int)matrix[i][j];
        }
        if (max_cols < COLS) cout << " ...";
        cout << endl;
    }
    if (max_rows < ROWS) cout << "  ..." << endl;
    cout << endl;
}

/**
 * Compare INT8 outputs with reference (allowing for quantization error)
 */
template<int ROWS, int COLS>
bool compare_int8_outputs(int8_t result[ROWS][COLS], int8_t reference[ROWS][COLS], int tolerance = 3) {
    bool pass = true;
    int max_error = 0;
    int total_errors = 0;
    int total_nonzero = 0;  // Count non-exact matches
    
    cout << "Comparing outputs (tolerance = ±" << tolerance << " LSB):" << endl;
    
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            int error = abs((int)result[i][j] - (int)reference[i][j]);
            if (error > max_error) {
                max_error = error;
            }
            if (error > 0) {
                total_nonzero++;
                // Print all non-zero errors for debugging
                cout << "  DIFF [" << i << "," << j << "]: "
                     << "result=" << (int)result[i][j]
                     << " reference=" << (int)reference[i][j]
                     << " error=" << error << endl;
            }
            if (error > tolerance) {
                total_errors++;
                pass = false;
            }
        }
    }
    
    cout << "Max error: " << max_error << " LSB" << endl;
    cout << "Non-zero errors: " << total_nonzero << " / " << (ROWS * COLS) << endl;
    cout << "Errors exceeding tolerance: " << total_errors << " / " << (ROWS * COLS) << endl;
    
    return pass;
}

// ============================================================================
// Main Testbench
// ============================================================================

int main() {
    cout << "========================================" << endl;
    cout << "GraphSAGE INT8 Quantized Testbench (PTQ)" << endl;
    cout << "========================================" << endl << endl;

    // Test vector directory (absolute paths from project root)
    const char* test_dir = "/home/pelayo/work/simple-gnn/build/test_vectors_ptq";

    // ========== Declare arrays ==========" 
    // Adjacency matrix (still float for precision)
    static float adj_matrix[NUM_NODES][NUM_NODES];
    
    // Quantized inputs and outputs (INT8)
    static int8_t input[NUM_NODES][IN_FEATURES];
    static int8_t output[NUM_NODES][OUT_FEATURES];
    static int8_t reference_output[NUM_NODES][OUT_FEATURES];
    
    // Quantized weights (INT8) and biases (INT32)
    static int8_t weights1[HIDDEN_FEATURES][IN_FEATURES];
    static int32_t bias1[HIDDEN_FEATURES];     // INT32 bias in accumulator scale
    static int8_t weights2[OUT_FEATURES][HIDDEN_FEATURES];
    static int32_t bias2[OUT_FEATURES];        // INT32 bias in accumulator scale
    
    // Quantization scales (no bias scales needed - bias is in accumulator scale)
    float scale_in, scale_w1, scale_w2, scale_hidden, scale_out;

    // ========== Load test vectors ==========
    cout << "Loading test vectors..." << endl;
    
    char filename[512];
    
    // Load adjacency matrix (float)
    sprintf(filename, "%s/adj_matrix.txt", test_dir);
    if (!load_float_matrix<NUM_NODES, NUM_NODES>(filename, adj_matrix)) {
        return 1;
    }

    // Load quantization scales
    sprintf(filename, "%s/scales.txt", test_dir);
    if (!load_scales(filename, scale_in, scale_w1, scale_w2, scale_hidden, scale_out)) {
        return 1;
    }

    // Load quantized input (INT8)
    sprintf(filename, "%s/network_input.txt", test_dir);
    if (!load_int8_matrix<NUM_NODES, IN_FEATURES>(filename, input)) {
        return 1;
    }

    // Load quantized weights (INT8) and biases (INT32)
    sprintf(filename, "%s/weights_layer1.txt", test_dir);
    if (!load_int8_matrix<HIDDEN_FEATURES, IN_FEATURES>(filename, weights1)) {
        return 1;
    }
    sprintf(filename, "%s/bias_layer1.txt", test_dir);
    if (!load_int32_vector<HIDDEN_FEATURES>(filename, bias1)) {
        return 1;
    }
    sprintf(filename, "%s/weights_layer2.txt", test_dir);
    if (!load_int8_matrix<OUT_FEATURES, HIDDEN_FEATURES>(filename, weights2)) {
        return 1;
    }
    sprintf(filename, "%s/bias_layer2.txt", test_dir);
    if (!load_int32_vector<OUT_FEATURES>(filename, bias2)) {
        return 1;
    }

    // Load reference output (INT8)
    sprintf(filename, "%s/network_output_reference.txt", test_dir);
    if (!load_int8_matrix<NUM_NODES, OUT_FEATURES>(filename, reference_output)) {
        return 1;
    }

    cout << endl;

    // ========== Print sample data ==========
    print_int8_matrix<NUM_NODES, IN_FEATURES>("Input", input);
    print_int8_matrix<NUM_NODES, OUT_FEATURES>("Reference Output", reference_output);

    // ========== Run GraphSAGE network ==========
    cout << "Running PTQ quantized GraphSAGE network..." << endl;
    
    graphsage_network_ptq(
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

    cout << "Inference complete!" << endl << endl;

    // ========== Compare results ==========
    print_int8_matrix<NUM_NODES, OUT_FEATURES>("Network Output", output);

    bool pass = compare_int8_outputs<NUM_NODES, OUT_FEATURES>(
        output, reference_output, 3  // Allow ±3 LSB tolerance for PTQ quantization noise
    );

    // ========== Test result ==========
    cout << endl << "========================================" << endl;
    if (pass) {
        cout << "TEST PASSED! ✓" << endl;
        cout << "Quantized output matches reference within tolerance." << endl;
        return 0;
    } else {
        cout << "TEST FAILED! ✗" << endl;
        cout << "Quantized output differs from reference." << endl;
        return 1;
    }
}
