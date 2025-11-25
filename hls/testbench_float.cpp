/**
 * Testbench for GraphSAGE FLOAT Implementation - Reduced Model
 * 
 * This testbench loads test vectors and runs the FLOAT version of GraphSAGE.
 * It should match PyG output exactly (no quantization errors).
 * 
 * Configuration matches reduced model:
 *   - IN_FEATURES = 16 (after projection)
 *   - HIDDEN_FEATURES = 24
 *   - OUT_FEATURES = 7
 *   - NUM_NODES = 8 (test subgraph)
 */

#include "graphsage_layer_float.h"
#include "graphsage_layer_float.cpp"  // Include implementation for C simulation
#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>

using namespace std;

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Load matrix from text file (float format)
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
 * Load INT8 matrix and dequantize to float
 */
template<int ROWS, int COLS>
bool load_and_dequantize(const char* filename, float matrix[ROWS][COLS], float scale) {
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
            matrix[i][j] = (float)value * scale;
        }
    }

    file.close();
    cout << "Loaded and dequantized " << filename << " (" << ROWS << "x" << COLS << ")" << endl;
    return true;
}

/**
 * Load INT32 vector and dequantize to float (for biases)
 */
template<int SIZE>
bool load_bias_and_dequantize(const char* filename, float bias[SIZE], 
                              float scale_in, float scale_weight) {
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
        // Bias was quantized as: b_int32 = b_real / (scale_in * scale_weight)
        // So: b_real = b_int32 * (scale_in * scale_weight)
        bias[i] = (float)value * (scale_in * scale_weight);
    }

    file.close();
    cout << "Loaded and dequantized bias " << filename << " (" << SIZE << ")" << endl;
    return true;
}

/**
 * Load scales from file
 */
bool load_scales(const char* filename, float &scale_in, float &scale_w1, float &scale_w2,
                float &scale_hidden, float &scale_out) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "ERROR: Cannot open file: " << filename << endl;
        return false;
    }

    string key;
    float value;
    char colon;

    while (file >> key >> colon >> value) {
        if (key == "scale_in:") scale_in = value;
        else if (key == "scale_w1:") scale_w1 = value;
        else if (key == "scale_w2:") scale_w2 = value;
        else if (key == "scale_hidden:") scale_hidden = value;
        else if (key == "scale_out:") scale_out = value;
    }

    file.close();
    cout << "Loaded scales from " << filename << endl;
    cout << "  scale_in: " << scale_in << endl;
    cout << "  scale_w1: " << scale_w1 << endl;
    cout << "  scale_w2: " << scale_w2 << endl;
    cout << "  scale_hidden: " << scale_hidden << endl;
    cout << "  scale_out: " << scale_out << endl;
    return true;
}

// ============================================================================
// Main Testbench
// ============================================================================

int main() {
    cout << "======================================================================" << endl;
    cout << "GraphSAGE FLOAT Testbench - Reduced Model" << endl;
    cout << "Goal: Match PyG float output exactly (no quantization errors)" << endl;
    cout << "Configuration: IN=" << IN_FEATURES << ", HIDDEN=" << HIDDEN_FEATURES 
         << ", OUT=" << OUT_FEATURES << ", NODES=" << NUM_NODES << endl;
    cout << "======================================================================" << endl;

    // Test directory (relative to HLS csim working directory)
    const char* test_dir = "../../../../../../build/test_vectors_float";

    // Allocate arrays with correct dimensions
    static float adj_matrix[NUM_NODES][NUM_NODES];
    static float input[NUM_NODES][IN_FEATURES];
    static float weights1[HIDDEN_FEATURES][IN_FEATURES];
    static float bias1[HIDDEN_FEATURES];
    static float weights2[OUT_FEATURES][HIDDEN_FEATURES];
    static float bias2[OUT_FEATURES];
    static float output[NUM_NODES][OUT_FEATURES];
    static float reference[NUM_NODES][OUT_FEATURES];

    // Load adjacency matrix (FLOAT)
    char adj_file[256];
    sprintf(adj_file, "%s/adj_matrix.txt", test_dir);
    if (!load_float_matrix<NUM_NODES, NUM_NODES>(adj_file, adj_matrix)) {
        return 1;
    }

    // Load input features (FLOAT)
    char input_file[256];
    sprintf(input_file, "%s/network_input.txt", test_dir);
    if (!load_float_matrix<NUM_NODES, IN_FEATURES>(input_file, input)) {
        return 1;
    }

    // Load weights1 (FLOAT)
    char w1_file[256];
    sprintf(w1_file, "%s/weights_layer1.txt", test_dir);
    if (!load_float_matrix<HIDDEN_FEATURES, IN_FEATURES>(w1_file, weights1)) {
        return 1;
    }

    // Load bias1 (FLOAT)
    char b1_file[256];
    sprintf(b1_file, "%s/bias_layer1.txt", test_dir);
    
    // Load as 1D array first, then copy
    float bias1_temp[HIDDEN_FEATURES];
    ifstream b1(b1_file);
    if (!b1.is_open()) {
        cout << "ERROR: Cannot open file: " << b1_file << endl;
        return 1;
    }
    for (int i = 0; i < HIDDEN_FEATURES; i++) {
        if (!(b1 >> bias1_temp[i])) {
            cout << "ERROR: Failed to read bias1[" << i << "]" << endl;
            return 1;
        }
        bias1[i] = bias1_temp[i];
    }
    b1.close();
    cout << "Loaded " << b1_file << " (" << HIDDEN_FEATURES << ")" << endl;

    // Load weights2 (FLOAT)
    char w2_file[256];
    sprintf(w2_file, "%s/weights_layer2.txt", test_dir);
    if (!load_float_matrix<OUT_FEATURES, HIDDEN_FEATURES>(w2_file, weights2)) {
        return 1;
    }

    // Load bias2 (FLOAT)
    char b2_file[256];
    sprintf(b2_file, "%s/bias_layer2.txt", test_dir);
    
    float bias2_temp[OUT_FEATURES];
    ifstream b2(b2_file);
    if (!b2.is_open()) {
        cout << "ERROR: Cannot open file: " << b2_file << endl;
        return 1;
    }
    for (int i = 0; i < OUT_FEATURES; i++) {
        if (!(b2 >> bias2_temp[i])) {
            cout << "ERROR: Failed to read bias2[" << i << "]" << endl;
            return 1;
        }
        bias2[i] = bias2_temp[i];
    }
    b2.close();
    cout << "Loaded " << b2_file << " (" << OUT_FEATURES << ")" << endl;

    // Load reference output (FLOAT)
    char ref_file[256];
    sprintf(ref_file, "%s/network_output_reference.txt", test_dir);
    if (!load_float_matrix<NUM_NODES, OUT_FEATURES>(ref_file, reference)) {
        return 1;
    }

    cout << "\n======================================================================" << endl;
    cout << "Running HLS GraphSAGE (FLOAT)..." << endl;
    cout << "======================================================================" << endl;

    // Print input sample
    cout << "\nInput sample [0,:5]: ";
    for (int i = 0; i < min(5, IN_FEATURES); i++) {
        cout << fixed << setprecision(4) << input[0][i] << " ";
    }
    cout << endl;

    // Run DUT
    graphsage_network(
        adj_matrix,
        input,
        weights1,
        bias1,
        weights2,
        bias2,
        output
    );

    cout << "\n======================================================================" << endl;
    cout << "Comparing HLS output vs Reference" << endl;
    cout << "======================================================================" << endl;

    // Print output samples
    cout << "\nHLS Output [0,:]: ";
    for (int i = 0; i < OUT_FEATURES; i++) {
        cout << fixed << setprecision(4) << output[0][i] << " ";
    }
    cout << endl;

    cout << "Reference  [0,:]: ";
    for (int i = 0; i < OUT_FEATURES; i++) {
        cout << fixed << setprecision(4) << reference[0][i] << " ";
    }
    cout << endl;

    // Compare outputs
    int errors = 0;
    float max_diff = 0.0f;
    float tolerance = 0.01f;  // 1% tolerance for float comparison

    for (int i = 0; i < NUM_NODES; i++) {
        for (int j = 0; j < OUT_FEATURES; j++) {
            float diff = fabs(output[i][j] - reference[i][j]);
            float rel_diff = diff / (fabs(reference[i][j]) + 1e-6f);
            
            if (diff > max_diff) max_diff = diff;
            
            if (rel_diff > tolerance) {
                if (errors < 10) {  // Print first 10 errors
                    cout << "  Mismatch [" << i << "," << j << "]: "
                         << "HLS=" << output[i][j] << ", "
                         << "Ref=" << reference[i][j] << ", "
                         << "Diff=" << diff << " (rel=" << (rel_diff*100) << "%)" << endl;
                }
                errors++;
            }
        }
    }

    cout << "\n======================================================================" << endl;
    cout << "Test Results" << endl;
    cout << "======================================================================" << endl;
    cout << "Max absolute difference: " << max_diff << endl;
    cout << "Errors (>" << (tolerance*100) << "% relative): " << errors 
         << " / " << (NUM_NODES * OUT_FEATURES) << endl;

    if (errors == 0) {
        cout << "\n✓ FLOAT TESTBENCH PASSED: HLS matches PyG reference!" << endl;
        cout << "Ready for synthesis with reduced model configuration" << endl;
        return 0;
    } else {
        cout << "\n✗ FLOAT TESTBENCH FAILED: HLS does not match reference" << endl;
        cout << "Debug: Check aggregation, linear transform, or weight loading" << endl;
        return 1;
    }
}
