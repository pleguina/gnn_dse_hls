/**
 * Testbench for GraphSAGE FLOAT Implementation - PHASE 1
 * 
 * This testbench loads test vectors and runs the FLOAT version of GraphSAGE.
 * It should match PyG output exactly (no quantization errors).
 */

#include "graphsage_layer_float.h"
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
bool load_float_matrix(const char* filename, float matrix[MAX_NODES][MAX_NODES], int rows, int cols) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "ERROR: Cannot open file: " << filename << endl;
        return false;
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (!(file >> matrix[i][j])) {
                cout << "ERROR: Failed to read element [" << i << "," << j << "] from " << filename << endl;
                return false;
            }
        }
    }

    file.close();
    cout << "Loaded " << filename << " (" << rows << "x" << cols << ")" << endl;
    return true;
}

/**
 * Load INT8 matrix and dequantize to float
 */
bool load_and_dequantize(const char* filename, float matrix[MAX_NODES][MAX_NODES], 
                        int rows, int cols, float scale) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "ERROR: Cannot open file: " << filename << endl;
        return false;
    }

    int value;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (!(file >> value)) {
                cout << "ERROR: Failed to read element [" << i << "," << j << "] from " << filename << endl;
                return false;
            }
            matrix[i][j] = (float)value * scale;
        }
    }

    file.close();
    cout << "Loaded and dequantized " << filename << " (" << rows << "x" << cols << ")" << endl;
    return true;
}

/**
 * Load INT32 vector and dequantize to float (for biases)
 */
bool load_bias_and_dequantize(const char* filename, float bias[MAX_FEATURES_HIDDEN], 
                              int size, float scale_in, float scale_weight) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "ERROR: Cannot open file: " << filename << endl;
        return false;
    }

    int value;
    for (int i = 0; i < size; i++) {
        if (!(file >> value)) {
            cout << "ERROR: Failed to read element [" << i << "] from " << filename << endl;
            return false;
        }
        // Bias was quantized as: b_int32 = b_real / (scale_in * scale_weight)
        // So: b_real = b_int32 * (scale_in * scale_weight)
        bias[i] = (float)value * (scale_in * scale_weight);
    }

    file.close();
    cout << "Loaded and dequantized bias " << filename << " (" << size << ")" << endl;
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
    cout << "GraphSAGE FLOAT Testbench - PHASE 1" << endl;
    cout << "Goal: Match PyG float output exactly (no quantization errors)" << endl;
    cout << "======================================================================" << endl;

    // Test configuration
    const int num_nodes = 8;  // Should match test vectors
    const char* test_dir = "../build/test_vectors";

    // Allocate arrays
    static float adj_matrix[MAX_NODES][MAX_NODES];
    static float input[MAX_NODES][MAX_FEATURES_IN];
    static float weights1[MAX_FEATURES_HIDDEN][MAX_FEATURES_IN];
    static float bias1[MAX_FEATURES_HIDDEN];
    static float weights2[MAX_FEATURES_OUT][MAX_FEATURES_HIDDEN];
    static float bias2[MAX_FEATURES_OUT];
    static float output[MAX_NODES][MAX_FEATURES_OUT];
    static float reference[MAX_NODES][MAX_FEATURES_OUT];

    // Load scales
    float scale_in, scale_w1, scale_w2, scale_hidden, scale_out;
    char scales_file[256];
    sprintf(scales_file, "%s/scales.txt", test_dir);
    if (!load_scales(scales_file, scale_in, scale_w1, scale_w2, scale_hidden, scale_out)) {
        return 1;
    }

    // Load adjacency matrix (already in float)
    char adj_file[256];
    sprintf(adj_file, "%s/adj_matrix.txt", test_dir);
    if (!load_float_matrix(adj_file, adj_matrix, num_nodes, num_nodes)) {
        return 1;
    }

    // Load input (INT8 → dequantize to float)
    char input_file[256];
    sprintf(input_file, "%s/network_input.txt", test_dir);
    if (!load_and_dequantize(input_file, input, num_nodes, MAX_FEATURES_IN, scale_in)) {
        return 1;
    }

    // Load weights1 (INT8 → dequantize to float)
    char w1_file[256];
    sprintf(w1_file, "%s/weights_layer1.txt", test_dir);
    if (!load_and_dequantize(w1_file, weights1, MAX_FEATURES_HIDDEN, MAX_FEATURES_IN, scale_w1)) {
        return 1;
    }

    // Load bias1 (INT32 → dequantize to float)
    char b1_file[256];
    sprintf(b1_file, "%s/bias_layer1.txt", test_dir);
    if (!load_bias_and_dequantize(b1_file, bias1, MAX_FEATURES_HIDDEN, scale_in, scale_w1)) {
        return 1;
    }

    // Load weights2 (INT8 → dequantize to float)
    char w2_file[256];
    sprintf(w2_file, "%s/weights_layer2.txt", test_dir);
    if (!load_and_dequantize(w2_file, weights2, MAX_FEATURES_OUT, MAX_FEATURES_HIDDEN, scale_w2)) {
        return 1;
    }

    // Load bias2 (INT32 → dequantize to float)
    char b2_file[256];
    sprintf(b2_file, "%s/bias_layer2.txt", test_dir);
    if (!load_bias_and_dequantize(b2_file, bias2, MAX_FEATURES_OUT, scale_hidden, scale_w2)) {
        return 1;
    }

    // Load reference output (INT8 → dequantize to float)
    char ref_file[256];
    sprintf(ref_file, "%s/network_output_reference.txt", test_dir);
    if (!load_and_dequantize(ref_file, reference, num_nodes, MAX_FEATURES_OUT, scale_out)) {
        return 1;
    }

    cout << "\n======================================================================" << endl;
    cout << "Running HLS GraphSAGE (FLOAT)..." << endl;
    cout << "======================================================================" << endl;

    // Print input sample
    cout << "\nInput sample [0,:5]: ";
    for (int i = 0; i < 5; i++) {
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
        output,
        num_nodes
    );

    cout << "\n======================================================================" << endl;
    cout << "Comparing HLS output vs Reference" << endl;
    cout << "======================================================================" << endl;

    // Print output samples
    cout << "\nHLS Output [0,:]: ";
    for (int i = 0; i < MAX_FEATURES_OUT; i++) {
        cout << fixed << setprecision(4) << output[0][i] << " ";
    }
    cout << endl;

    cout << "Reference  [0,:]: ";
    for (int i = 0; i < MAX_FEATURES_OUT; i++) {
        cout << fixed << setprecision(4) << reference[0][i] << " ";
    }
    cout << endl;

    // Compare outputs
    int errors = 0;
    float max_diff = 0.0f;
    float tolerance = 0.01f;  // 1% tolerance for float comparison

    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < MAX_FEATURES_OUT; j++) {
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
         << " / " << (num_nodes * MAX_FEATURES_OUT) << endl;

    if (errors == 0) {
        cout << "\n✓ PHASE 1 PASSED: HLS FLOAT matches PyG reference!" << endl;
        cout << "Next step: Proceed to Phase 2 (INT8 quantization)" << endl;
        return 0;
    } else {
        cout << "\n✗ PHASE 1 FAILED: HLS FLOAT does not match reference" << endl;
        cout << "Debug: Check aggregation, linear transform, or weight loading" << endl;
        return 1;
    }
}
