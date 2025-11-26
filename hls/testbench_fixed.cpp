/**
 * Testbench for Fixed-Point GraphSAGE
 * 
 * Tests the fixed-point implementation against floating-point reference.
 * Allows experimenting with different Q formats to analyze precision vs resource tradeoffs.
 */

#include "graphsage_layer_fixed.h"
#include "graphsage_layer_fixed.cpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdio>

using namespace std;

// ============================================================================
// HELPER FUNCTIONS FOR LOADING TEST DATA
// ============================================================================

/**
 * Load floating-point matrix from file and convert to fixed-point
 */
template<int ROWS, int COLS, typename T>
bool load_matrix(const char* filename, T matrix[ROWS][COLS]) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Cannot open " << filename << endl;
        return false;
    }

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            float val;
            if (!(file >> val)) {
                cerr << "Error: Not enough data in " << filename << endl;
                return false;
            }
            matrix[i][j] = val;  // Implicit conversion to fixed-point
        }
    }

    cout << "Loaded " << filename << " (" << ROWS << "x" << COLS << ")" << endl;
    return true;
}

/**
 * Load floating-point vector from file and convert to fixed-point
 */
template<int SIZE, typename T>
bool load_vector(const char* filename, T vector[SIZE]) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Cannot open " << filename << endl;
        return false;
    }

    for (int i = 0; i < SIZE; i++) {
        float val;
        if (!(file >> val)) {
            cerr << "Error: Not enough data in " << filename << endl;
            return false;
        }
        vector[i] = val;  // Implicit conversion to fixed-point
    }

    cout << "Loaded " << filename << " (" << SIZE << " elements)" << endl;
    return true;
}

/**
 * Load floating-point reference output for comparison
 */
template<int ROWS, int COLS>
bool load_float_matrix(const char* filename, float matrix[ROWS][COLS]) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Cannot open " << filename << endl;
        return false;
    }

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            if (!(file >> matrix[i][j])) {
                cerr << "Error: Not enough data in " << filename << endl;
                return false;
            }
        }
    }

    cout << "Loaded " << filename << " (" << ROWS << "x" << COLS << ")" << endl;
    return true;
}

/**
 * Print fixed-point matrix (showing first few rows/cols)
 */
template<int ROWS, int COLS, typename T>
void print_matrix(const char* name, T matrix[ROWS][COLS], int max_rows = 3, int max_cols = 8) {
    cout << name << " (showing " << max_rows << "x" << max_cols << "):" << endl;
    for (int i = 0; i < max_rows && i < ROWS; i++) {
        cout << "  ";
        for (int j = 0; j < max_cols && j < COLS; j++) {
            cout << setw(8) << fixed << setprecision(4) << (float)matrix[i][j];
        }
        if (max_cols < COLS) cout << " ...";
        cout << endl;
    }
    if (max_rows < ROWS) cout << "  ..." << endl;
    cout << endl;
}

/**
 * Compare fixed-point output with floating-point reference
 */
template<int ROWS, int COLS, typename T>
bool compare_outputs(T result[ROWS][COLS], float reference[ROWS][COLS], float tolerance_percent = 1.0) {
    bool pass = true;
    float max_error = 0.0;
    float max_rel_error = 0.0;
    int total_errors = 0;
    
    cout << "Comparing outputs (tolerance = " << tolerance_percent << "% relative error):" << endl;
    
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            float result_float = (float)result[i][j];
            float ref_float = reference[i][j];
            float error = fabs(result_float - ref_float);
            float rel_error = 0.0;
            
            if (fabs(ref_float) > 1e-6) {
                rel_error = (error / fabs(ref_float)) * 100.0;
            }
            
            if (error > max_error) {
                max_error = error;
            }
            if (rel_error > max_rel_error) {
                max_rel_error = rel_error;
            }
            
            if (rel_error > tolerance_percent && error > 0.01) {  // Skip tiny absolute errors
                if (total_errors < 10) {  // Print first 10 errors
                    cout << "  ERROR [" << i << "," << j << "]: "
                         << "result=" << result_float
                         << " reference=" << ref_float
                         << " error=" << error
                         << " (" << rel_error << "%)" << endl;
                }
                total_errors++;
                pass = false;
            }
        }
    }
    
    cout << "Max absolute error: " << max_error << endl;
    cout << "Max relative error: " << max_rel_error << "%" << endl;
    cout << "Errors (>" << tolerance_percent << "% relative): " << total_errors << " / " << (ROWS * COLS) << endl;
    
    return pass;
}

// ============================================================================
// MAIN TESTBENCH
// ============================================================================

int main() {
    cout << "========================================" << endl;
    cout << "GraphSAGE FIXED-POINT Testbench" << endl;
    cout << "========================================" << endl;
    cout << "Configuration: IN=" << IN_FEATURES 
         << ", HIDDEN=" << HIDDEN_FEATURES 
         << ", OUT=" << OUT_FEATURES 
         << ", NODES=" << NUM_NODES << endl;
    cout << "Fixed-Point Formats:" << endl;
    cout << "  Data:   ap_fixed<" << DATA_W << "," << DATA_I << "> (Q" << DATA_I << "." << (DATA_W-DATA_I) << ")" << endl;
    cout << "  Weight: ap_fixed<" << WEIGHT_W << "," << WEIGHT_I << "> (Q" << WEIGHT_I << "." << (WEIGHT_W-WEIGHT_I) << ")" << endl;
    cout << "  Acc:    ap_fixed<" << ACC_W << "," << ACC_I << "> (Q" << ACC_I << "." << (ACC_W-ACC_I) << ")" << endl;
    cout << "  Scale:  ap_fixed<" << SCALE_W << "," << SCALE_I << "> (Q" << SCALE_I << "." << (SCALE_W-SCALE_I) << ")" << endl;
    cout << "========================================" << endl << endl;

    // Test vector directory (absolute paths from project root)
    const char* test_dir = "/home/pelayo/work/simple-gnn/build/test_vectors_float";

    // ========== Declare arrays ==========
    // Fixed-point arrays for HLS
    static scale_t adj_matrix[NUM_NODES][NUM_NODES];
    static data_t input[NUM_NODES][IN_FEATURES];
    static data_t output[NUM_NODES][OUT_FEATURES];
    static weight_t weights1[HIDDEN_FEATURES][IN_FEATURES];
    static weight_t bias1[HIDDEN_FEATURES];
    static weight_t weights2[OUT_FEATURES][HIDDEN_FEATURES];
    static weight_t bias2[OUT_FEATURES];
    
    // Floating-point reference
    static float reference_output[NUM_NODES][OUT_FEATURES];

    // ========== Load test vectors ==========
    cout << "Loading test vectors..." << endl;
    
    char filename[512];
    
    // Load adjacency matrix
    sprintf(filename, "%s/adj_matrix.txt", test_dir);
    if (!load_matrix<NUM_NODES, NUM_NODES>(filename, adj_matrix)) {
        return 1;
    }

    // Load input features
    sprintf(filename, "%s/network_input.txt", test_dir);
    if (!load_matrix<NUM_NODES, IN_FEATURES>(filename, input)) {
        return 1;
    }

    // Load layer 1 weights and bias
    sprintf(filename, "%s/weights_layer1.txt", test_dir);
    if (!load_matrix<HIDDEN_FEATURES, IN_FEATURES>(filename, weights1)) {
        return 1;
    }
    sprintf(filename, "%s/bias_layer1.txt", test_dir);
    if (!load_vector<HIDDEN_FEATURES>(filename, bias1)) {
        return 1;
    }

    // Load layer 2 weights and bias
    sprintf(filename, "%s/weights_layer2.txt", test_dir);
    if (!load_matrix<OUT_FEATURES, HIDDEN_FEATURES>(filename, weights2)) {
        return 1;
    }
    sprintf(filename, "%s/bias_layer2.txt", test_dir);
    if (!load_vector<OUT_FEATURES>(filename, bias2)) {
        return 1;
    }

    // Load reference output (floating-point)
    sprintf(filename, "%s/network_output_reference.txt", test_dir);
    if (!load_float_matrix<NUM_NODES, OUT_FEATURES>(filename, reference_output)) {
        return 1;
    }

    cout << endl;

    // ========== Print sample inputs ==========
    print_matrix<NUM_NODES, IN_FEATURES>("Input", input);

    // ========== Run GraphSAGE network ==========
    cout << "Running fixed-point GraphSAGE network..." << endl;
    
    graphsage_network_fixed(
        adj_matrix,
        input,
        weights1,
        bias1,
        weights2,
        bias2,
        output
    );

    cout << "Inference complete!" << endl << endl;

    // ========== Compare results ==========
    print_matrix<NUM_NODES, OUT_FEATURES>("Fixed-Point Output", output);
    
    cout << "Reference Output (showing 3x8):" << endl;
    for (int i = 0; i < 3 && i < NUM_NODES; i++) {
        cout << "  ";
        for (int j = 0; j < 8 && j < OUT_FEATURES; j++) {
            cout << setw(8) << fixed << setprecision(4) << reference_output[i][j];
        }
        if (OUT_FEATURES > 8) cout << " ...";
        cout << endl;
    }
    if (NUM_NODES > 3) cout << "  ..." << endl;
    cout << endl;

    bool pass = compare_outputs<NUM_NODES, OUT_FEATURES>(
        output, reference_output, 1.0  // 1% tolerance
    );

    // ========== Test result ==========
    cout << endl << "========================================" << endl;
    if (pass) {
        cout << "TEST PASSED! ✓" << endl;
        cout << "Fixed-point output matches reference within tolerance." << endl;
        return 0;
    } else {
        cout << "TEST FAILED! ✗" << endl;
        cout << "Fixed-point output differs from reference." << endl;
        cout << "Try adjusting bit widths (DATA_W, WEIGHT_W, ACC_W) for better precision." << endl;
        return 1;
    }
}
