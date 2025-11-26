# INT8-Only HLS Implementation Plan

## Objective
Create a fully integer-only HLS implementation for PTQ GraphSAGE, eliminating all floating-point operations from the datapath.

## Current Status
✅ PTQ float quant/dequant HLS working (testbench passed)
✅ PTQ parameters generated in `build/weights_ptq_float/`
✅ PTQ INT8 parameters will be in `build/weights_ptq_int8/`
🔄 Waiting for pipeline to generate INT8 parameters

## Implementation Steps

### Step 1: Verify INT8 Parameters Available
Files needed in `build/weights_ptq_int8/`:
- `bias_layer1_int32.txt` - INT32 biases for layer 1
- `bias_layer2_int32.txt` - INT32 biases for layer 2
- `adj_matrix_int16.txt` - Fixed-point adjacency matrix
- `int8_params.json` - Fixed-point scale factors (M, K, eff_scale_fp, beta_fp)

Weights reused from PTQ float:
- `build/weights_ptq_float/conv1_lin_l_weight.txt` (INT8)
- `build/weights_ptq_float/conv2_lin_l_weight.txt` (INT8)

### Step 2: Create INT8 HLS Header
File: `hls/graphsage_layer_int8.h`

```cpp
// Network dimensions
#define NUM_NODES 8
#define IN_FEAT 16
#define HIDDEN_FEAT 24
#define OUT_FEAT 7

// Fixed-point precision
#define M 20              // Fractional bits for scale factors
#define K 4096            // Adjacency scale (2^12)

// Fixed-point scale factors (precomputed from Python)
extern const int32_t EFF_SCALE1_FP;  // Layer 1 requantization
extern const int32_t EFF_SCALE2_FP;  // Layer 2 requantization
extern const int32_t BETA1_FP;       // Aggregation 1 scale
extern const int32_t BETA2_FP;       // Aggregation 2 scale

// Network parameters (INT8/INT16/INT32)
extern const int8_t W1[HIDDEN_FEAT][IN_FEAT];
extern const int8_t W2[OUT_FEAT][HIDDEN_FEAT];
extern const int32_t B1[HIDDEN_FEAT];
extern const int32_t B2[OUT_FEAT];
extern const int16_t ADJ[NUM_NODES][NUM_NODES];
```

### Step 3: Implement Integer-Only Functions
File: `hls/graphsage_layer_int8.cpp`

#### 3.1 Integer Aggregation
```cpp
void aggregate_int8_only(
    const int8_t features[NUM_NODES][FEAT_DIM],
    const int16_t adj[NUM_NODES][NUM_NODES],
    int8_t aggregated[NUM_NODES][FEAT_DIM],
    int32_t beta_fp,
    int m_bits
) {
    for (int i = 0; i < NUM_NODES; i++) {
        for (int f = 0; f < FEAT_DIM; f++) {
            int32_t tmp = 0;
            
            // Sum: A_fp[i][j] * q[j]
            for (int j = 0; j < NUM_NODES; j++) {
                tmp += (int32_t)adj[i][j] * (int32_t)features[j][f];
            }
            
            // Scale: tmp * beta_fp >> M
            int64_t tmp_scaled = (int64_t)tmp * (int64_t)beta_fp;
            int64_t tmp_rounded = tmp_scaled + (1LL << (m_bits - 1));
            int32_t q_int = (int32_t)(tmp_rounded >> m_bits);
            
            // Clamp to INT8
            if (q_int > 127) q_int = 127;
            if (q_int < -128) q_int = -128;
            aggregated[i][f] = (int8_t)q_int;
        }
    }
}
```

#### 3.2 Integer Linear Layer
```cpp
void linear_int8_only(
    const int8_t input[NUM_NODES][IN_DIM],
    const int8_t weights[OUT_DIM][IN_DIM],
    const int32_t bias[OUT_DIM],
    int8_t output[NUM_NODES][OUT_DIM],
    int32_t eff_scale_fp,
    int m_bits
) {
    for (int n = 0; n < NUM_NODES; n++) {
        for (int o = 0; o < OUT_DIM; o++) {
            int32_t acc = 0;
            
            // Dot product: W * x
            for (int i = 0; i < IN_DIM; i++) {
                acc += (int32_t)weights[o][i] * (int32_t)input[n][i];
            }
            
            // Add bias (already in accumulator domain)
            acc += bias[o];
            
            // Requantize: acc * eff_scale_fp >> M
            int64_t tmp_scaled = (int64_t)acc * (int64_t)eff_scale_fp;
            int64_t tmp_rounded = tmp_scaled + (1LL << (m_bits - 1));
            int32_t q_int = (int32_t)(tmp_rounded >> m_bits);
            
            // Clamp to INT8
            if (q_int > 127) q_int = 127;
            if (q_int < -128) q_int = -128;
            output[n][o] = (int8_t)q_int;
        }
    }
}
```

#### 3.3 Integer ReLU
```cpp
void relu_int8(int8_t data[NUM_NODES][FEAT_DIM]) {
    for (int i = 0; i < NUM_NODES; i++) {
        for (int f = 0; f < FEAT_DIM; f++) {
            if (data[i][f] < 0) {
                data[i][f] = 0;
            }
        }
    }
}
```

#### 3.4 Top-Level Network
```cpp
void graphsage_network_int8(
    int8_t input[NUM_NODES][IN_FEAT],
    int8_t output[NUM_NODES][OUT_FEAT]
) {
    #pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1
    #pragma HLS INTERFACE s_axilite port=return
    
    static int8_t agg1[NUM_NODES][IN_FEAT];
    static int8_t hidden[NUM_NODES][HIDDEN_FEAT];
    static int8_t agg2[NUM_NODES][HIDDEN_FEAT];
    
    // Layer 1: Aggregate
    aggregate_int8_only(input, ADJ, agg1, BETA1_FP, M);
    
    // Layer 1: Linear + ReLU
    linear_int8_only(agg1, W1, B1, hidden, EFF_SCALE1_FP, M);
    relu_int8(hidden);
    
    // Layer 2: Aggregate
    aggregate_int8_only(hidden, ADJ, agg2, BETA2_FP, M);
    
    // Layer 2: Linear (no ReLU)
    linear_int8_only(agg2, W2, B2, output, EFF_SCALE2_FP, M);
}
```

### Step 4: Create INT8 Testbench
File: `hls/testbench_int8.cpp`

Load test vectors from `build/test_vectors_ptq_int8/`:
- `network_input_int8.txt`
- `network_output_int8_reference.txt`
- `adj_matrix_int16.txt`
- `weights_layer1_int8.txt`, `weights_layer2_int8.txt`
- `bias_layer1_int32.txt`, `bias_layer2_int32.txt`
- `int8_config.txt` (M, K, scale factors)

Compare HLS output vs Python integer emulator output (should match bit-exactly).

### Step 5: Create TCL Scripts
File: `hls/run_csim_int8.tcl`

```tcl
open_project graphsage_int8
set_top graphsage_network_int8
add_files graphsage_layer_int8.cpp
add_files graphsage_layer_int8.h
add_files -tb testbench_int8.cpp
open_solution "solution1"
set_part {xcvu13p-fsga2577-1-e}
create_clock -period 2.77 -name default
csim_design -clean
close_project
```

### Step 6: Create Makefile
File: `hls/Makefile.int8`

```makefile
# INT8-only PTQ HLS Makefile

.PHONY: csim csynth cosim clean

csim:
	@echo "Running C-Simulation for INT8 PTQ..."
	cd ../build/hls && vitis_hls -f ../../hls/run_csim_int8.tcl

csynth:
	@echo "Running C-Synthesis for INT8 PTQ..."
	cd ../build/hls && vitis_hls -f ../../hls/run_hls_int8.tcl

clean:
	rm -rf ../build/hls/graphsage_int8

help:
	@echo "INT8-only PTQ HLS Targets:"
	@echo "  make csim     - Run C simulation"
	@echo "  make csynth   - Run C synthesis"
	@echo "  make clean    - Clean HLS project"
```

## Testing Strategy

### Phase 1: Validate Parameters
1. Check `int8_params.json` contains M, K, eff_scale_fp, beta_fp
2. Verify INT32 biases are in accumulator domain
3. Verify INT16 adjacency is scaled by K

### Phase 2: Python-HLS Comparison
1. Run Python integer emulator: `tests/generate_test_vectors_ptq_int8.py`
2. Run HLS C-sim: `make -f Makefile.int8 csim`
3. Compare outputs - must match bit-exactly

### Phase 3: Resource Analysis
1. Run C-synthesis
2. Compare resources vs PTQ float:
   - Expected: ~90% DSP reduction (no float multiply)
   - Expected: Similar LUT/FF usage
   - Expected: Same throughput

## Success Criteria
✅ HLS C-sim passes (matches Python integer emulator)
✅ No floating-point operations in datapath
✅ DSP usage drops to near-zero
✅ Output matches PTQ float within quantization error bounds

## Next Steps After Completion
1. Document resource savings
2. Update README with INT8 HLS variant
3. Consider INT16 accumulator optimization (if INT32 too wide)
4. Benchmark latency vs PTQ float
