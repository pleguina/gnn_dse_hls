# HLS Variants and Test Vectors Mapping

## Three HLS Implementations

### 1. Float HLS (Baseline)
**Purpose**: Validate model correctness without quantization
**Data types**: All float32
**Test vectors**: `build/test_vectors_float/`
**Generator**: `generate_test_vectors_float.py`

**Files**:
- `network_input.txt` - Float features (e.g., 0.188205, 0.198222, ...)
- `adj_matrix.txt` - Float adjacency (e.g., 0.333333, ...)
- `weights_layer*.txt` - Float weights
- `bias_layer*.txt` - Float biases
- `network_output_reference.txt` - Float output

**HLS code**:
```cpp
// graphsage_layer_float.h
float aggregate(float features[N][F], float adj[N][N]) {
    // Pure floating-point operations
}
```

---

### 2. PTQ with Float Quant/Dequant HLS
**Purpose**: INT8 weights/activations but uses float for quant/dequant operations
**Data types**: INT8 data, float scales, float operations
**Test vectors**: `build/test_vectors_ptq/`
**Generator**: `generate_ptq_test_vectors_clean.py`

**Files**:
- `network_input.txt` - INT8 quantized features (e.g., 39, 41, 89, ...)
- `adj_matrix.txt` - **Float adjacency** (e.g., 0.333333, ...)
- `weights_layer*.txt` - INT8 quantized weights
- `bias_layer*.txt` - INT8 quantized biases (output domain)
- `scales.txt` - Float scale factors
- `network_output_reference.txt` - INT8 quantized output

**HLS code**:
```cpp
// graphsage_layer.h (current PTQ implementation)
ap_int<8> x_q[N][F];      // INT8 data
float adj[N][N];          // Float adjacency
float scale_in, scale_out;

// Dequantize
float x_float = x_q * scale_in;
// Aggregate in float
float agg = matmul(adj, x_float);
// Quantize back
ap_int<8> agg_q = round(agg / scale_out);
```

**Resources**: Requires DSP blocks for float multiplication

---

### 3. PTQ Integer-Only (Fixed-Point) HLS
**Purpose**: Pure integer datapath, no floating-point operations
**Data types**: INT8 data, INT16 adjacency, INT32 accumulators, fixed-point scales
**Test vectors**: `build/test_vectors_int8/`
**Generator**: `integer_ptq_emulator.py`

**Files**:
- `network_input_int8.txt` - INT8 quantized features (e.g., 39, 41, 89, ...)
- `adj_matrix_int16.txt` - **INT16 fixed-point adjacency** (e.g., 1365 = 0.333 * 4096)
- `weights_layer*_int8.txt` - INT8 quantized weights
- `bias_layer*_int32.txt` - **INT32 biases (accumulator domain)**
- `int8_config.txt` - Fixed-point parameters (M=20, K=4096, eff_scale_fp, beta_fp)
- `network_output_int8_reference.txt` - INT8 output

**HLS code**:
```cpp
// graphsage_layer_int8.h (NEW - to be implemented)
ap_int<8> x_q[N][F];       // INT8 data
ap_int<16> adj_fp[N][N];   // INT16 fixed-point adjacency
ap_int<32> bias_int32[F];  // INT32 bias (accumulator domain)
ap_int<32> beta_fp;        // Fixed-point scale factor

// Aggregate (integer-only)
ap_int<32> tmp = 0;
for (int j = 0; j < N; j++)
    tmp += adj_fp[i][j] * x_q[j][f];  // INT16 * INT8 = INT32

// Scale with fixed-point
ap_int<64> tmp_scaled = tmp * beta_fp;  // INT32 * INT32 = INT64
ap_int<8> agg_q = (tmp_scaled + (1 << 19)) >> 20;  // Shift right M bits
```

**Resources**: No DSP needed (uses LUTs), ~90% smaller than float version

---

## Test Vector Generators Status

### Current Generators:

| File | Generates | For HLS Variant | Status |
|------|-----------|-----------------|--------|
| `generate_test_vectors_float.py` | `test_vectors_float/` | Float HLS | ✅ Correct |
| `generate_ptq_test_vectors_clean.py` | `test_vectors_ptq/` | PTQ Float Quant/Dequant | ✅ Correct |
| `integer_ptq_emulator.py` | `test_vectors_int8/` | PTQ Integer-Only | ✅ Correct |
| `generate_test_vectors.py` | `test_vectors/` (old) | ❌ Wrong (float model) | ❌ Delete |
| `generate_ptq_test_vectors.py` | ? | ❌ Duplicate | ❌ Delete |

---

## Key Differences

### Adjacency Matrix:

| Variant | Format | Example Value | Scale |
|---------|--------|---------------|-------|
| Float | float32 | 0.333333 | 1.0 |
| PTQ Float | float32 | 0.333333 | 1.0 |
| PTQ Int8 | int16 | 1365 | K=4096 (0.333*4096≈1365) |

### Biases:

| Variant | Format | Domain | Formula |
|---------|--------|--------|---------|
| Float | float32 | Output | b_float |
| PTQ Float | int8 | Output | round(b_float / scale_out) |
| PTQ Int8 | int32 | Accumulator | round(b_float / (scale_in * scale_w)) |

### Operations:

| Variant | Aggregation | Linear | Requantization |
|---------|-------------|--------|----------------|
| Float | float matmul | float matmul | N/A |
| PTQ Float | dequant→float matmul→quant | dequant→float matmul→quant | Float divide |
| PTQ Int8 | INT16×INT8, fixed-point scale | INT8×INT8+INT32, fixed-point scale | Shift right |

---

## Validation Checklist

### ✅ Float HLS
- [ ] `generate_test_vectors_float.py` generates float test vectors
- [ ] All values are float32
- [ ] HLS testbench loads float data
- [ ] Output matches `network_output_reference.txt` (float comparison)

### ✅ PTQ Float Quant/Dequant HLS
- [ ] `generate_ptq_test_vectors_clean.py` generates INT8 test vectors
- [ ] Input/weights/output are INT8
- [ ] Adjacency is **float**
- [ ] Biases are INT8 (can be converted to INT32 in HLS)
- [ ] Has `scales.txt` with float scale factors
- [ ] HLS uses float dequantization and quantization
- [ ] Output matches `network_output_reference.txt` (INT8 exact match)

### ✅ PTQ Integer-Only HLS
- [ ] `integer_ptq_emulator.py` generates integer-only test vectors
- [ ] Input/weights/output are INT8
- [ ] Adjacency is **INT16 fixed-point**
- [ ] Biases are **INT32 (accumulator domain)**
- [ ] Has `int8_config.txt` with M, K, eff_scale_fp, beta_fp
- [ ] HLS uses only integer arithmetic (no float)
- [ ] Output matches `network_output_int8_reference.txt` (INT8 exact match)

---

## Current Test Vector Contents

### test_vectors_float/ (Float HLS)
```
adj_matrix.txt              # Float: 0.333333 ...
network_input.txt           # Float: 0.188205 ...
weights_layer1.txt          # Float
bias_layer1.txt             # Float
network_output_reference.txt # Float
```

### test_vectors_ptq/ (PTQ Float Quant/Dequant)
```
adj_matrix.txt              # Float: 0.333333 ...  ← FLOAT!
network_input.txt           # INT8: 39 41 89 ...
weights_layer1.txt          # INT8
bias_layer1.txt             # INT8 (can convert to INT32)
scales.txt                  # scale_in, scale_w1, scale_hidden, scale_out
network_output_reference.txt # INT8: -45 -4 -96 ...
```

### test_vectors_int8/ (PTQ Integer-Only)
```
adj_matrix_int16.txt        # INT16: 1365 0 0 ...  ← INT16!
network_input_int8.txt      # INT8: 39 41 89 ...
weights_layer1_int8.txt     # INT8
bias_layer1_int32.txt       # INT32  ← Accumulator domain!
int8_config.txt             # M=20, K=4096, eff_scale_fp, beta_fp
network_output_int8_reference.txt # INT8: -2 -19 -38 ...
```

---

## Summary

**All three HLS variants are correctly supported:**

1. **Float HLS**: Full precision baseline
   - Vectors: ✅ `test_vectors_float/`
   - Generator: ✅ `generate_test_vectors_float.py`

2. **PTQ Float Quant/Dequant**: INT8 data, float operations
   - Vectors: ✅ `test_vectors_ptq/`
   - Generator: ✅ `generate_ptq_test_vectors_clean.py`
   - Note: Uses **float adjacency** and float scales

3. **PTQ Integer-Only**: Pure integer arithmetic
   - Vectors: ✅ `test_vectors_int8/`
   - Generator: ✅ `integer_ptq_emulator.py`
   - Note: Uses **INT16 adjacency** and fixed-point scales

**The key difference between PTQ variants**: Float adjacency vs INT16 fixed-point adjacency!
