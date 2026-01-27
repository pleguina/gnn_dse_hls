# INT8 vs Fixed-Point HLS Implementation Comparison

**Date**: January 27, 2026  
**Comparison**: `graphsage_layer_int8.h/cpp` vs `graphsage_layer_fixed.h/cpp`

---

## Summary: Key Philosophical Difference

| Aspect | INT8 Implementation | Fixed-Point Implementation |
|--------|---------------------|---------------------------|
| **Numeric Type** | `ap_int<>` (pure integers) | `ap_fixed<>` (fractional representation) |
| **Philosophy** | Explicit quantization with manual scaling | Implicit fractional arithmetic |
| **Control Level** | Low-level (bit-exact control) | High-level (HLS manages fractional bits) |
| **Quantization** | Manual (matches PyTorch PTQ) | Automatic (HLS synthesis decides) |

---

## 1. Data Type Definitions

### INT8 Implementation
```cpp
// Pure integer types with explicit bit-widths
typedef ap_int<8>           data_t;        // INT8 activations
typedef ap_int<8>           weight_t;      // INT8 weights
typedef ap_int<ADJ_BITS>    adj_t;         // INT16 adjacency (scaled by K=2^12)
typedef ap_int<ACC_BITS>    acc_t;         // INT32 accumulators
typedef ap_int<SCALE_BITS>  scale_fp_t;    // Fixed-point scales (M fractional bits)
typedef ap_int<MULT_BITS>   mult_t;        // 64-bit products

// Explicit configuration
#define M_BITS 24              // Fractional bits for scaling
#define K_BITS 12              // Adjacency scaling exponent
#define ACC_BITS 32            // Accumulator width
#define SCALE_BITS 32          // Scale factor width
#define MULT_BITS 64           // Product width
```

**Characteristics**:
- ✅ **Explicit bit-widths**: You control every bit
- ✅ **Data-driven optimization**: Can reduce ACC_BITS from 32→22, MULT_BITS from 64→43
- ✅ **Matches PyTorch exactly**: Same quantization scheme as PyTorch's `fake_quantize`
- ✅ **Bit-exact validation**: 0 LSB error vs Python integer emulator

### Fixed-Point Implementation
```cpp
// Fixed-point types with automatic fractional handling
typedef ap_fixed<DATA_W, DATA_I>     data_t;    // ap_fixed<16, 8> = 8 int, 8 frac
typedef ap_fixed<WEIGHT_W, WEIGHT_I> weight_t;  // ap_fixed<16, 4> = 4 int, 12 frac
typedef ap_fixed<ACC_W, ACC_I>       acc_t;     // ap_fixed<32, 16> = 16 int, 16 frac
typedef ap_fixed<SCALE_W, SCALE_I>   scale_t;   // ap_fixed<16, 2> = 2 int, 14 frac

// Configuration with Q-format
#define DATA_W 16              // Total width
#define DATA_I 8               // Integer bits (DATA_W - DATA_I = fractional)
#define WEIGHT_W 16
#define WEIGHT_I 4
#define ACC_W 32
#define ACC_I 16
```

**Characteristics**:
- ✅ **Automatic fractional arithmetic**: HLS handles Q-format conversion
- ⚠️ **Less control**: Can't easily reduce bit-widths beyond W/I parameters
- ⚠️ **Different from PyTorch**: Doesn't match PyTorch's INT8 quantization exactly
- ⚠️ **Harder validation**: Can't compare bit-exact with PyTorch

---

## 2. Quantization & Scaling

### INT8 Implementation (Manual Quantization)
```cpp
// Explicit quantization with rounding
mult_t scaled = (mult_t)acc * (mult_t)eff_scale_fp;
mult_t rounded = scaled + ROUND_CONST;              // Add 2^(M-1) for rounding
mult_t result = rounded >> M_BITS;                  // Shift right by M
output[n][o] = int8_clamp(result);                  // Clamp to [-128, 127]

// Where:
// - eff_scale_fp = round((scale_in * scale_w / scale_out) * 2^M)
// - M_BITS = 24 (fractional bits)
// - ROUND_CONST = 2^(M-1) for round-to-nearest
```

**Advantages**:
- ✅ **Matches PyTorch exactly**: Same quantization formula
- ✅ **Explicit rounding**: Round-to-nearest with ROUND_CONST
- ✅ **Explicit clamping**: Manual clamp to [-128, 127]
- ✅ **Bit-exact validation**: Can verify 0 LSB error
- ✅ **Optimizable**: Can reduce M_BITS from 24→20 (trade precision for DSP)

### Fixed-Point Implementation (Automatic Conversion)
```cpp
// Implicit conversion between Q-formats
acc_t acc = 0;
for (int i = 0; i < IN_FEAT; i++) {
    acc += features[n][i] * weights[o][i];  // HLS handles Q-format matching
}
acc += bias[o];
output[n][o] = (data_t)acc;                // HLS inserts shift/round automatically
```

**Advantages**:
- ✅ **Simpler code**: No manual scaling
- ✅ **Automatic**: HLS synthesis inserts shifts/rounds
- ⚠️ **Less control**: Can't control rounding mode explicitly
- ⚠️ **Different semantics**: Not INT8 quantization, it's fixed-point arithmetic

---

## 3. Adjacency Matrix Handling

### INT8 Implementation (Integer Adjacency)
```cpp
// Adjacency scaled by K=2^12 and stored as INT16
typedef ap_int<16> adj_t;

// In Python: adj_int = round(adj_float * 2^12)
// In HLS:
acc_t tmp = 0;
for (int j = 0; j < N_NODES; j++) {
    acc_t prod = (acc_t)adj_matrix[i][j] * (acc_t)features[j][f];
    tmp += prod;
}
// Then apply beta_fp to rescale: (tmp * beta_fp + ROUND) >> M
```

**Characteristics**:
- ✅ **Integer storage**: Adjacency stored as INT16 (scaled)
- ✅ **Explicit scaling**: beta_fp compensates for K scaling
- ✅ **Efficient**: INT16×INT8 multiply is smaller than FLOAT32×FLOAT32

### Fixed-Point Implementation (Fractional Adjacency)
```cpp
// Adjacency stored as ap_fixed<16, 2> (2 int bits, 14 frac bits)
typedef ap_fixed<16, 2> scale_t;

// Direct multiply (HLS handles Q-format)
acc_t sum = 0;
for (int j = 0; j < N_NODES; j++) {
    if (adj_matrix[i][j] != 0) {
        sum += adj_matrix[i][j] * features[j][f];
    }
}
agg_out[i][f] = (data_t)sum;  // HLS inserts shift
```

**Characteristics**:
- ✅ **Simpler**: No explicit scaling factors
- ⚠️ **Zero-check branch**: `if (adj_matrix[i][j] != 0)` can hurt pipeline
- ⚠️ **Q-format matching**: HLS must align fractional bits

---

## 4. Bias Handling

### INT8 Implementation (Pre-scaled Bias)
```cpp
// Bias stored in accumulator domain (INT32)
typedef ap_int<ACC_BITS> bias_t;

// In Python: bias_int32 = round(bias_float / (scale_in * scale_w))
// In HLS: Just add directly
acc_t acc = bias[o];  // Start with pre-scaled bias
for (int f = 0; f < IN_FEAT; f++) {
    acc += (acc_t)features[n][f] * (acc_t)weights[o][f];
}
// Then scale: (acc * eff_scale_fp + ROUND) >> M
```

**Advantages**:
- ✅ **No extra multiply**: Bias already in correct domain
- ✅ **Efficient**: One fewer operation per output
- ✅ **Matches PyTorch**: Same bias quantization as PyTorch

### Fixed-Point Implementation (Direct Bias)
```cpp
// Bias stored as ap_fixed<16, 4> (same as weights)
typedef ap_fixed<WEIGHT_W, WEIGHT_I> weight_t;

// Direct addition (HLS aligns Q-formats)
acc_t acc = 0;
for (int i = 0; i < IN_FEAT; i++) {
    acc += features[n][i] * weights[o][i];
}
acc += bias[o];  // HLS aligns ap_fixed<32,16> + ap_fixed<16,4>
```

**Characteristics**:
- ✅ **Simpler**: No pre-scaling needed
- ⚠️ **Q-format conversion**: HLS inserts implicit conversion logic

---

## 5. Optimization Capabilities

### INT8 Implementation
```cpp
#ifdef USE_OPTIMIZED_BITWIDTHS
#define ACC_BITS OPT_ACC_BITS     // 22 instead of 32 (10 bits saved)
#define SCALE_BITS OPT_SCALE_BITS // 21 instead of 32 (11 bits saved)
#define MULT_BITS OPT_MULT_BITS   // 43 instead of 64 (21 bits saved)
#endif
```

**Data-driven bit-width optimization** (from `optimize_bitwidths_int8.py`):
- ✅ Analyzes actual numerical ranges during simulation
- ✅ Computes minimum safe widths with margin
- ✅ **Total reduction**: 42 bits in multiplier paths
- ✅ **Validates**: Ensures <1 LSB error after optimization

### Fixed-Point Implementation
```cpp
// Must change W and I parameters
#define ACC_W 28   // Reduce from 32
#define ACC_I 14   // Adjust integer bits
```

**Manual tuning**:
- ⚠️ Must manually adjust W and I for each type
- ⚠️ Less granular control (coupled W and I)
- ⚠️ Harder to validate (no bit-exact reference)

---

## 6. Synthesis Results Comparison

### From `build/hls_comparison.json`:

| Metric | INT8 | Fixed-Point | Difference |
|--------|------|-------------|------------|
| **Latency** | 56 cycles | 77 cycles | Fixed is **37% slower** |
| **DSP** | 6,816 (55.5%) | 6,976 (56.8%) | Fixed uses **2.4% more** |
| **FF** | 569,609 (16.5%) | 262,110 (7.6%) | INT8 uses **2.2× more** |
| **LUT** | 278,289 (16.1%) | 109,536 (6.3%) | INT8 uses **2.5× more** |
| **Clock** | 493 MHz (meets timing) | 602 MHz (meets timing) | Fixed is faster |
| **Accuracy** | 75.7% (0 LSB error) | Unknown | INT8 validated |

**Analysis**:
- **INT8 is faster** (56 vs 77 cycles) due to explicit optimizations
- **Fixed uses fewer FFs/LUTs** because HLS optimizes Q-format logic better
- **INT8 uses similar DSPs** (both need multipliers)
- **INT8 has validation guarantee** (0 LSB error vs PyTorch)

---

## 7. Code Complexity

### INT8 Implementation
```cpp
// More lines, but explicit control
acc_t tmp = 0;
for (int j = 0; j < N_NODES; j++) {
    acc_t prod = (acc_t)adj_matrix[i][j] * (acc_t)features[j][f];
    tmp += prod;
}
mult_t scaled = (mult_t)tmp * (mult_t)beta_fp;
mult_t rounded = scaled + ROUND_CONST;
mult_t result = rounded >> M_BITS;
agg_out[i][f] = int8_clamp(result);
```

**Characteristics**:
- 📈 More lines (~565 lines)
- ✅ Explicit at every step
- ✅ Easy to debug (can printf intermediate values)
- ✅ Full control over every operation

### Fixed-Point Implementation
```cpp
// Fewer lines, implicit conversions
acc_t sum = 0;
for (int j = 0; j < N_NODES; j++) {
    if (adj_matrix[i][j] != 0) {
        sum += adj_matrix[i][j] * features[j][f];
    }
}
agg_out[i][f] = (data_t)sum;  // HLS does the rest
```

**Characteristics**:
- 📉 Fewer lines (~277 lines)
- ⚠️ Implicit conversions (harder to debug)
- ⚠️ Less control (HLS decides rounding modes)
- ⚠️ Branches in loops (`if != 0`)

---

## 8. Validation & Debugging

### INT8 Implementation
```cpp
#ifndef __SYNTHESIS__
printf("  AGG_INT8: node=%d feat=%d tmp=%d scaled=%lld result=%lld out=%d\n",
       i, f, (int)tmp, (long long)scaled, (long long)result, (int)agg_out[i][f]);
#endif
```

**Validation approach**:
1. ✅ **Python integer emulator** (`tests/compare_ptq_float_vs_int8_detailed.py`)
2. ✅ **Bit-exact comparison**: Check every intermediate value
3. ✅ **LSB error metric**: Measure max error (0 LSB at M=24)
4. ✅ **HLS C-simulation**: Run testbench with real test vectors

**Result**: **0 LSB error** achieved and validated

### Fixed-Point Implementation
```cpp
// Limited debug output (ap_fixed prints as decimal)
// Harder to validate bit-exact behavior
```

**Validation approach**:
1. ⚠️ **Functional testing**: Compare output accuracy
2. ⚠️ **No bit-exact reference**: ap_fixed doesn't match PyTorch quantization
3. ⚠️ **Approximate validation**: Check if accuracy is "close enough"

**Result**: Works, but no bit-exact guarantee

---

## 9. Portability & Maintainability

### INT8 Implementation

**Pros**:
- ✅ **Matches PyTorch**: Same quantization scheme as `torch.fake_quantize`
- ✅ **Transferable**: Can apply same approach to other models
- ✅ **Clear semantics**: INT8 quantization is well-defined
- ✅ **Industry standard**: INT8 is common in ML accelerators

**Cons**:
- 📈 More code to write
- 📈 Must handle all scaling manually

### Fixed-Point Implementation

**Pros**:
- ✅ **Simpler code**: HLS handles Q-format
- ✅ **Faster development**: Less manual work

**Cons**:
- ⚠️ **Vendor-specific**: ap_fixed is Xilinx-specific
- ⚠️ **Harder to match PyTorch**: Different quantization semantics
- ⚠️ **Less portable**: Can't easily validate against PyTorch

---

## 10. When to Use Each Approach

### Use INT8 When:
1. ✅ **Bit-exact validation required** (safety-critical, physics experiments)
2. ✅ **Matching PyTorch quantization** (PTQ, QAT workflows)
3. ✅ **Optimizing bit-widths** (need fine-grained control)
4. ✅ **Industry standard INT8** (ML accelerator ecosystem)
5. ✅ **Data-driven optimization** (want to run bit-width analysis)

**Best for**: Production deployments, validated ML pipelines

### Use Fixed-Point When:
1. ✅ **Rapid prototyping** (exploring different Q-formats quickly)
2. ✅ **Non-critical applications** (don't need bit-exact validation)
3. ✅ **Simple models** (small networks, few layers)
4. ✅ **Learning HLS** (easier to understand Q-format concepts)
5. ✅ **Resource-constrained** (want HLS to optimize automatically)

**Best for**: Early exploration, educational purposes

---

## 11. Recommended Workflow

### For Your Project (GNN Track Finder):

**Current choice**: INT8 ✅ **CORRECT**

**Rationale**:
1. ✅ **Physics experiment**: Need validated, trusted results (0 LSB error)
2. ✅ **PTQ workflow**: Already using PyTorch quantization → INT8 matches perfectly
3. ✅ **Data-driven optimization**: Your bit-width tool requires explicit control
4. ✅ **DSE framework**: Need parametric control over all bit-widths
5. ✅ **Latency critical**: 12.5 μs budget → INT8 is faster (56 vs 77 cycles)

### If Starting From Scratch:
1. **Prototype** with Fixed-Point (quick exploration)
2. **Validate** with INT8 (production deployment)
3. **Optimize** with data-driven bit-width analysis (INT8 only)

---

## 12. Key Takeaways

| Criterion | Winner | Reason |
|-----------|--------|--------|
| **Latency** | INT8 (56 cycles) | 37% faster |
| **Validation** | INT8 (0 LSB) | Bit-exact vs PyTorch |
| **PyTorch Matching** | INT8 | Same quantization scheme |
| **Bit-width Control** | INT8 | Explicit, data-driven |
| **Code Simplicity** | Fixed-Point | Fewer lines, implicit |
| **Resource (FF/LUT)** | Fixed-Point | HLS optimizes better |
| **DSP Usage** | Tie | Similar (6,816 vs 6,976) |
| **Development Speed** | Fixed-Point | Faster prototyping |
| **Production Ready** | INT8 | Validated, trusted |

---

## Conclusion

**Your choice of INT8 is correct** for this project because:

1. ✅ **Validation is critical** (physics experiment, trigger system)
2. ✅ **Matches PyTorch** (PTQ workflow, QAT planned)
3. ✅ **Better performance** (37% lower latency)
4. ✅ **Optimization capability** (data-driven bit-width reduction)
5. ✅ **Industry alignment** (INT8 is ML accelerator standard)

**Fixed-Point would be appropriate** for:
- Rapid prototyping (exploring architectures)
- Non-critical applications (no validation requirement)
- Learning HLS (educational purposes)

**For presentation**: Emphasize that INT8 provides **bit-exact validation** and **matches PyTorch quantization**, which are critical for a physics detector application.
