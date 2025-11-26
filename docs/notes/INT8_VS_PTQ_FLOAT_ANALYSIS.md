# INT8-Only vs PTQ-Float: Error Analysis

## Overview

When comparing the **PTQ-float** (Post-Training Quantization with float quant/dequant operations) and **INT8-only** (pure integer arithmetic with fixed-point scales) implementations, we observe different error levels depending on the configuration:

| Configuration | Max Error | Root Cause |
|---------------|-----------|------------|
| **M=20 (current)** | **13 LSB** | 2.7% beta approximation error |
| **M=24** | **2-3 LSB** | Rounding method difference only |
| **M=24 + HW round in PTQ** | **0 LSB** | **PERFECT MATCH!** |

This document explains where these errors come from and how to achieve exact match.

## KEY FINDING: Perfect Match is Achievable

With the right configuration:
```
M = 24  (4 extra bits of precision)
PTQ-float uses HW-style rounding (round half up)
```

**Result: INT8-only matches PTQ-float EXACTLY (0 LSB error)**

## The Two Implementations

### PTQ-Float Implementation
- Uses INT8 weights and activations
- Uses **float** scale factors for requantization
- Uses **Python's round()** by default (banker's rounding: round half to even)
- Can use **--hw-round** flag to match hardware behavior
- Performs: `output = round(accumulator * scale_factor)`

### INT8-Only Implementation  
- Uses INT8 weights and activations
- Uses **fixed-point** scale factors (M fractional bits)
- Uses **hardware rounding** (round half up: `(x + 0.5) >> shift`)
- Performs: `output = (accumulator * eff_scale_fp + (1 << (M-1))) >> M`

## Error Analysis with M=20 (Current Configuration)

### Scale Approximation Errors

```
beta1_fp = 14, beta1_true = 1.2999e-5
Relative error: 2.71%  ← This is the main problem!

eff_scale1_fp = 11206, scale_w1 = 0.01069
Relative error: 0.00092%  ← Good
```

The **2.7% error in beta1** causes values to cross rounding boundaries:
- 6 errors in Layer 1 Aggregation
- These propagate and amplify through the network
- Final result: **13 LSB max error**

## Error Analysis with M=24 (Recommended)

### Scale Approximation Errors

```
beta1_fp = 218, beta1_true = 1.2999e-5
Relative error: 0.042%  ← 65x better!

beta2_fp = 4096 = K (EXACT, since beta2 = 1/K)
Relative error: 0.000%  ← Perfect!

eff_scale1_fp = 179294, scale_w1 = 0.01069
Relative error: 0.00019%  ← Excellent
```

### With M=24 + Default Python Rounding

With M=24 but Python's banker's rounding:
- **Layer 1 Agg: 0 errors** (beta1 approximation good enough)
- **Layer 1 Linear: 0 errors**
- **Hidden layer: 0 differences**
- **Layer 2 Agg: 8 errors** (all from rounding boundaries)
- **Final output: 7 errors, max 2-3 LSB**

### With M=24 + HW Rounding (--hw-round flag)

When PTQ-float uses the same HW-style rounding:
- **ALL STAGES: 0 ERRORS**
- **Final output: PERFECT MATCH**

**Verified output comparison:**
```
INT8-Only Output (M=24):
  -3  -27   -2   51  -14    9  -56
  -1  -14    2   36    3  -10  -54
   0  -27   -8   48  -19   13  -47
 -11  -22    8   51  -12    6  -64
   0  -21   -5   43   -9    2  -50
  -3  -19    5   45    6  -11  -64
   1  -26   -1   51    2   -5  -61
  -5   -7  -13   26  -28   14  -33

PTQ-Float Reference (HW round):
  (IDENTICAL OUTPUT)

Difference: ALL ZEROS
Max error: 0
Non-zero errors: 0/56
```

### Root Cause of Errors When Using Banker's Rounding

All errors occur at values **EXACTLY on the 0.5 boundary**:

```
Location    Pre-quant Value    Python round()    HW round    Diff
[5,2]       1.500000           2 (to even)       2           0
[5,7]       6.500000           6 (to even)       7          -1
[5,21]      8.500000           8 (to even)       9          -1
[6,2]       0.500000           0 (to even)       1          -1
[6,10]      9.500000           10 (to even)      10          0
[6,16]      8.500000           8 (to even)       9          -1
[6,21]      9.500000           10 (to even)      10          0
[6,22]      9.500000           10 (to even)      10          0
```

**Why these exact .5 values occur:**
- beta2_fp = K = 4096 is EXACT (no approximation error)
- The aggregation produces: `sum(adj_int16[i,j] * hidden[j,f]) * K / (K * 2^M)`
- When the weighted sum equals `4096*X + 2048`, result is exactly `X.5`

**Rounding behavior difference:**
| Value | Python round() | HW round | Notes |
|-------|----------------|----------|-------|
| 0.5   | 0              | 1        | Python rounds to even (0) |
| 1.5   | 2              | 2        | Both round up |
| 2.5   | 2              | 3        | Python rounds to even (2) |
| 6.5   | 6              | 7        | Python rounds to even (6) |
| 8.5   | 8              | 9        | Python rounds to even (8) |
| 9.5   | 10             | 10       | Both round up |

### How Errors Propagate to Output

```
Layer 2 Agg: 8 values differ by ±1
    ↓ multiply by weights (up to ±50)
    ↓ sum 24 features
Layer 2 Lin: accumulators differ by up to 250
    ↓ multiply by eff_scale2
Final Output: 7 values differ by up to 3 LSB
```

Example worst case (Node 6, Output 4, diff = -3):
```
Agg2 differences at node 6: [(2,-1), (10,-1), (16,-1), (21,-1), (22,-1)]
acc_diff = -250 (5 features × weights)
After scaling: -3 LSB difference
```

## Summary: Error Sources by Configuration

### M=20 Configuration
| Layer | Max Error | Cause |
|-------|-----------|-------|
| L1 Agg | ±1 | 2.7% beta1 error crosses rounding boundaries |
| L1 Lin | ±2 | Accumulated from L1 Agg + bias differences |
| Hidden | ±2 | 78/192 values differ |
| L2 Agg | ±1 | Aggregates different hidden values |
| L2 Lin | **±13** | 24 inputs × weights × scale |

### M=24 Configuration (without HW round)
| Layer | Max Error | Cause |
|-------|-----------|-------|
| L1 Agg | **0** | beta1 error reduced to 0.04% |
| L1 Lin | **0** | No input differences |
| Hidden | **0** | No differences |
| L2 Agg | ±1 | **Rounding method only** (8 exact .5 cases) |
| L2 Lin | **±3** | 8 inputs × weights × scale |

### M=24 + HW Round Configuration (RECOMMENDED)
| Layer | Max Error | Cause |
|-------|-----------|-------|
| L1 Agg | **0** | beta1 error sufficient |
| L1 Lin | **0** | Perfect match |
| Hidden | **0** | Perfect match |
| L2 Agg | **0** | Rounding methods match |
| L2 Lin | **0** | **PERFECT MATCH** |

## Recommendations

### ✅ RECOMMENDED: M=24 + HW Rounding in PTQ-Float

**Configuration:**
```python
M = 24  # 4 extra bits of precision (20 → 24)
USE_HW_ROUND = True  # Use --hw-round flag
```

**Fixed-point scales for M=24:**
```python
beta1_fp = 218              # vs 14 for M=20
beta2_fp = 4096 (= K)       # EXACT
eff_scale1_fp = 179,294     # vs 11,206 for M=20
eff_scale2_fp = 176,976     # vs 11,061 for M=20
```

**Results:**
- **Max error: 0 LSB**
- **Non-zero errors: 0/56**
- **INT8-only EXACTLY matches PTQ-float**

**Trade-offs:**
- +4 bits in multiplier (24-bit vs 20-bit scale factors)
- Negligible hardware cost increase
- Perfect bit-exact verification

### Alternative: M=20 (Current Configuration)

If 13 LSB error is acceptable:
- Keep M=20
- HLS will match INT8-only Python exactly
- Error is vs PTQ-float reference only
- Smaller multipliers

### How to Use

**Generate PTQ-float test vectors with HW rounding:**
```bash
cd tests
python generate_test_vectors_ptq_float.py --hw-round
```

**Compare with HW rounding:**
```bash
python compare_ptq_float_vs_int8_detailed.py --hw-round --network
```

**Update HLS for M=24:**
- Change `M` constant from 20 to 24
- Regenerate fixed-point weights with M=24
- Update bit widths for scale factors

## Conclusion

**The 13 LSB error was NOT fundamental.** It was caused by:
1. Insufficient precision in beta1_fp (M=20 gave 2.7% error)
2. Different rounding methods (Python banker's vs HW round half up)

**Solution:** Use M=24 + HW rounding → **PERFECT MATCH (0 LSB error)**

### Option C: Keep M=20
- HLS matches INT8-only Python exactly (0 error)
- Accept 13 LSB difference vs PTQ-float
- Acceptable if PTQ-float is just a reference, not golden

## Classification Accuracy

Despite the differences, classification remains unchanged:
```python
# M=20
PTQ-Float output[7] = [-5, -7, -13, 26, -28, 14, -33]  # argmax = 3
INT8-Only output[7] = [-6, -9, -4, 31, -15, 4, -45]    # argmax = 3 ✓

# M=24
PTQ-Float output[7] = [-5, -7, -13, 26, -28, 14, -33]  # argmax = 3
INT8-Only output[7] = [-5, -7, -13, 26, -28, 14, -33]  # argmax = 3 ✓ (same!)
```

## Configuration Parameters

### M=20 (Current)
```
M = 20, K = 4096
beta1_fp = 14      (2.7% error)
beta2_fp = 256     (0% error)
eff_scale1_fp = 11206
eff_scale2_fp = 11061
```

### M=24 (Recommended)
```
M = 24, K = 4096
beta1_fp = 218     (0.04% error)
beta2_fp = 4096    (0% error, = K exactly)
eff_scale1_fp = 179294
eff_scale2_fp = 176976
```

---

## VERIFIED FINAL CONFIGURATION

**Date:** Successfully verified

### Commands Used:
```bash
# 1. Regenerate INT8 parameters with M=24
python src/prepare_ptq_int8_parameters.py --m24

# 2. Regenerate PTQ-Float test vectors with HW rounding
python tests/generate_test_vectors_ptq_float.py --hw-round

# 3. Regenerate INT8-only test vectors
python tests/generate_test_vectors_ptq_int8.py

# 4. Verify exact match
python tests/compare_ptq_float_vs_int8_detailed.py --hw-round
```

### Results:
```
Layer                          Max Error    Non-zero     Total Elements
-------------------------------------------------------------------
Layer 1 Aggregation            0            0            128
Layer 1 Linear                 0            0            192
Hidden (ReLU)                  0            0            192
Layer 2 Aggregation            0            0            192
Layer 2 Linear (Output)        0            0            56

✅ PERFECT BIT-EXACT MATCH ACHIEVED!
   - 56/56 outputs match exactly
   - 0 LSB error across all layers
```

### HLS Configuration:
The pure INT8 HLS is ready with M=24 as default:
- `hls/graphsage_layer_int8.h` - Type definitions
- `hls/graphsage_layer_int8.cpp` - Implementation
- `hls/testbench_int8.cpp` - Testbench
- `hls/Makefile.int8` - Build system

Build:
```bash
cd hls
make -f Makefile.int8  # Uses M_BITS=24 by default
./test_int8
```
