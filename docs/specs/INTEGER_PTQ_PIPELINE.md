# Integer-Only PTQ Pipeline

## Overview

This document explains the complete pipeline from float model training to integer-only HLS implementation.

## Pipeline Stages

```
┌─────────────────────┐
│  1. Train Float     │  python3 train.py
│     Model           │  → build/models/reduced_graphsage_no_root_best.pth
│                     │  → build/weights_float/*.txt (NEW!)
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│  2. PTQ Quantize    │  python3 quantization.py
│     (INT8)          │  → build/quantized_no_root/*.txt (INT8 weights/biases)
│                     │  → build/test_vectors_ptq/*.txt
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│  3. Prepare INT     │  python3 prepare_int8_parameters.py
│     Parameters      │  Uses: weights_float/*.txt (biases)
│                     │        quantized_no_root/ (scales)
│                     │  → build/quantized_int8/bias_*_int32.txt
│                     │  → build/quantized_int8/adj_matrix_int16.txt
│                     │  → build/quantized_int8/int8_params.json
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│  4. Integer         │  python3 integer_ptq_emulator.py
│     Emulator        │  → build/test_vectors_int8/*.txt (reference)
│     (Python)        │
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│  5. Integer HLS     │  Create graphsage_layer_int8.{h,cpp}
│     Implementation  │  No float operations in datapath
│                     │
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│  6. HLS Validation  │  vitis_hls -f csim.tcl
│                     │  Compare with integer emulator (bit-exact)
└─────────────────────┘
```

## Key Changes

### What Changed?

**Before:**
- Only exported INT8 quantized weights/biases
- Had to dequantize to get float biases for INT32 conversion
- Risk of accumulating quantization errors

**After:**
- Export float weights directly from trained model
- Use original float biases for INT32 conversion
- Cleaner, more accurate pipeline

### Why This Matters

**IMPORTANT**: We **DO reuse** the existing PTQ INT8 weights and activations directly! The only thing we need to reconvert from float is **biases**.

#### Why Only Biases?

1. **Weights/Activations (INT8)**: Already in correct domain
   - PTQ quantization: `w_int8 = round(w_float / scale_w)`
   - Integer HLS: Uses exact same `w_int8` values ✓
   - **No reconversion needed!**

2. **Biases (must be INT32)**: Need domain transformation
   - PTQ quantization: `b_int8 = round(b_float / scale_out)` ← Output domain
   - Integer HLS needs: `b_int32 = round(b_float / (scale_in * scale_w))` ← Accumulator domain
   - These are **different scales**! Cannot reuse PTQ INT8 bias.

#### The Accumulator Domain Problem

In integer-only HLS, the bias is added to the accumulator **before** requantization:

```c++
// Integer-only linear layer:
int32_t acc = 0;
for (int i = 0; i < dim; i++) {
    acc += x_int8[i] * w_int8[i];  // Accumulator scale = scale_x * scale_w
}
acc += bias_int32;  // Must be in accumulator domain!

// Requantize to output domain:
output_int8 = (acc * eff_scale_fp) >> M;  // Now in scale_out domain
```

But PTQ quantized the bias to the **output** domain:
```python
# PTQ does this:
bias_int8 = round(bias_float / scale_out)  # Output domain, wrong for accumulator!
```

So we need to start from float and convert to accumulator domain:
```python
# Correct for integer HLS:
bias_int32 = round(bias_float / (scale_x * scale_w))  # Accumulator domain ✓
```

3. **Cannot dequantize PTQ bias**: Would accumulate quantization error:
   ```
   BAD:  b_float → PTQ → b_int8 (output domain) → dequant → b_float' → convert → b_int32 (wrong!)
   GOOD: b_float → convert directly → b_int32 (accumulator domain) ✓
   ```

## File Locations

### After Training (Step 1)
```
build/
  models/
    reduced_graphsage_no_root_best.pth     # Model checkpoint
  weights_float/                            # NEW: Float weights
    conv1_lin_l_weight_no_root.txt         # Layer 1 weights (float32)
    conv1_lin_l_bias_no_root.txt           # Layer 1 bias (float32) ← USED
    conv2_lin_l_weight_no_root.txt         # Layer 2 weights (float32)
    conv2_lin_l_bias_no_root.txt           # Layer 2 bias (float32) ← USED
    *.shape                                # Shape metadata
```

### After PTQ (Step 2)
```
build/
  quantized_no_root/
    conv1_lin_l_weight.txt                 # INT8 quantized weights
    conv1_lin_l_bias.txt                   # INT8 quantized biases
    conv2_lin_l_weight.txt                 # INT8 quantized weights
    conv2_lin_l_bias.txt                   # INT8 quantized biases
    quant_params.json                      # Scales and zero_points ← USED
  test_vectors_ptq/
    scales.txt                             # Activation/weight scales ← USED
    network_input.txt                      # INT8 test input
    adj_matrix.txt                         # Float adjacency ← USED
    network_output_reference.txt           # INT8 reference output
```

### After INT8 Preparation (Step 3)
```
build/
  quantized_int8/                          # NEW: Integer-only parameters
    bias_layer1_int32.txt                  # From weights_float/conv1_lin_l_bias
    bias_layer2_int32.txt                  # From weights_float/conv2_lin_l_bias
    adj_matrix_int16.txt                   # From test_vectors_ptq/adj_matrix.txt
    int8_params.json                       # All fixed-point scale factors
```

## Usage

### Step 1: Train and Export Float Weights

```bash
cd src
python3 train.py
```

This will:
- Train reduced_graphsage_no_root model
- Save checkpoint to `build/models/`
- **Export float weights to `build/weights_float/`** ← NEW

### Step 2: PTQ Quantization

```bash
cd src
python3 quantization.py
```

This will:
- Load float model
- Compute quantization scales
- Export INT8 weights/biases to `build/quantized_no_root/`
- Generate PTQ test vectors in `build/test_vectors_ptq/`

### Step 3: Prepare Integer Parameters

```bash
cd src
python3 prepare_int8_parameters.py
```

This will:
- Load **float biases** from `weights_float/` ← Uses original floats
- Load quantization scales from `quantized_no_root/quant_params.json`
- Load activation scales from `test_vectors_ptq/scales.txt`
- Convert biases to INT32 accumulator domain
- Quantize adjacency to INT16 fixed-point
- Compute fixed-point scale factors (M=20, K=4096)
- Save all to `build/quantized_int8/`

### Step 4: Generate Integer Test Vectors

```bash
cd tests
python3 integer_ptq_emulator.py
```

This will:
- Load integer-only parameters
- Run bit-exact integer-only forward pass
- Compare with PTQ float reference
- Save test vectors to `build/test_vectors_int8/`

### Step 5: Implement Integer HLS

Create `hls/graphsage_layer_int8.{h,cpp}` with:
- INT8 weights and activations
- INT16 adjacency matrix
- INT32 accumulators and biases
- INT64 for intermediate multiply-accumulate
- Fixed-point scale factors
- No floating-point operations in datapath

### Step 6: Validate HLS

```bash
cd build/hls/graphsage_int8
vitis_hls -f csim.tcl
```

Expected: **Bit-exact match** with Python integer emulator

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| M | 20 | Fractional bits for scale factors |
| K | 4096 (2^12) | Fixed-point scale for adjacency |
| scale_in | 0.00486 | Input activation scale |
| scale_hidden | 0.1 | Hidden activation scale |
| scale_out | 0.1 | Output activation scale |
| scale_w1 | 0.0112 | Layer 1 weight scale |
| scale_w2 | 0.0108 | Layer 2 weight scale |

## Verification Checklist

Before HLS implementation:
- [ ] Float weights exported after training
- [ ] PTQ quantization completed
- [ ] INT32 biases computed from float (not dequantized)
- [ ] INT16 adjacency generated
- [ ] Fixed-point scale factors computed
- [ ] Python integer emulator matches PTQ reference
- [ ] Integer test vectors generated

After HLS implementation:
- [ ] HLS C-sim passes
- [ ] HLS output matches Python integer emulator bit-exactly
- [ ] Synthesis successful
- [ ] Resource usage within budget
- [ ] Timing requirements met

## Expected Accuracy

- **Python Integer Emulator vs PTQ Float**: May differ by 1-3 LSB due to fixed-point approximation
- **HLS vs Python Integer Emulator**: **Bit-exact match** (0 LSB difference)

## Troubleshooting

### Error: "Float weights directory not found"
**Solution**: Run `python3 train.py` first to train model and export float weights

### Error: "scales.txt not found"
**Solution**: Run `python3 quantization.py` to generate PTQ parameters

### Accuracy mismatch between integer and PTQ
**Expected**: 1-3 LSB difference due to fixed-point approximation of adjacency and scale factors. This is acceptable.

### HLS doesn't match Python emulator
**Problem**: Bug in HLS implementation. Both should use identical integer arithmetic.
