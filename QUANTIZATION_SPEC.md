# GraphSAGE FPGA Quantization Specification

**Version**: 1.0  
**Date**: November 24, 2025  
**Status**: Phase 1 - Float/Wide Precision Validation

---

## 1. Overview

This document defines the exact quantization scheme used for the GraphSAGE GNN implementation on FPGA. All tools (Python, HLS, verification) must follow this specification to ensure bit-exact reproducibility.

---

## 2. Current Implementation Status

### Phase 1: Float Validation (CURRENT)
- **Goal**: Validate graph structure and SAGEConv logic without quantization noise
- **Python Reference**: PyG SAGEConv in float32 (VALIDATED ✓)
- **HLS Target**: Float or ap_fixed<32,16> (IN PROGRESS)
- **Comparison**: HLS float ↔ PyG float (should match exactly)

### Phase 2: Hardware-Style INT8 (NEXT)
- **Goal**: Implement quantized inference with documented spec
- **Python Reference**: Custom INT8 GraphSAGE (TO BE CREATED)
- **HLS Target**: INT8/INT32 quantized (TO BE UPDATED)
- **Comparison**: HLS INT8 ↔ Python INT8 (should match exactly)

### Phase 3: QAT Optimization (FUTURE)
- **Goal**: Retrain model for quantization-friendly weights
- **If accuracy drop > 5%**: Apply Quantization-Aware Training
- **If accuracy acceptable**: Skip QAT

---

## 3. Network Architecture

### Model: ReducedGraphSAGE (no root_weight)
```
Input: 1433 features (Cora dataset)
  ↓
Projection: Linear(1433 → 16) + ReLU
  ↓
Layer 1: SAGEConv(16 → 24, aggr='mean', root_weight=False)
  ↓
ReLU
  ↓
Layer 2: SAGEConv(24 → 7, aggr='mean', root_weight=False)
  ↓
Output: 7 classes
```

### SAGEConv Formula (root_weight=False)
```
h_agg[i] = mean(h[j] for j in neighbors(i))  # Mean aggregation
out[i] = W_l @ h_agg[i] + b_l                # Linear transform only
```

**Note**: With `root_weight=False`, there is NO `W_r @ h[i]` term, simplifying hardware implementation.

---

## 4. Quantization Specification (Phase 2)

### 4.1 Data Types

| Tensor Type | Format | Bits | Range | Notes |
|-------------|--------|------|-------|-------|
| Input Features | INT8 | 8 | [-128, 127] | Symmetric quantization |
| Weights (W) | INT8 | 8 | [-128, 127] | Symmetric, per-tensor |
| Biases (b) | INT32 | 32 | - | In accumulator domain |
| Activations | INT8 | 8 | [-128, 127] | Per-layer scale |
| Accumulator | INT32 | 32 | - | For matmul results |

### 4.2 Quantization Formula

#### Symmetric Quantization (zero_point = 0)
```python
# Forward: float → int8
scale = max(abs(tensor_min), abs(tensor_max)) / 127
quantized = clamp(round(tensor_float / scale), -128, 127)

# Inverse: int8 → float  
dequantized = quantized_int8 * scale
```

### 4.3 Bias Quantization (Accumulator Domain)

Biases are stored in INT32 in the accumulator domain to avoid requantization during addition:

```python
# Given: b_float (original bias in FP32)
#        scale_input (input activation scale)
#        scale_weight (weight scale)

b_int32 = round(b_float / (scale_input * scale_weight))
```

**Rationale**: After `acc = X_int8 @ W_int8`, the accumulator has scale `scale_input * scale_weight`. By pre-scaling the bias to this domain, we can do `acc + b_int32` directly without requantization.

### 4.4 Requantization Formula

After computing `acc_int32 = X_int8 @ W_int8 + b_int32`, we requantize to the next layer's scale:

```python
# acc_int32 is in scale domain: scale_input_prev * scale_weight
# We want output in scale domain: scale_output

y_float = acc_int32 * (scale_input_prev * scale_weight)  # Dequantize to float
y_int8 = clamp(round(y_float / scale_output), -128, 127)  # Quantize to output scale
```

**Simplified formula**:
```python
requant_scale = (scale_input_prev * scale_weight) / scale_output
y_int8 = clamp(round(acc_int32 * requant_scale), -128, 127)
```

### 4.5 Layer-by-Layer Scales

| Layer | Input Scale | Weight Scale | Output Scale | Notes |
|-------|-------------|--------------|--------------|-------|
| Projection | `scale_in` | - | - | Computed from input features |
| Conv1 | `scale_in` | `scale_w1` | `scale_hidden` | Fixed at 0.1 |
| Conv2 | `scale_hidden` | `scale_w2` | `scale_out` | Computed from output range |

**Current Values** (from test vectors, may vary per run):
```
scale_in:     0.005586 (from quantized input features)
scale_w1:     0.012327 (from conv1 weights)
scale_w2:     0.011364 (from conv2 weights)  
scale_hidden: 0.1      (FIXED - for hidden activations)
scale_out:    ~0.08-0.10 (computed dynamically per subgraph)
```

---

## 5. Forward Pass Execution (Phase 2 Target)

### 5.1 Layer 1: SAGEConv(16 → 24)

```python
# Input: x_int8 (N × 16), adj_matrix (N × N)
# Weights: w1_int8 (24 × 16), b1_int32 (24,)

# Step 1: Dequantize input
x_float = x_int8 * scale_in

# Step 2: Aggregate neighbors (mean)
agg_float = adj_matrix @ x_float  # Adjacency already normalized

# Step 3: Quantize aggregated features
agg_int8 = quantize(agg_float, scale_hidden)

# Step 4: Integer linear transform
acc1_int32 = agg_int8.astype(int32) @ w1_int8.T.astype(int32) + b1_int32

# Step 5: Requantize to scale_hidden
requant_scale_1 = (scale_hidden * scale_w1) / scale_hidden  # = scale_w1
hidden_float = acc1_int32 * requant_scale_1
hidden_int8 = quantize(hidden_float, scale_hidden)

# Step 6: ReLU (in INT8 domain)
hidden_int8 = clamp(hidden_int8, 0, 127)
```

### 5.2 Layer 2: SAGEConv(24 → 7)

```python
# Input: hidden_int8 (N × 24)
# Weights: w2_int8 (7 × 24), b2_int32 (7,)

# Step 1: Dequantize hidden
hidden_float = hidden_int8 * scale_hidden

# Step 2: Aggregate neighbors
agg2_float = adj_matrix @ hidden_float

# Step 3: Quantize aggregated features  
agg2_int8 = quantize(agg2_float, scale_hidden)

# Step 4: Integer linear transform
acc2_int32 = agg2_int8.astype(int32) @ w2_int8.T.astype(int32) + b2_int32

# Step 5: Requantize to scale_out
requant_scale_2 = (scale_hidden * scale_w2) / scale_out
output_float = acc2_int32 * requant_scale_2
output_int8 = quantize(output_float, scale_out)

# Output: output_int8 (N × 7)
```

---

## 6. HLS Implementation Guidelines (Phase 1)

### Phase 1 Target: Float/Wide Precision

**Objective**: Match PyG SAGEConv float output exactly, WITHOUT quantization.

#### Data Types (Phase 1)
```cpp
typedef float data_t;           // For activations
typedef float weight_t;         // For weights  
typedef float acc_t;            // For accumulation

// OR use wide fixed-point:
typedef ap_fixed<32,16> data_t;
typedef ap_fixed<32,16> weight_t;
typedef ap_fixed<48,24> acc_t;
```

#### Aggregation Function
```cpp
void aggregate_neighbors(
    data_t h_in[MAX_NODES][FEATURES],
    float adj[MAX_NODES][MAX_NODES],
    data_t h_agg[MAX_NODES][FEATURES],
    int num_nodes,
    int num_features
) {
    // Mean aggregation: h_agg[i] = sum(adj[i,j] * h_in[j]) for all j
    // Adjacency matrix already row-normalized
    for (int i = 0; i < num_nodes; i++) {
        for (int f = 0; f < num_features; f++) {
            acc_t sum = 0;
            for (int j = 0; j < num_nodes; j++) {
                sum += adj[i][j] * h_in[j][f];
            }
            h_agg[i][f] = sum;
        }
    }
}
```

#### Linear Transform
```cpp
void linear_transform(
    data_t h_in[MAX_NODES][FEATURES_IN],
    weight_t W[FEATURES_OUT][FEATURES_IN],
    weight_t b[FEATURES_OUT],
    data_t h_out[MAX_NODES][FEATURES_OUT],
    int num_nodes
) {
    for (int i = 0; i < num_nodes; i++) {
        for (int out = 0; out < FEATURES_OUT; out++) {
            acc_t acc = b[out];
            for (int in = 0; in < FEATURES_IN; in++) {
                acc += h_in[i][in] * W[out][in];
            }
            h_out[i][out] = acc;
        }
    }
}
```

#### Complete Layer
```cpp
void graphsage_layer(
    data_t x[MAX_NODES][FEATURES_IN],
    float adj[MAX_NODES][MAX_NODES],
    weight_t W[FEATURES_OUT][FEATURES_IN],
    weight_t b[FEATURES_OUT],
    data_t out[MAX_NODES][FEATURES_OUT],
    int num_nodes
) {
    data_t h_agg[MAX_NODES][FEATURES_IN];
    
    // Step 1: Aggregate neighbors
    aggregate_neighbors(x, adj, h_agg, num_nodes, FEATURES_IN);
    
    // Step 2: Linear transform
    linear_transform(h_agg, W, b, out, num_nodes);
    
    // Note: ReLU applied by caller between layers
}
```

**Verification**: Compare HLS C-sim output against Python testbench's `pyg_exact_reference()` or `pyg_forward_pass()`.

---

## 7. Verification Strategy

### Phase 1 Verification (CURRENT)

**Test**: HLS float vs PyG float
```bash
cd hls && vitis_hls -f run_csim.tcl
cd tests && python python_testbench.py
```

**Expected Result**: 
- HLS output matches PyG reference **exactly** (max diff ≤ 1e-5 for float, or ≤1 for ap_fixed)
- Zero quantization noise

**Debug if not matching**:
1. Check adjacency matrix normalization
2. Verify edge_index ordering
3. Check weight/bias loading
4. Verify aggregation formula (mean vs sum)

### Phase 2 Verification (NEXT)

**Test**: HLS INT8 vs Python hardware-style INT8
```bash
cd hls && vitis_hls -f run_csim.tcl
cd tests && python hardware_golden_int8.py
```

**Expected Result**:
- HLS INT8 output matches Python INT8 golden reference **exactly** (bit-exact)
- Both use identical quantization formulas

**Debug if not matching**:
1. Check requantization scale computation
2. Verify INT32 accumulator width
3. Check rounding mode (round-to-nearest)
4. Verify clipping bounds

### Phase 3 Verification (QAT)

**Test**: Accuracy on Cora validation set
```bash
cd tests && python evaluate_accuracy.py --quantized
```

**Acceptance Criteria**:
- Quantized accuracy ≥ 70% (Cora baseline ~75%)
- If drop > 5%: Apply QAT and retrain

---

## 8. Test Vector Specification

### Reproducibility Requirements

All test vector generation must be **deterministic** and **reproducible**:

```python
# In tests/generate_test_vectors.py:
torch.manual_seed(42)
np.random.seed(42)

# Subgraph extraction:
subgraph_data = extract_fixed_subgraph(
    data, 
    num_nodes=32, 
    center_node=0,  # FIXED, not random
    num_hops=2
)
```

### Test Vector Files

| File | Format | Description |
|------|--------|-------------|
| `adj_matrix.txt` | Float, space-delimited | Row-normalized adjacency (N×N) |
| `edge_index.txt` | Int, space-delimited | COO format edges (E×2) |
| `network_input.txt` | INT8, space-delimited | Projected input features (N×16) |
| `weights_layer1.txt` | INT8, space-delimited | Conv1 weights (24×16) |
| `bias_layer1.txt` | INT32, space-delimited | Conv1 biases (24,) |
| `weights_layer2.txt` | INT8, space-delimited | Conv2 weights (7×24) |
| `bias_layer2.txt` | INT32, space-delimited | Conv2 biases (7,) |
| `scales.txt` | Float, key:value | All quantization scales |
| `network_output_reference.txt` | INT8, space-delimited | PyG quantized output (N×7) |

### Reference Generation Method

The reference output is generated using:
```python
# Run PyG SAGEConv in FLOAT
out_float = model.conv2(relu(model.conv1(x_float, edge_index)), edge_index)

# Quantize ONLY the final output
out_int8, scale_out, _ = quantize_tensor(out_float, num_bits=8)
```

**Important**: The reference is NOT from INT8 intermediate layers. It's float→float→quantize_final.

---

## 9. Common Pitfalls and Solutions

### ❌ Comparing HLS INT8 vs PyG Float
**Problem**: These use different numeric domains.  
**Solution**: Compare HLS INT8 vs Python INT8 (Phase 2).

### ❌ Non-deterministic Subgraph
**Problem**: Random center node causes different test vectors each run.  
**Solution**: Fixed seed + fixed center_node=0.

### ❌ Edge Ordering Mismatch
**Problem**: Reconstructing edge_index from adjacency gives different order.  
**Solution**: Save edge_index.txt and load it directly.

### ❌ Scale Mismatch
**Problem**: Reference generated with different scale than loaded.  
**Solution**: Regenerate all test vectors from scratch after changing model.

### ❌ Bias Not in Accumulator Domain
**Problem**: Adding float bias to INT32 accumulator.  
**Solution**: Pre-scale bias: `b_int32 = b_float / (scale_in * scale_w)`.

---

## 10. Next Steps

### Immediate (Phase 1)
- [ ] Update `hls/graphsage_layer.cpp` to use float/ap_fixed
- [ ] Remove INT8 quantization from HLS aggregation
- [ ] Run HLS C-sim and verify exact match with PyG
- [ ] Document any deviations in this spec

### Short-term (Phase 2)
- [ ] Create `tests/hardware_golden_int8.py` with integer-only math
- [ ] Implement quantize/dequantize/requantize per spec above
- [ ] Verify Python INT8 matches reference (within ±1 due to rounding)
- [ ] Update HLS to INT8 matching Python INT8 exactly

### Long-term (Phase 3)
- [ ] Measure quantized accuracy on Cora validation set
- [ ] If accuracy drop > 5%: Implement QAT in training pipeline
- [ ] Fine-tune scale_hidden and per-layer scales
- [ ] Synthesize and measure FPGA resource utilization

---

## 11. References

- **PyTorch Geometric SAGEConv**: [torch_geometric.nn.SAGEConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SAGEConv)
- **Quantization Methods**: `src/quantization.py` (symmetric per-tensor)
- **Test Vectors**: `tests/generate_test_vectors.py`
- **Python Testbench**: `tests/python_testbench.py`
- **HLS Implementation**: `hls/graphsage_layer.cpp`

---

**Document Owner**: GraphSAGE FPGA Team  
**Last Updated**: November 24, 2025  
**Next Review**: After Phase 1 completion
