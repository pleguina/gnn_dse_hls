# Test Vector Generator Scripts Comparison

## Current Scripts

### 1. generate_test_vectors.py (257 lines)
**Status**: ❌ **WRONG - Uses float forward pass**

**What it does**:
```python
# Runs FLOAT PyTorch model
x = torch.from_numpy(features).float()
out = model.conv1(x, edge_index)  # <- FLOAT operations
out = torch.relu(out)
# Then quantizes output
out_quant, scale_out, _ = quantize_tensor(out, num_bits=8)
```

**Problem**: This is NOT true PTQ! It runs the float model and quantizes the result.

**Output**: `build/test_vectors/` (not used for HLS)

---

### 2. generate_ptq_test_vectors.py (201 lines)
**Status**: ⚠️ **PARTIAL - Has quantized forward but uses edge_index**

**What it does**:
```python
def quantized_aggregate(x_int8, edge_index, adj_matrix, num_nodes, ...):
    x_float = x_int8.float() * scale_in  # Dequantize
    # Manual loop using edge_index
    for i in range(num_nodes):
        agg_float[i] = torch.sum(x_float * adj_matrix[:, i:i+1], dim=0)
    agg_int8 = torch.round(agg_float / scale_hidden).clamp(-128, 127)
```

**Issues**:
- Uses `edge_index` (not needed, complicates things)
- Manual loop aggregation (slower, more complex)
- Still correct but less clean

**Output**: Not clear which directory it outputs to

---

### 3. generate_ptq_test_vectors_clean.py (294 lines)
**Status**: ✅ **CORRECT - Clean quantized forward pass**

**What it does**:
```python
def quantized_aggregate(x_int8, adj_matrix, scale_in, scale_hidden):
    x_float = x_int8.float() * scale_in  # Dequantize
    adj_tensor = torch.from_numpy(adj_matrix).float()
    agg_float = torch.matmul(adj_tensor, x_float)  # Matrix multiply
    agg_int8 = torch.round(agg_float / scale_hidden).clamp(-128, 127)
```

**Advantages**:
- Uses adjacency matrix directly (matches HLS)
- Clean matrix multiplication
- Explicit "quantized forward pass" implementation
- Outputs intermediate values for debugging

**Output**: `build/test_vectors_ptq/`

**This is the CORRECT one to use!**

---

### 4. generate_test_vectors_float.py
**Status**: ✅ **CORRECT - For float baseline**

**What it does**:
- Generates float (non-quantized) test vectors
- For float HLS validation

**Output**: `build/test_vectors_float/`

---

### 5. integer_ptq_emulator.py
**Status**: ✅ **CORRECT - For integer-only PTQ**

**What it does**:
- Pure integer arithmetic (no float dequant/quant)
- Fixed-point aggregation with INT16 adjacency
- Uses M=20, K=4096 parameters

**Output**: `build/test_vectors_int8/`

---

## Summary

| Script | Purpose | Status | Keep? |
|--------|---------|--------|-------|
| generate_test_vectors.py | Float model → quantize output | ❌ Wrong | Delete |
| generate_ptq_test_vectors.py | PTQ quantized forward (edge_index) | ⚠️ Suboptimal | Delete |
| **generate_ptq_test_vectors_clean.py** | **PTQ quantized forward (clean)** | ✅ **CORRECT** | **KEEP** |
| generate_test_vectors_float.py | Float baseline | ✅ Correct | Keep |
| integer_ptq_emulator.py | Integer-only PTQ | ✅ Correct | Keep |

---

## Recommendation

### Keep These 3 Scripts:

1. **generate_test_vectors_float.py** → `build/test_vectors_float/`
   - Float baseline (no quantization)

2. **generate_ptq_test_vectors_clean.py** → `build/test_vectors_ptq/`
   - Rename to: `generate_test_vectors_ptq_float.py`
   - PTQ with float dequant/quant operations

3. **integer_ptq_emulator.py** → `build/test_vectors_int8/`
   - Rename to: `generate_test_vectors_ptq_int8.py`
   - PTQ with integer-only operations

### Delete These:

- `generate_test_vectors.py` (wrong implementation)
- `generate_ptq_test_vectors.py` (duplicate, less clean)

---

## Proposed Renaming

```bash
# Keep and rename
mv generate_test_vectors_float.py         generate_test_vectors_float.py        # No change
mv generate_ptq_test_vectors_clean.py     generate_test_vectors_ptq_float.py    # Rename for clarity
mv integer_ptq_emulator.py                 generate_test_vectors_ptq_int8.py     # Rename for consistency

# Delete
rm generate_test_vectors.py
rm generate_ptq_test_vectors.py
```

After this, we have a clean structure:
- `generate_test_vectors_float.py` → float baseline
- `generate_test_vectors_ptq_float.py` → PTQ with float ops
- `generate_test_vectors_ptq_int8.py` → PTQ with integer ops
- (Future) `generate_test_vectors_qat.py` → QAT

---

## Why generate_ptq_test_vectors_clean.py is Better

### Comparison of quantized_aggregate():

**generate_ptq_test_vectors.py** (complex):
```python
def quantized_aggregate(x_int8, edge_index, adj_matrix, num_nodes, scale_in, scale_hidden):
    x_float = x_int8.float() * scale_in
    agg_float = torch.zeros((num_nodes, x_float.shape[1]))
    for i in range(num_nodes):
        agg_float[i] = torch.sum(x_float * adj_matrix[:, i:i+1], dim=0)
    agg_int8 = torch.round(agg_float / scale_hidden).clamp(-128, 127).to(torch.int8)
    return agg_int8
```
Issues:
- Requires `edge_index` (unnecessary)
- Manual loop (slower)
- Less clear what it's doing

**generate_ptq_test_vectors_clean.py** (simple):
```python
def quantized_aggregate(x_int8, adj_matrix, scale_in, scale_hidden):
    x_float = x_int8.float() * scale_in
    adj_tensor = torch.from_numpy(adj_matrix).float()
    agg_float = torch.matmul(adj_tensor, x_float)
    agg_int8 = torch.round(agg_float / scale_hidden).clamp(-128, 127).to(torch.int8)
    return agg_int8
```
Advantages:
- Direct matrix multiplication (matches HLS)
- No extra parameters needed
- Clearer implementation
- Faster execution

---

## Validation

**Question**: Is the output of `generate_ptq_test_vectors_clean.py` currently in `build/test_vectors_ptq/`?

**Check**:
```bash
cat build/test_vectors_ptq/network_output_reference.txt | head -1
```

If it shows the output from the clean script (INT8 values like -45, -4, -96, -23, -118, 124, -128), then yes, it's the correct one.
