# SAGEConv Adjacency Matrix Normalization Fix

## Problem Summary

The HLS float implementation was showing large mismatches (max diff ~4.24) compared to the PyTorch reference output, despite using identical weights and input features. All 56 outputs had >1% relative error.

## Root Cause Analysis

### The Issue

The adjacency matrix in the test vectors was using **GCN-style symmetric normalization**:

```python
# Old code in subgraph_extraction.py
adj_matrix = adj_matrix + torch.eye(num_nodes_actual)  # Add self-loops
deg = adj_matrix.sum(dim=1)
deg_inv_sqrt = deg.pow(-0.5)
adj_matrix = deg_inv_sqrt.view(-1, 1) * adj_matrix * deg_inv_sqrt.view(1, -1)
```

This creates a **symmetrically normalized** adjacency matrix: **D^{-1/2} A D^{-1/2}**

### Why This Was Wrong

PyTorch Geometric's `SAGEConv` with `aggr='mean'` does **NOT** use symmetric normalization. Instead, it:

1. **Does NOT add self-loops automatically** (unlike GCN layers)
2. Uses **simple mean aggregation**: For each target node i, it sums features from source nodes j and divides by the number of incoming edges

The `MessagePassing.aggregate()` function with `aggr='mean'` performs:
```python
# Pseudocode for what SAGEConv does internally
for each edge (source -> target):
    accumulate source_features to target_node
    
for each target_node:
    aggregated[target_node] = accumulated_features[target_node] / incoming_degree[target_node]
```

This is equivalent to **row normalization** of the adjacency matrix: each row divided by its sum.

### The Mismatch

- **Test vectors had**: Symmetric normalization (D^{-1/2} A D^{-1/2}) + self-loops
- **PyTorch SAGEConv uses**: Row normalization (D^{-1} A) without self-loops
- **HLS implementation**: Matrix multiplication with the test vector adjacency (wrong normalization)

This caused the HLS to compute:
```
HLS_output = (D^{-1/2} A D^{-1/2}) @ features
```

When it should compute:
```
Correct_output = (D^{-1} A) @ features
```

## The Fix

Modified `src/subgraph_extraction.py` to use **row normalization** matching SAGEConv behavior:

```python
# New code - SAGEConv-compatible normalization
adj_matrix = torch.zeros((num_nodes_actual, num_nodes_actual), dtype=torch.float32)
adj_matrix[edge_index[1], edge_index[0]] = 1.0

# Normalize adjacency matrix for SAGEConv mean aggregation
# SAGEConv with aggr='mean' does NOT add self-loops automatically
# It simply divides by the number of incoming neighbors
# So we row-normalize: adj[i,j] = 1/degree(i) if there's an edge j->i
deg = adj_matrix.sum(dim=1)  # Incoming degree per node
deg_inv = 1.0 / deg
deg_inv[deg_inv == float('inf')] = 0  # Handle isolated nodes
adj_matrix = deg_inv.view(-1, 1) * adj_matrix  # Row-normalize
```

### Key Changes

1. **Removed self-loop addition**: No `+ torch.eye(num_nodes_actual)`
2. **Changed to row normalization**: `deg^{-1}` instead of `deg^{-1/2}`
3. **One-sided normalization**: Multiply only by `deg_inv` on the left (row-wise), not on both sides

## Verification

### Before Fix
```
Max absolute difference: 4.2425
Errors (>1.0000% relative): 56 / 56 (100%)
HLS Output [0,:]: -0.7554 -1.4875 -2.6545  5.7932  1.7481 -1.3037 -5.3825
Reference  [0,:]: -0.1809 -2.1708 -2.6095  6.2785  1.2966 -0.4822 -6.1783
```

### After Fix
```
Max absolute difference: 0.0000
Errors (>1.0000% relative): 0 / 56 (0%)
HLS Output [0,:]: -0.1809 -2.1708 -2.6095  6.2785  1.2966 -0.4822 -6.1783
Reference  [0,:]: -0.1809 -2.1708 -2.6095  6.2785  1.2966 -0.4822 -6.1783
```

✅ **Perfect match** with PyTorch reference!

## Technical Details

### Adjacency Matrix Properties

**Old (GCN-style) adjacency matrix:**
- Had self-loops (diagonal = non-zero)
- Symmetric normalization: values in [0, ~0.5]
- Row sums ≠ 1.0
- Used for GCN layers, not SAGEConv

**New (SAGEConv-style) adjacency matrix:**
- No self-loops (diagonal = 0)
- Row normalization: each row sums to 1.0
- Values = 1/degree for existing edges, 0 otherwise
- Equivalent to `aggr='mean'` in PyTorch Geometric

### Example

For node 0 with 3 incoming edges from nodes [1, 5, 7]:

**Old adjacency (row 0):**
```
[0.25677717, 0.25677717, 0.0, 0.0, 0.0, 0.22966849, 0.0, 0.25677717]
```
- Includes self-loop (position 0)
- Symmetric normalization
- Sum ≈ 0.973 (not 1.0)

**New adjacency (row 0):**
```
[0.0, 0.33333334, 0.0, 0.0, 0.0, 0.33333334, 0.0, 0.33333334]
```
- No self-loop (position 0 = 0)
- Row normalized: 1/3 for each of the 3 neighbors
- Sum = 1.0

### SAGEConv Architecture

From the PyTorch Geometric paper, SAGEConv computes:

**Without root_weight (our case):**
```
h_i = W_l * mean(h_j for j in N(i)) + b_l
```

**With root_weight:**
```
h_i = W_l * mean(h_j for j in N(i)) + W_r * h_i + b_l
```

Where:
- `N(i)` = neighbors of node i (NOT including i itself unless explicitly in edges)
- `mean()` = simple average (sum / count)
- No self-loops added automatically

## Impact on Other Components

### Quantized Test Vectors

The same fix needs to be applied to quantized test vector generation:
- `tests/generate_test_vectors.py` (PTQ quantization)
- `src/quantization_qat.py` (QAT quantization)

Both should use the corrected `subgraph_extraction.py` automatically since they import and call `extract_fixed_subgraph()`.

### Recommendation

Regenerate ALL test vectors (FLOAT, PTQ, QAT) to ensure consistency:
```bash
python run_pipeline.py --steps vectors
```

## Lessons Learned

1. **Layer-specific normalization**: Different GNN layers use different aggregation schemes
   - GCN: Symmetric normalization with self-loops
   - SAGEConv: Row normalization without automatic self-loops
   - GAT: Attention-based (no fixed adjacency matrix)

2. **Test vector validation**: Always verify test vectors match the actual PyTorch execution, not assumptions about the algorithm

3. **MessagePassing framework**: Understanding the `propagate()` mechanism is crucial for implementing GNN layers correctly

4. **Documentation matters**: The comment "Adjacency matrix is already row-normalized for mean aggregation" was incorrect - it was using symmetric normalization

## References

- PyTorch Geometric SAGEConv: `torch_geometric.nn.SAGEConv`
- MessagePassing base class: `torch_geometric.nn.MessagePassing`
- Original GraphSAGE paper: "Inductive Representation Learning on Large Graphs" (Hamilton et al., 2017)
- Our model config: `root_weight=False`, `aggr='mean'`, `normalize=False`, `project=False`
