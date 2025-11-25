#!/usr/bin/env python3
"""
Test to understand the difference between:
1. PyTorch SAGEConv aggregation (mean via MessagePassing)
2. GCN symmetric normalization (what we have in adj_matrix)
3. Simple row normalization (what we should use)
"""

import torch
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_base import ReducedGraphSAGE

def main():
    print("="*80)
    print("TESTING DIFFERENT AGGREGATION METHODS")
    print("="*80)
    
    # Load test vectors
    test_dir = '../build/test_vectors_float'
    
    # Load edge index
    edge_index_data = np.loadtxt(f'{test_dir}/edge_index.txt', dtype=int)
    edge_index = torch.tensor(edge_index_data, dtype=torch.long).t()
    
    # Load input features
    input_data = np.loadtxt(f'{test_dir}/network_input.txt', dtype=np.float32)
    x = torch.from_numpy(input_data).float()
    
    # Load the pre-computed adjacency matrix (GCN normalized)
    adj_gcn = np.loadtxt(f'{test_dir}/adj_matrix.txt', dtype=np.float32)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Number of edges: {edge_index.shape[1]}")
    print(f"Adjacency matrix (GCN norm) shape: {adj_gcn.shape}")
    
    # Method 1: PyTorch SAGEConv (the CORRECT one)
    print(f"\n" + "="*80)
    print("METHOD 1: PyTorch SAGEConv (using edge_index directly)")
    print("="*80)
    
    model_path = '../build/models/reduced_graphsage_no_root_best.pth'
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = ReducedGraphSAGE(
        in_channels=checkpoint.get('in_channels', 1433),
        in_channels_reduced=checkpoint.get('in_channels_reduced', 16),
        hidden_channels=checkpoint.get('hidden_channels', 24),
        out_channels=checkpoint.get('out_channels', 7),
        use_projection=checkpoint.get('use_projection', True),
        root_weight=checkpoint.get('root_weight', False)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    with torch.no_grad():
        # Just first layer to see aggregation
        agg1_pytorch = model.conv1.propagate(edge_index, x=(x, x), size=None)
    
    print(f"PyTorch aggregation [0,:5]: {agg1_pytorch[0,:5]}")
    print(f"PyTorch aggregation range: [{agg1_pytorch.min():.4f}, {agg1_pytorch.max():.4f}]")
    
    # Method 2: GCN symmetric normalization (what's in the test vectors)
    print(f"\n" + "="*80)
    print("METHOD 2: GCN Symmetric Normalization (adj_matrix @ features)")
    print("="*80)
    
    agg1_gcn = torch.from_numpy(adj_gcn @ input_data)
    
    print(f"GCN aggregation [0,:5]: {agg1_gcn[0,:5]}")
    print(f"GCN aggregation range: [{agg1_gcn.min():.4f}, {agg1_gcn.max():.4f}]")
    
    # Method 3: Build raw adjacency with self-loops and row-normalize
    print(f"\n" + "="*80)
    print("METHOD 3: Raw adjacency + self-loops + row normalization")
    print("="*80)
    
    num_nodes = x.shape[0]
    
    # Build raw adjacency matrix
    adj_raw = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        tgt = edge_index[1, i].item()
        adj_raw[tgt, src] = 1.0  # Note: PyG uses edge_index[0]=source, edge_index[1]=target
    
    # Add self-loops
    adj_raw = adj_raw + np.eye(num_nodes, dtype=np.float32)
    
    print(f"\nRaw adjacency (with self-loops):")
    print(adj_raw)
    
    # Row-normalize (divide each row by its sum)
    row_sums = adj_raw.sum(axis=1, keepdims=True)
    adj_row_norm = adj_raw / row_sums
    
    print(f"\nRow-normalized adjacency:")
    print(adj_row_norm)
    print(f"Row sums (should all be 1.0): {adj_row_norm.sum(axis=1)}")
    
    agg1_row_norm = adj_row_norm @ input_data
    agg1_row_norm_tensor = torch.from_numpy(agg1_row_norm)
    
    print(f"\nRow-norm aggregation [0,:5]: {agg1_row_norm_tensor[0,:5]}")
    print(f"Row-norm aggregation range: [{agg1_row_norm_tensor.min():.4f}, {agg1_row_norm_tensor.max():.4f}]")
    
    # Comparison
    print(f"\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    diff_pytorch_gcn = torch.abs(agg1_pytorch - agg1_gcn)
    diff_pytorch_rownorm = torch.abs(agg1_pytorch - agg1_row_norm_tensor)
    
    print(f"\nPyTorch vs GCN (current test vectors):")
    print(f"  Max diff: {diff_pytorch_gcn.max():.6f}")
    print(f"  Mean diff: {diff_pytorch_gcn.mean():.6f}")
    
    print(f"\nPyTorch vs Row-normalized:")
    print(f"  Max diff: {diff_pytorch_rownorm.max():.6f}")
    print(f"  Mean diff: {diff_pytorch_rownorm.mean():.6f}")
    
    if diff_pytorch_rownorm.max() < 1e-5:
        print(f"\n✅ Row-normalized adjacency MATCHES PyTorch!")
        print(f"   This is what we should use for HLS")
    elif diff_pytorch_gcn.max() < 1e-5:
        print(f"\n✅ GCN symmetric normalization MATCHES PyTorch!")
        print(f"   Current adjacency matrix is correct")
    else:
        print(f"\n⚠️  Neither method matches perfectly")
        print(f"   Need to investigate further")
    
    # Save the correct adjacency matrix
    print(f"\n" + "="*80)
    print("SAVING CORRECTED ADJACENCY MATRIX")
    print("="*80)
    
    output_file = f'{test_dir}/adj_matrix_row_normalized.txt'
    np.savetxt(output_file, adj_row_norm, fmt='%.8f')
    print(f"Saved row-normalized adjacency to: {output_file}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
