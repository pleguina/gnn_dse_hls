#!/usr/bin/env python3
"""
Trace through SAGEConv step-by-step to understand exact aggregation behavior.
"""

import torch
import numpy as np
from torch_geometric.utils import add_self_loops, degree

def main():
    print("="*80)
    print("TRACING SAGEConv AGGREGATION STEP-BY-STEP")
    print("="*80)
    
    # Load test vectors
    test_dir = '../build/test_vectors_float'
    
    edge_index_data = np.loadtxt(f'{test_dir}/edge_index.txt', dtype=int)
    edge_index = torch.tensor(edge_index_data, dtype=torch.long).t()
    
    input_data = np.loadtxt(f'{test_dir}/network_input.txt', dtype=np.float32)
    x = torch.from_numpy(input_data).float()
    
    print(f"\nOriginal edge_index shape: {edge_index.shape}")
    print(f"Original edge_index:\n{edge_index}")
    print(f"\nNumber of edges (without self-loops): {edge_index.shape[1]}")
    
    # Check if edge_index already has self-loops
    has_self_loops = False
    for i in range(edge_index.shape[1]):
        if edge_index[0, i] == edge_index[1, i]:
            has_self_loops = True
            break
    
    print(f"Has self-loops: {has_self_loops}")
    
    # Manual mean aggregation (what SAGEConv should do with aggr='mean')
    print(f"\n" + "="*80)
    print("MANUAL MEAN AGGREGATION (following MessagePassing)")
    print("="*80)
    
    num_nodes = x.shape[0]
    num_features = x.shape[1]
    
    # Step 1: Gather source node features for each edge
    # edge_index[0] = source, edge_index[1] = target
    x_j = x[edge_index[0]]  # Features of source nodes
    
    print(f"\nSource node features (x_j) shape: {x_j.shape}")
    print(f"First 3 edges:")
    for i in range(min(3, edge_index.shape[1])):
        src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
        print(f"  Edge {i}: {src} -> {tgt}, x_j[{i},:5] = {x_j[i,:5]}")
    
    # Step 2: Aggregate using scatter (sum then divide by degree for mean)
    # This is what aggr='mean' does
    index = edge_index[1]  # Target nodes
    
    # Count incoming edges per node (degree)
    degree_count = torch.zeros(num_nodes, dtype=torch.long)
    for tgt in index:
        degree_count[tgt] += 1
    
    print(f"\nIncoming degree per node: {degree_count}")
    
    # Sum features for each target node
    out_sum = torch.zeros(num_nodes, num_features)
    for i in range(edge_index.shape[1]):
        tgt = index[i]
        out_sum[tgt] += x_j[i]
    
    print(f"\nSum of incoming features [0,:5]: {out_sum[0,:5]}")
    
    # Divide by degree to get mean
    out_mean = out_sum / degree_count.unsqueeze(1).float()
    
    print(f"Mean aggregation [0,:5]: {out_mean[0,:5]}")
    print(f"Mean aggregation range: [{out_mean.min():.4f}, {out_mean.max():.4f}]")
    
    # Compare with PyTorch SAGEConv
    print(f"\n" + "="*80)
    print("COMPARISON WITH PYTORCH SAGECONV")
    print("="*80)
    
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
    from model_base import ReducedGraphSAGE
    
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
        agg_pytorch = model.conv1.propagate(edge_index, x=(x, x), size=None)
    
    print(f"\nPyTorch SAGEConv aggregation [0,:5]: {agg_pytorch[0,:5]}")
    print(f"Manual mean aggregation [0,:5]:      {out_mean[0,:5]}")
    
    diff = torch.abs(agg_pytorch - out_mean)
    print(f"\nMax difference: {diff.max():.8f}")
    print(f"Mean difference: {diff.mean():.8f}")
    
    if diff.max() < 1e-5:
        print(f"\n✅ PERFECT MATCH! Manual aggregation matches PyTorch")
    else:
        print(f"\n⚠️  Mismatch detected")
        print(f"\nChecking if SAGEConv adds self-loops...")
        
        # Try with self-loops added
        edge_index_with_loops, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        print(f"\nEdge index with self-loops shape: {edge_index_with_loops.shape}")
        print(f"Number of edges (with self-loops): {edge_index_with_loops.shape[1]}")
        
        with torch.no_grad():
            agg_pytorch_loops = model.conv1.propagate(edge_index_with_loops, x=(x, x), size=None)
        
        print(f"\nPyTorch with explicit self-loops [0,:5]: {agg_pytorch_loops[0,:5]}")
        
        diff_loops = torch.abs(agg_pytorch_loops - out_mean)
        print(f"Diff with manual (no self-loops): {diff_loops.max():.8f}")

if __name__ == '__main__':
    main()
