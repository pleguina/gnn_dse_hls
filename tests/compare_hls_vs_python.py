"""
Debug script to compare Python and HLS implementations step by step.
This helps identify where the mismatch occurs.
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_base import ReducedGraphSAGE
from subgraph_extraction import extract_fixed_subgraph
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


def load_test_vectors(test_dir):
    """Load test vectors from directory."""
    data = {}
    
    # Load adjacency matrix
    data['adj_matrix'] = np.loadtxt(f'{test_dir}/adj_matrix.txt')
    
    # Load inputs
    data['input'] = np.loadtxt(f'{test_dir}/network_input.txt')
    
    # Load weights and biases
    data['weights1'] = np.loadtxt(f'{test_dir}/weights_layer1.txt')
    data['bias1'] = np.loadtxt(f'{test_dir}/bias_layer1.txt')
    data['weights2'] = np.loadtxt(f'{test_dir}/weights_layer2.txt')
    data['bias2'] = np.loadtxt(f'{test_dir}/bias_layer2.txt')
    
    # Load reference output
    data['output_ref'] = np.loadtxt(f'{test_dir}/network_output_reference.txt')
    
    return data


def manual_aggregation(adj_matrix, features):
    """Manual implementation of mean aggregation."""
    # agg[i,f] = sum_j (adj[i,j] * features[j,f])
    return adj_matrix @ features


def manual_linear(features, weights, bias):
    """Manual implementation of linear transformation."""
    # output = features @ weights.T + bias
    return features @ weights.T + bias


def manual_relu(x):
    """Manual ReLU implementation."""
    return np.maximum(0, x)


def run_manual_forward(data):
    """Run forward pass manually following HLS implementation."""
    print("\n" + "="*70)
    print("MANUAL FORWARD PASS (matching HLS logic)")
    print("="*70)
    
    adj_matrix = data['adj_matrix']
    input_features = data['input']
    weights1 = data['weights1']
    bias1 = data['bias1']
    weights2 = data['weights2']
    bias2 = data['bias2']
    
    print(f"\nInput shapes:")
    print(f"  adj_matrix: {adj_matrix.shape}")
    print(f"  input: {input_features.shape}")
    print(f"  weights1: {weights1.shape}")
    print(f"  bias1: {bias1.shape}")
    print(f"  weights2: {weights2.shape}")
    print(f"  bias2: {bias2.shape}")
    
    # Layer 1: Aggregate
    print(f"\n--- Layer 1: Aggregation ---")
    agg1 = manual_aggregation(adj_matrix, input_features)
    print(f"agg1 shape: {agg1.shape}")
    print(f"agg1[0,:5]: {agg1[0,:5]}")
    
    # Layer 1: Linear + ReLU
    print(f"\n--- Layer 1: Linear Transform ---")
    hidden = manual_linear(agg1, weights1, bias1)
    print(f"hidden (before ReLU) shape: {hidden.shape}")
    print(f"hidden[0,:5]: {hidden[0,:5]}")
    
    hidden = manual_relu(hidden)
    print(f"hidden (after ReLU) shape: {hidden.shape}")
    print(f"hidden[0,:5]: {hidden[0,:5]}")
    
    # Layer 2: Aggregate
    print(f"\n--- Layer 2: Aggregation ---")
    agg2 = manual_aggregation(adj_matrix, hidden)
    print(f"agg2 shape: {agg2.shape}")
    print(f"agg2[0,:5]: {agg2[0,:5]}")
    
    # Layer 2: Linear (no ReLU)
    print(f"\n--- Layer 2: Linear Transform ---")
    output = manual_linear(agg2, weights2, bias2)
    print(f"output shape: {output.shape}")
    print(f"output[0,:]: {output[0,:]}")
    
    return output


def run_pytorch_forward(model, data):
    """Run forward pass using PyTorch model."""
    print("\n" + "="*70)
    print("PYTORCH FORWARD PASS")
    print("="*70)
    
    # Create edge_index from adjacency matrix
    adj_matrix = data['adj_matrix']
    edge_index = []
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i, j] != 0:
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    
    print(f"\nEdge index shape: {edge_index.shape}")
    print(f"Number of edges: {edge_index.shape[1]}")
    
    # Run through model
    x = torch.from_numpy(data['input']).float()
    
    model.eval()
    with torch.no_grad():
        # First layer
        print(f"\n--- Layer 1 ---")
        out1 = model.conv1(x, edge_index)
        print(f"After conv1 (before ReLU): {out1[0,:5]}")
        out1 = torch.relu(out1)
        print(f"After ReLU: {out1[0,:5]}")
        
        # Second layer
        print(f"\n--- Layer 2 ---")
        out2 = model.conv2(out1, edge_index)
        print(f"After conv2: {out2[0,:]}")
        
        output = out2.numpy()
    
    return output


def compare_outputs(manual_output, pytorch_output, reference_output):
    """Compare all three outputs."""
    print("\n" + "="*70)
    print("OUTPUT COMPARISON")
    print("="*70)
    
    print(f"\nManual output [0,:]:   {manual_output[0,:]}")
    print(f"PyTorch output [0,:]:  {pytorch_output[0,:]}")
    print(f"Reference output [0,:]: {reference_output[0,:]}")
    
    # Compare Manual vs PyTorch
    diff_manual_pytorch = np.abs(manual_output - pytorch_output)
    max_diff_mp = np.max(diff_manual_pytorch)
    print(f"\n--- Manual vs PyTorch ---")
    print(f"Max absolute difference: {max_diff_mp}")
    print(f"Mean absolute difference: {np.mean(diff_manual_pytorch)}")
    
    # Compare Manual vs Reference
    diff_manual_ref = np.abs(manual_output - reference_output)
    max_diff_mr = np.max(diff_manual_ref)
    print(f"\n--- Manual vs Reference ---")
    print(f"Max absolute difference: {max_diff_mr}")
    print(f"Mean absolute difference: {np.mean(diff_manual_ref)}")
    
    # Compare PyTorch vs Reference
    diff_pytorch_ref = np.abs(pytorch_output - reference_output)
    max_diff_pr = np.max(diff_pytorch_ref)
    print(f"\n--- PyTorch vs Reference ---")
    print(f"Max absolute difference: {max_diff_pr}")
    print(f"Mean absolute difference: {np.mean(diff_pytorch_ref)}")
    
    # Check if they match
    tolerance = 1e-5
    if max_diff_mp < tolerance:
        print(f"\n✓ Manual and PyTorch match! (diff < {tolerance})")
    else:
        print(f"\n✗ Manual and PyTorch DIFFER! (max diff = {max_diff_mp})")
    
    if max_diff_pr < tolerance:
        print(f"✓ PyTorch and Reference match! (diff < {tolerance})")
    else:
        print(f"✗ PyTorch and Reference DIFFER! (max diff = {max_diff_pr})")


def main():
    print("="*70)
    print("DEBUG: HLS vs Python Comparison")
    print("="*70)
    
    # Load test vectors
    test_dir = '../build/test_vectors_float'
    print(f"\nLoading test vectors from: {test_dir}")
    data = load_test_vectors(test_dir)
    
    # Load PyTorch model
    print(f"\nLoading PyTorch model...")
    dataset = Planetoid(root='../data', name='Cora', transform=NormalizeFeatures())
    
    model = ReducedGraphSAGE(
        in_channels=dataset.num_features,
        in_channels_reduced=16,
        hidden_channels=24,
        out_channels=dataset.num_classes,
        dropout=0.5,
        use_projection=True,
        root_weight=False
    )
    
    checkpoint = torch.load('../build/models/reduced_graphsage_no_root_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Verify weights match
    print(f"\n--- Verifying weights match test vectors ---")
    w1_model = model.conv1.lin_l.weight.data.numpy()
    w1_test = data['weights1']
    print(f"Weights1 match: {np.allclose(w1_model, w1_test)}")
    print(f"  Max diff: {np.max(np.abs(w1_model - w1_test))}")
    
    b1_model = model.conv1.lin_l.bias.data.numpy()
    b1_test = data['bias1']
    print(f"Bias1 match: {np.allclose(b1_model, b1_test)}")
    print(f"  Max diff: {np.max(np.abs(b1_model - b1_test))}")
    
    # Run manual forward pass
    manual_output = run_manual_forward(data)
    
    # Run PyTorch forward pass
    pytorch_output = run_pytorch_forward(model, data)
    
    # Compare outputs
    compare_outputs(manual_output, pytorch_output, data['output_ref'])
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("If Manual matches PyTorch but differs from Reference:")
    print("  → Problem is in test vector generation")
    print("\nIf Manual matches Reference but differs from PyTorch:")
    print("  → Problem is in PyTorch model implementation")
    print("\nIf all three differ:")
    print("  → Problem is in aggregation or linear transform logic")
    print("="*70)


if __name__ == '__main__':
    main()
