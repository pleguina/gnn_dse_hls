#!/usr/bin/env python3
"""
Verify that the float reference output matches PyTorch model execution.
"""

import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_base import ReducedGraphSAGE

def load_float_matrix(filename):
    """Load matrix from text file."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    matrix = []
    for line in lines:
        row = [float(x) for x in line.strip().split()]
        matrix.append(row)
    
    return np.array(matrix, dtype=np.float32)


def main():
    print("="*70)
    print("VERIFY FLOAT REFERENCE OUTPUT")
    print("="*70)
    
    # Load model
    model_path = '../build/models/reduced_graphsage_no_root_best.pth'
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Get model config
    in_channels = checkpoint.get('in_channels', 1433)
    in_channels_reduced = checkpoint.get('in_channels_reduced', 16)
    hidden_channels = checkpoint.get('hidden_channels', 24)
    out_channels = checkpoint.get('out_channels', 7)
    use_projection = checkpoint.get('use_projection', True)
    root_weight = checkpoint.get('root_weight', False)
    
    print(f"\nModel configuration:")
    print(f"  in_channels: {in_channels}")
    print(f"  in_channels_reduced: {in_channels_reduced}")
    print(f"  hidden_channels: {hidden_channels}")
    print(f"  out_channels: {out_channels}")
    print(f"  use_projection: {use_projection}")
    print(f"  root_weight: {root_weight}")
    
    # Create model
    model = ReducedGraphSAGE(
        in_channels=in_channels,
        in_channels_reduced=in_channels_reduced,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        use_projection=use_projection,
        root_weight=root_weight
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test vectors
    test_dir = '../build/test_vectors_float'
    
    print(f"\nLoading test vectors from: {test_dir}")
    
    # Load input features
    input_features = load_float_matrix(f'{test_dir}/network_input.txt')
    print(f"  Input features shape: {input_features.shape}")
    
    # The test vectors are already projected! So we need to skip projection
    # and feed directly into conv layers
    actual_input_dim = input_features.shape[1]
    
    if actual_input_dim == in_channels_reduced and use_projection:
        print(f"\n  Note: Input is already projected to {in_channels_reduced} dimensions")
        print(f"        Will disable projection for testing")
        use_projection_for_test = False
    else:
        use_projection_for_test = use_projection
    
    # Load edge index
    edge_index_data = load_float_matrix(f'{test_dir}/edge_index.txt')
    edge_index = torch.tensor(edge_index_data, dtype=torch.long).t()
    print(f"  Edge index shape: {edge_index.shape}")
    print(f"  Number of edges: {edge_index.shape[1]}")
    
    # Load reference output
    reference_output = load_float_matrix(f'{test_dir}/network_output_reference.txt')
    print(f"  Reference output shape: {reference_output.shape}")
    
    # Run model
    print(f"\nRunning PyTorch model...")
    x = torch.from_numpy(input_features).float()
    
    with torch.no_grad():
        # Skip projection if input is already projected
        if use_projection_for_test and model.use_projection:
            x = model.projection(x)
        
        # Run through conv layers
        x = model.conv1(x, edge_index)
        x = torch.relu(x)
        output = model.conv2(x, edge_index)
    
    output_np = output.numpy()
    
    # Compare
    print(f"\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    print(f"\nPyTorch output [0,:]:  {output_np[0,:]}")
    print(f"Reference output [0,:]: {reference_output[0,:]}")
    
    diff = np.abs(output_np - reference_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"\nMax absolute difference: {max_diff:.8f}")
    print(f"Mean absolute difference: {mean_diff:.8f}")
    
    # Check if they match within floating point tolerance
    if max_diff < 1e-5:
        print(f"\n✅ PERFECT MATCH! Reference is correct.")
        return 0
    elif max_diff < 1e-3:
        print(f"\n✓ Good match (within 0.001)")
        return 0
    else:
        print(f"\n✗ MISMATCH! Reference does not match PyTorch output.")
        
        # Show first few mismatches
        print(f"\nFirst mismatches:")
        count = 0
        for i in range(output_np.shape[0]):
            for j in range(output_np.shape[1]):
                if diff[i,j] > 1e-5:
                    print(f"  [{i},{j}]: PyTorch={output_np[i,j]:.6f}, Ref={reference_output[i,j]:.6f}, Diff={diff[i,j]:.6f}")
                    count += 1
                    if count >= 10:
                        break
            if count >= 10:
                break
        
        return 1


if __name__ == '__main__':
    sys.exit(main())
