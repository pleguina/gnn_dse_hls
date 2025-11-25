"""
Generate FLOAT test vectors for HLS testbench validation.
Creates non-quantized (float) inputs, weights, and reference outputs
for the reduced model float HLS implementation.
"""

import torch
import numpy as np
import os
import sys

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_base import ReducedGraphSAGE
from subgraph_extraction import extract_fixed_subgraph
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


def save_float_matrix(matrix, filename):
    """Save matrix to text file in float format."""
    np.savetxt(filename, matrix, fmt='%.8f')
    print(f"Saved {filename}")


def save_float_vector(vector, filename):
    """Save 1D vector to text file in float format (one value per line)."""
    np.savetxt(filename, vector.reshape(-1, 1), fmt='%.8f')
    print(f"Saved {filename}")


def generate_float_test_vectors(model, subgraph_data, output_dir):
    """
    Generate FLOAT test vectors for full two-layer network.
    
    This generates test vectors WITHOUT quantization for validating
    the float HLS implementation.

    Args:
        model: Trained reduced model
        subgraph_data: Subgraph data dictionary
        output_dir: Output directory for test vectors
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get subgraph adjacency and features
    adj_matrix = subgraph_data['adj_matrix']
    features = subgraph_data['x']

    # Process through projection layer if it exists
    if hasattr(model, 'projection') and model.use_projection:
        features_tensor = torch.from_numpy(features).float()
        with torch.no_grad():
            features_tensor = model.projection(features_tensor)
            features_tensor = torch.relu(features_tensor)
        features = features_tensor.numpy()

    # Save adjacency matrix (float, row-normalized)
    save_float_matrix(adj_matrix, f'{output_dir}/adj_matrix.txt')
    
    # Save edge_index for exact reproduction
    edge_index_np = subgraph_data['edge_index']
    np.savetxt(f'{output_dir}/edge_index.txt', edge_index_np.T, fmt='%d')
    print(f"Saved {output_dir}/edge_index.txt")

    # Save input features (FLOAT, after projection)
    save_float_matrix(features, f'{output_dir}/network_input.txt')

    # Layer 1: Get weights and bias (FLOAT)
    conv1 = model.conv1
    weights1 = conv1.lin_l.weight.data.numpy()  # [hidden, in_reduced]
    bias1 = conv1.lin_l.bias.data.numpy() if conv1.lin_l.bias is not None else np.zeros(weights1.shape[0])

    save_float_matrix(weights1, f'{output_dir}/weights_layer1.txt')
    save_float_vector(bias1, f'{output_dir}/bias_layer1.txt')

    # Layer 2: Get weights and bias (FLOAT)
    conv2 = model.conv2
    weights2 = conv2.lin_l.weight.data.numpy()  # [out, hidden]
    bias2 = conv2.lin_l.bias.data.numpy() if conv2.lin_l.bias is not None else np.zeros(weights2.shape[0])

    save_float_matrix(weights2, f'{output_dir}/weights_layer2.txt')
    save_float_vector(bias2, f'{output_dir}/bias_layer2.txt')

    # Generate reference output using PyTorch (FLOAT)
    model.eval()
    with torch.no_grad():
        # Create edge_index from subgraph
        edge_index = torch.from_numpy(subgraph_data['edge_index']).long()

        # Run through full network (without projection, already applied)
        x = torch.from_numpy(features).float()
        out = model.conv1(x, edge_index)
        out = torch.relu(out)
        out = model.conv2(out, edge_index)

        # Save output as FLOAT (no quantization)
        out_np = out.numpy()

    # Save reference output (FLOAT)
    save_float_matrix(out_np, f'{output_dir}/network_output_reference.txt')

    # Save info file
    with open(f'{output_dir}/model_info.txt', 'w') as f:
        f.write(f"# Float Test Vectors for Reduced Model\n")
        f.write(f"num_nodes: {features.shape[0]}\n")
        f.write(f"in_features: {features.shape[1]}\n")
        f.write(f"hidden_features: {weights1.shape[0]}\n")
        f.write(f"out_features: {weights2.shape[0]}\n")
        f.write(f"root_weight: False\n")
        f.write(f"data_format: FLOAT32\n")

    print(f"\nFloat test vectors for network generated in {output_dir}/")
    print(f"  Input shape: {features.shape} (FLOAT)")
    print(f"  Layer 1 weights shape: {weights1.shape} (FLOAT)")
    print(f"  Layer 2 weights shape: {weights2.shape} (FLOAT)")
    print(f"  Output shape: {out_np.shape} (FLOAT)")


if __name__ == '__main__':
    print("="*60)
    print("Generating FLOAT Test Vectors for HLS Validation")
    print("(Reduced Model - No Quantization)")
    print("="*60)

    # Load Cora dataset
    print("\nLoading Cora dataset...")
    dataset = Planetoid(root='../data', name='Cora', transform=NormalizeFeatures())
    data = dataset[0]

    # Extract subgraph with fixed center node for reproducibility
    print("Extracting subgraph...")
    subgraph_data = extract_fixed_subgraph(data, num_nodes=8, center_node=0, num_hops=2)

    # Load trained reduced model
    print("Loading trained reduced model...")
    model = ReducedGraphSAGE(
        in_channels=dataset.num_features,
        in_channels_reduced=16,
        hidden_channels=24,
        out_channels=dataset.num_classes,
        dropout=0.5,
        use_projection=True,
        root_weight=False  # HLS-compatible version
    )

    try:
        checkpoint = torch.load('../build/models/reduced_graphsage_no_root_best.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Loaded trained model from ../build/models/reduced_graphsage_no_root_best.pth")
        print("  Using HLS-compatible model (no root_weight)")
    except Exception as e:
        print(f"Warning: Could not load trained model: {e}")
        print("Using random weights for test vector generation")

    # Generate FLOAT test vectors
    output_dir = '../build/test_vectors_float'

    print("\n" + "="*60)
    print("Generating FLOAT test vectors for full network...")
    print("="*60)
    generate_float_test_vectors(model, subgraph_data, output_dir)

    print("\n" + "="*60)
    print("FLOAT test vector generation complete!")
    print("="*60)
    print(f"\nTest vectors saved to: {output_dir}/")
    print("\nThese are PURE FLOAT vectors (no quantization)")
    print("Use these for validating the float HLS implementation")
    print("\nYou can now run the HLS testbench with these vectors.")
