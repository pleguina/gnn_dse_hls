"""
Generate test vectors for HLS testbench validation.
Creates quantized inputs, weights, and reference outputs.
"""

import torch
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_base import ReducedGraphSAGE
from quantization import quantize_tensor
from subgraph_extraction import extract_fixed_subgraph
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


def save_matrix(matrix, filename):
    """Save matrix to text file."""
    np.savetxt(filename, matrix, fmt='%.6f')
    print(f"Saved {filename}")


def save_int_matrix(matrix, filename):
    """Save integer matrix to text file."""
    np.savetxt(filename, matrix, fmt='%d')
    print(f"Saved {filename}")


def generate_test_vectors_for_layer(model, subgraph_data, output_dir):
    """
    Generate test vectors for a single GraphSAGE layer.

    Args:
        model: Trained model
        subgraph_data: Subgraph data dictionary
        output_dir: Output directory for test vectors
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get subgraph adjacency and features
    adj_matrix = subgraph_data['adj_matrix']
    features = subgraph_data['x']

    # Quantize input features
    features_tensor = torch.from_numpy(features).float()
    features_quant, scale_in, _ = quantize_tensor(features_tensor, num_bits=8)
    features_quant_np = features_quant.numpy()

    # Save adjacency matrix (float, already normalized)
    save_matrix(adj_matrix, f'{output_dir}/adj_matrix.txt')

    # Save quantized input features
    save_int_matrix(features_quant_np, f'{output_dir}/input_features.txt')

    # Get first layer weights and bias
    conv1 = model.conv1
    weights = conv1.lin_l.weight.data  # [out_features, in_features]
    bias = conv1.lin_l.bias.data if conv1.lin_l.bias is not None else torch.zeros(weights.shape[0])

    # Quantize weights
    weights_quant, scale_weight, _ = quantize_tensor(weights, num_bits=8)
    weights_quant_np = weights_quant.numpy()

    # Quantize bias (stored as int32)
    bias_quant = (bias / (scale_in * scale_weight)).to(torch.int32)
    bias_quant_np = bias_quant.numpy()

    # Save weights and bias
    save_int_matrix(weights_quant_np, f'{output_dir}/weights.txt')
    save_int_matrix(bias_quant_np, f'{output_dir}/bias.txt')

    # Generate reference output using PyTorch
    model.eval()
    with torch.no_grad():
        # Create edge_index from subgraph
        edge_index = torch.from_numpy(subgraph_data['edge_index']).long()

        # Run through first layer only
        x = torch.from_numpy(features).float()
        out = model.conv1(x, edge_index)
        out = torch.relu(out)

        # Quantize output
        out_quant, scale_out, _ = quantize_tensor(out, num_bits=8)
        out_quant_np = out_quant.numpy()

    # Save reference output
    save_int_matrix(out_quant_np, f'{output_dir}/output_reference.txt')

    # Save scale factors
    with open(f'{output_dir}/scales.txt', 'w') as f:
        f.write(f"scale_in: {scale_in}\n")
        f.write(f"scale_weight: {scale_weight}\n")
        f.write(f"scale_out: {scale_out}\n")

    print(f"\nTest vectors for single layer generated in {output_dir}/")
    print(f"  Input shape: {features_quant_np.shape}")
    print(f"  Weights shape: {weights_quant_np.shape}")
    print(f"  Output shape: {out_quant_np.shape}")


def generate_test_vectors_for_network(model, subgraph_data, output_dir):
    """
    Generate test vectors for full two-layer network.

    Args:
        model: Trained model
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

    # Quantize input features
    features_tensor = torch.from_numpy(features).float()
    features_quant, scale_in, _ = quantize_tensor(features_tensor, num_bits=8)
    features_quant_np = features_quant.numpy()

    # Save adjacency matrix
    save_matrix(adj_matrix, f'{output_dir}/adj_matrix.txt')

    # Save quantized input features
    save_int_matrix(features_quant_np, f'{output_dir}/network_input.txt')

    # Layer 1: Get weights and bias
    conv1 = model.conv1
    weights1 = conv1.lin_l.weight.data
    bias1 = conv1.lin_l.bias.data if conv1.lin_l.bias is not None else torch.zeros(weights1.shape[0])

    # Quantize layer 1
    weights1_quant, scale_w1, _ = quantize_tensor(weights1, num_bits=8)
    bias1_quant = (bias1 / (scale_in * scale_w1)).to(torch.int32)

    save_int_matrix(weights1_quant.numpy(), f'{output_dir}/weights_layer1.txt')
    save_int_matrix(bias1_quant.numpy(), f'{output_dir}/bias_layer1.txt')

    # Layer 2: Get weights and bias
    conv2 = model.conv2
    weights2 = conv2.lin_l.weight.data
    bias2 = conv2.lin_l.bias.data if conv2.lin_l.bias is not None else torch.zeros(weights2.shape[0])

    # Quantize layer 2
    weights2_quant, scale_w2, _ = quantize_tensor(weights2, num_bits=8)
    scale_hidden = 0.1  # Approximate scale for hidden layer
    bias2_quant = (bias2 / (scale_hidden * scale_w2)).to(torch.int32)

    save_int_matrix(weights2_quant.numpy(), f'{output_dir}/weights_layer2.txt')
    save_int_matrix(bias2_quant.numpy(), f'{output_dir}/bias_layer2.txt')

    # Generate reference output using PyTorch
    model.eval()
    with torch.no_grad():
        # Create edge_index from subgraph
        edge_index = torch.from_numpy(subgraph_data['edge_index']).long()

        # Run through full network (without projection, already applied)
        x = features_tensor
        out = model.conv1(x, edge_index)
        out = torch.relu(out)
        out = model.conv2(out, edge_index)

        # Quantize output
        out_quant, scale_out, _ = quantize_tensor(out, num_bits=8)
        out_quant_np = out_quant.numpy()

    # Save reference output
    save_int_matrix(out_quant_np, f'{output_dir}/network_output_reference.txt')

    # Save scale factors
    with open(f'{output_dir}/scales.txt', 'w') as f:
        f.write(f"scale_in: {scale_in}\n")
        f.write(f"scale_w1: {scale_w1}\n")
        f.write(f"scale_w2: {scale_w2}\n")
        f.write(f"scale_hidden: {scale_hidden}\n")
        f.write(f"scale_out: {scale_out}\n")

    print(f"\nTest vectors for network generated in {output_dir}/")
    print(f"  Input shape: {features_quant_np.shape}")
    print(f"  Layer 1 weights shape: {weights1_quant.shape}")
    print(f"  Layer 2 weights shape: {weights2_quant.shape}")
    print(f"  Output shape: {out_quant_np.shape}")


if __name__ == '__main__':
    print("="*60)
    print("Generating Test Vectors for HLS Validation")
    print("="*60)

    # Load Cora dataset
    print("\nLoading Cora dataset...")
    dataset = Planetoid(root='./data', name='Cora', transform=NormalizeFeatures())
    data = dataset[0]

    # Extract subgraph
    print("Extracting subgraph...")
    subgraph_data = extract_fixed_subgraph(data, num_nodes=32, num_hops=2)

    # Load trained reduced model
    print("Loading trained model...")
    model = ReducedGraphSAGE(
        in_channels=dataset.num_features,
        in_channels_reduced=16,
        hidden_channels=24,
        out_channels=dataset.num_classes,
        dropout=0.5,
        use_projection=True
    )

    try:
        checkpoint = torch.load('../build/models/reduced_graphsage_best.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded trained model from ./models/reduced_graphsage_best.pth")
    except Exception as e:
        print(f"Warning: Could not load trained model: {e}")
        print("Using random weights for test vector generation")

    # Generate test vectors
    output_dir = '../build/test_vectors'

    print("\n" + "="*60)
    print("Generating test vectors for full network...")
    print("="*60)
    generate_test_vectors_for_network(model, subgraph_data, output_dir)

    print("\n" + "="*60)
    print("Test vector generation complete!")
    print("="*60)
    print(f"\nTest vectors saved to: {output_dir}/")
    print("\nYou can now run the HLS testbench with these vectors.")
