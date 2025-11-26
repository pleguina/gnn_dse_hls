#!/usr/bin/env python3
"""
Generate test vectors for PTQ (Post-Training Quantization) HLS validation
Uses QUANTIZED forward pass instead of float forward pass
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model_base import ReducedGraphSAGE
from quantization import quantize_tensor, dequantize_tensor
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from subgraph_extraction import extract_fixed_subgraph

def save_matrix(matrix, filename):
    """Save float matrix to text file"""
    with open(filename, 'w') as f:
        for row in matrix:
            f.write(' '.join(map(str, row)) + '\n')

def save_int_matrix(matrix, filename):
    """Save integer matrix to text file"""
    with open(filename, 'w') as f:
        for row in matrix:
            f.write(' '.join(map(lambda x: str(int(x)), row)) + '\n')

def quantized_aggregate(x_int8, edge_index, adj_matrix, num_nodes, scale_in, scale_hidden):
    """Perform quantized aggregation like HLS"""
    # Dequantize to float
    x_float = x_int8.float() * scale_in
    
    # Aggregate in float
    agg_float = torch.zeros((num_nodes, x_float.shape[1]))
    for i in range(num_nodes):
        agg_float[i] = torch.sum(x_float * adj_matrix[:, i:i+1], dim=0)
    
    # Quantize to hidden scale
    agg_int8 = torch.round(agg_float / scale_hidden).clamp(-128, 127).to(torch.int8)
    
    return agg_int8

def quantized_linear(x_int8, weight_int8, bias_int32, scale_in, scale_w, scale_out):
    """Perform quantized linear transformation like HLS"""
    # Compute in int32
    acc_int32 = torch.mm(x_int8.to(torch.int32), weight_int8.t().to(torch.int32)) + bias_int32
    
    # Requantize: acc_int32 * (scale_in * scale_w) / scale_out
    scale_factor = (scale_in * scale_w) / scale_out
    out_int8 = torch.round(acc_int32.float() * scale_factor).clamp(-128, 127).to(torch.int8)
    
    return out_int8

def main():
    print("="*60)
    print("Generating PTQ Test Vectors (Quantized Forward Pass)")
    print("="*60)

    # Load Cora dataset
    print("\nLoading Cora dataset...")
    import torch_geometric
    torch.serialization.add_safe_globals([torch_geometric.data.data.Data])
    dataset = Planetoid(root='../data', name='Cora', transform=NormalizeFeatures())
    data = dataset[0]

    # Extract subgraph
    print("Extracting subgraph...")
    subgraph_data = extract_fixed_subgraph(data, num_nodes=8, center_node=0, num_hops=2)

    # Load trained model
    print("Loading trained model...")
    model = ReducedGraphSAGE(
        in_channels=dataset.num_features,
        in_channels_reduced=16,
        hidden_channels=24,
        out_channels=dataset.num_classes,
        dropout=0.5,
        root_weight=False
    )
    
    try:
        checkpoint = torch.load('../build/models/reduced_graphsage_no_root_best.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Loaded model: reduced_graphsage_no_root_best.pth")
    except Exception as e:
        print(f"Warning: Could not load trained model: {e}")
        return

    model.eval()
    output_dir = '../build/test_vectors_ptq'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get projected features
    features = subgraph_data['x']  # Use 'x' key instead of 'features'
    with torch.no_grad():
        if hasattr(model, 'projection') and model.projection is not None:
            features_tensor = model.projection(torch.from_numpy(features).float())
            features_tensor = torch.relu(features_tensor)
            features = features_tensor.numpy()
        else:
            features = features

    # Build adjacency matrix
    edge_index = subgraph_data['edge_index']
    num_nodes = features.shape[0]
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        adj_matrix[src, dst] = 1.0
    # Normalize by degree
    degree = np.sum(adj_matrix, axis=0, keepdims=True)
    degree[degree == 0] = 1
    adj_matrix = adj_matrix / degree

    # Quantize input
    features_tensor = torch.from_numpy(features).float()
    features_quant, scale_in, _ = quantize_tensor(features_tensor, num_bits=8)

    # Get layer 1 weights and quantize
    conv1 = model.conv1
    weights1 = conv1.lin_l.weight.data
    bias1 = conv1.lin_l.bias.data if conv1.lin_l.bias is not None else torch.zeros(weights1.shape[0])
    
    weights1_quant, scale_w1, _ = quantize_tensor(weights1, num_bits=8)
    bias1_quant = (bias1 / (scale_in * scale_w1)).to(torch.int32)

    # Get layer 2 weights and quantize
    conv2 = model.conv2
    weights2 = conv2.lin_l.weight.data
    bias2 = conv2.lin_l.bias.data if conv2.lin_l.bias is not None else torch.zeros(weights2.shape[0])
    
    weights2_quant, scale_w2, _ = quantize_tensor(weights2, num_bits=8)
    scale_hidden = 0.1
    bias2_quant = (bias2 / (scale_hidden * scale_w2)).to(torch.int32)

    # QUANTIZED FORWARD PASS
    print("\n" + "="*60)
    print("Running quantized forward pass...")
    print("="*60)
    
    adj_tensor = torch.from_numpy(adj_matrix).float()
    
    # Layer 1
    print("\nLayer 1:")
    print(f"  Input (INT8) shape: {features_quant.shape}")
    agg1 = quantized_aggregate(features_quant, edge_index, adj_tensor, num_nodes, scale_in, scale_hidden)
    print(f"  Aggregated (INT8) shape: {agg1.shape}")
    print(f"  Agg1 sample [0,:5]: {agg1[0,:5].tolist()}")
    
    hidden = quantized_linear(agg1, weights1_quant, bias1_quant, scale_hidden, scale_w1, scale_hidden)
    print(f"  After linear (INT8) shape: {hidden.shape}")
    
    # ReLU
    hidden = torch.clamp(hidden, min=0)
    print(f"  After ReLU sample [0,:5]: {hidden[0,:5].tolist()}")
    
    # Layer 2
    print("\nLayer 2:")
    scale_out = 0.1
    agg2 = quantized_aggregate(hidden, edge_index, adj_tensor, num_nodes, scale_hidden, scale_out)
    print(f"  Aggregated (INT8) shape: {agg2.shape}")
    
    output = quantized_linear(agg2, weights2_quant, bias2_quant, scale_out, scale_w2, scale_out)
    print(f"  Output (INT8) shape: {output.shape}")
    print(f"  Output sample [0,:]: {output[0,:].tolist()}")

    # Save all vectors
    print("\n" + "="*60)
    print("Saving test vectors...")
    print("="*60)
    
    save_matrix(adj_matrix, f'{output_dir}/adj_matrix.txt')
    save_int_matrix(edge_index.T, f'{output_dir}/edge_index.txt')
    save_int_matrix(features_quant.numpy(), f'{output_dir}/network_input.txt')
    save_int_matrix(weights1_quant.numpy(), f'{output_dir}/weights_layer1.txt')
    save_int_matrix(bias1_quant.numpy(), f'{output_dir}/bias_layer1.txt')
    save_int_matrix(weights2_quant.numpy(), f'{output_dir}/weights_layer2.txt')
    save_int_matrix(bias2_quant.numpy(), f'{output_dir}/bias_layer2.txt')
    save_int_matrix(output.numpy(), f'{output_dir}/network_output_reference.txt')

    # Save scales
    with open(f'{output_dir}/scales.txt', 'w') as f:
        f.write(f"scale_in: {scale_in}\n")
        f.write(f"scale_w1: {scale_w1}\n")
        f.write(f"scale_w2: {scale_w2}\n")
        f.write(f"scale_hidden: {scale_hidden}\n")
        f.write(f"scale_out: {scale_out}\n")

    print(f"\n✅ Test vectors saved to: {output_dir}/")
    print(f"   Input shape: {features_quant.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Scales: in={scale_in:.6f}, hidden={scale_hidden}, out={scale_out}")
    print("\nExpected output [0,:]: " + str(output[0,:].tolist()))

if __name__ == '__main__':
    main()
