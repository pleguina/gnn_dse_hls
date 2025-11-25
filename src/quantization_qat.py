"""
QAT quantization export for FPGA implementation.
Exports quantized weights, scales, and test vectors from QAT-trained model.
"""

import torch
import numpy as np
import json
import os
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data

# Fix for PyTorch 2.6+ weights_only default change
torch.serialization.add_safe_globals([Data])

from model_qat import ReducedGraphSAGEQAT
from config import get_config
from subgraph_extraction import extract_fixed_subgraph


def extract_qat_scales(model):
    """
    Extract quantization scales from QAT model's fake quantizers.

    Returns:
        Dictionary with scale information for each layer
    """
    scales = {}

    # Extract scales from conv1
    if hasattr(model.conv1, 'act_in_fake_quant'):
        scales['scale_in'] = model.conv1.act_in_fake_quant.scale.item()
        scales['scale_hidden'] = model.conv1.act_out_fake_quant.scale.item()
        scales['scale_w1'] = model.conv1.w_l_fake_quant.scale.detach().cpu().numpy()

        # Handle per-channel scales (take mean for simple reporting)
        if isinstance(scales['scale_w1'], np.ndarray):
            scales['scale_w1_per_channel'] = scales['scale_w1'].tolist()
            scales['scale_w1'] = float(scales['scale_w1'].mean())

    # Extract scales from conv2
    if hasattr(model.conv2, 'act_out_fake_quant'):
        scales['scale_out'] = model.conv2.act_out_fake_quant.scale.item()
        scales['scale_w2'] = model.conv2.w_l_fake_quant.scale.detach().cpu().numpy()

        # Handle per-channel scales
        if isinstance(scales['scale_w2'], np.ndarray):
            scales['scale_w2_per_channel'] = scales['scale_w2'].tolist()
            scales['scale_w2'] = float(scales['scale_w2'].mean())

    return scales


def quantize_with_scale(tensor, scale):
    """
    Quantize tensor using a given scale (symmetric quantization).

    Args:
        tensor: Float tensor to quantize
        scale: Quantization scale (can be scalar or per-channel)

    Returns:
        Quantized INT8 tensor
    """
    if isinstance(scale, np.ndarray):
        # Per-channel quantization
        scale_tensor = torch.from_numpy(scale).float().view(-1, 1)
        quantized = torch.clamp(
            torch.round(tensor / scale_tensor),
            -128, 127
        ).to(torch.int8)
    else:
        # Per-tensor quantization
        quantized = torch.clamp(
            torch.round(tensor / scale),
            -128, 127
        ).to(torch.int8)

    return quantized


def export_qat_weights(model, scales, output_dir):
    """
    Export quantized weights from QAT model to text files.

    Args:
        model: Trained QAT model
        scales: Dictionary of quantization scales
        output_dir: Directory to save weights
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract and quantize conv1 weights (lin_l)
    w1 = model.conv1.lin_l.weight.data  # Shape: [hidden_channels, in_channels_reduced]
    if 'scale_w1_per_channel' in scales:
        w1_scale = np.array(scales['scale_w1_per_channel'])
    else:
        w1_scale = scales['scale_w1']

    w1_quantized = quantize_with_scale(w1, w1_scale).cpu().numpy()
    np.savetxt(f'{output_dir}/weights_layer1_qat.txt', w1_quantized, fmt='%d')
    print(f"Exported layer 1 weights: {w1_quantized.shape}")

    # Extract and quantize conv1 bias
    if model.conv1.lin_l.bias is not None:
        b1 = model.conv1.lin_l.bias.data
        # Bias is quantized to INT32 accumulator domain
        # b_int32 = round(b_float / (scale_in * scale_w1))
        scale_b1 = scales['scale_in'] * scales['scale_w1']
        b1_quantized = torch.round(b1 / scale_b1).to(torch.int32).cpu().numpy()
        np.savetxt(f'{output_dir}/bias_layer1_qat.txt', b1_quantized, fmt='%d')
        print(f"Exported layer 1 bias: {b1_quantized.shape}")

    # Extract and quantize conv2 weights (lin_l)
    w2 = model.conv2.lin_l.weight.data  # Shape: [out_channels, hidden_channels]
    if 'scale_w2_per_channel' in scales:
        w2_scale = np.array(scales['scale_w2_per_channel'])
    else:
        w2_scale = scales['scale_w2']

    w2_quantized = quantize_with_scale(w2, w2_scale).cpu().numpy()
    np.savetxt(f'{output_dir}/weights_layer2_qat.txt', w2_quantized, fmt='%d')
    print(f"Exported layer 2 weights: {w2_quantized.shape}")

    # Extract and quantize conv2 bias
    if model.conv2.lin_l.bias is not None:
        b2 = model.conv2.lin_l.bias.data
        # b_int32 = round(b_float / (scale_hidden * scale_w2))
        scale_b2 = scales['scale_hidden'] * scales['scale_w2']
        b2_quantized = torch.round(b2 / scale_b2).to(torch.int32).cpu().numpy()
        np.savetxt(f'{output_dir}/bias_layer2_qat.txt', b2_quantized, fmt='%d')
        print(f"Exported layer 2 bias: {b2_quantized.shape}")

    # Export root weights if present
    if model.root_weight:
        # Layer 1 root weights
        w1_r = model.conv1.lin_r.weight.data
        w1_r_quantized = quantize_with_scale(w1_r, w1_scale).cpu().numpy()
        np.savetxt(f'{output_dir}/weights_layer1_root_qat.txt', w1_r_quantized, fmt='%d')
        print(f"Exported layer 1 root weights: {w1_r_quantized.shape}")

        # Layer 2 root weights
        w2_r = model.conv2.lin_r.weight.data
        w2_r_quantized = quantize_with_scale(w2_r, w2_scale).cpu().numpy()
        np.savetxt(f'{output_dir}/weights_layer2_root_qat.txt', w2_r_quantized, fmt='%d')
        print(f"Exported layer 2 root weights: {w2_r_quantized.shape}")


def export_qat_test_vectors(model, data, scales, output_dir, num_nodes=8, center_node=0, num_hops=2):
    """
    Generate test vectors from QAT model for HLS verification.
    Uses SAME subgraph extraction as PTQ for fair comparison.

    Args:
        model: Trained QAT model
        data: Full dataset
        scales: Dictionary of quantization scales
        output_dir: Directory to save test vectors
        num_nodes: Number of nodes in test subgraph
        center_node: Center node for k-hop extraction (default: 0)
        num_hops: Number of hops for neighborhood (default: 2)
    """
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    model.enable_fake_quant()

    # Extract the SAME subgraph as PTQ (for fair comparison)
    # This ensures both models are tested on the same connected subgraph
    subgraph_data = extract_fixed_subgraph(data, num_nodes=num_nodes, center_node=center_node, num_hops=num_hops)

    # Get node indices and features from extracted subgraph
    node_indices = torch.from_numpy(subgraph_data['subset_indices']).long()
    features = torch.from_numpy(subgraph_data['x']).float()

    # Get input features (after projection if used)
    with torch.no_grad():
        if model.use_projection:
            x_proj = model.projection(features)
            x_proj = torch.relu(x_proj)
            x_input = x_proj
        else:
            x_input = features

    # Quantize input features
    scale_in = scales['scale_in']
    x_input_quantized = quantize_with_scale(x_input, scale_in).cpu().numpy()
    np.savetxt(f'{output_dir}/network_input_qat.txt', x_input_quantized, fmt='%d')
    print(f"Exported input features: {x_input_quantized.shape}")

    # Use edge_index from extracted subgraph
    edge_index_sub = torch.from_numpy(subgraph_data['edge_index']).long()

    # Save edge index
    edge_index_np = edge_index_sub.cpu().numpy()
    np.savetxt(f'{output_dir}/edge_index_qat.txt', edge_index_np.T, fmt='%d')
    print(f"Exported edge index: {edge_index_np.shape}")

    # Use adjacency matrix from extracted subgraph (already normalized by extract_fixed_subgraph)
    adj_normalized = subgraph_data['adj_matrix']
    np.savetxt(f'{output_dir}/adj_matrix_qat.txt', adj_normalized, fmt='%.6f')
    print(f"Exported adjacency matrix: {adj_normalized.shape}")

    # Generate reference output
    with torch.no_grad():
        # Run forward pass on the SAME subgraph
        # Use the subgraph's edge_index, not the full graph
        if model.use_projection:
            x_start = x_input  # Already projected
        else:
            x_start = features

        out = model.conv1(x_start, edge_index_sub)
        out = torch.relu(out)
        out = model.conv2(out, edge_index_sub)

    # Quantize output
    scale_out = scales['scale_out']
    out_quantized = quantize_with_scale(out, scale_out).cpu().numpy()
    np.savetxt(f'{output_dir}/network_output_reference_qat.txt', out_quantized, fmt='%d')
    print(f"Exported reference output: {out_quantized.shape}")


def export_scales_and_params(scales, output_dir):
    """
    Export scales to text file and JSON format.

    Args:
        scales: Dictionary of quantization scales
        output_dir: Directory to save scales
    """
    # Export scales as text (human-readable for HLS)
    with open(f'{output_dir}/scales_qat.txt', 'w') as f:
        for key, value in scales.items():
            if not key.endswith('_per_channel'):
                f.write(f"{key}: {value}\n")
    print(f"Exported scales to scales_qat.txt")

    # Export full parameters as JSON
    quant_params = {
        'scales': scales,
        'dtype': 'int8',
        'accumulator_dtype': 'int32',
        'quantization_scheme': 'symmetric',
        'zero_point': 0,
    }

    with open(f'{output_dir}/quant_params_qat.json', 'w') as f:
        json.dump(quant_params, f, indent=2)
    print(f"Exported quantization parameters to quant_params_qat.json")


def main():
    """Main export function."""
    # Load configuration
    cfg = get_config()

    print("\n" + "="*60)
    print("QAT Quantization Export")
    print("="*60)
    print(f"📋 Using config: {cfg.get('active_qat_config', 'qat_model')}")

    # Load dataset
    dataset = Planetoid(root='../data', name='Cora', transform=NormalizeFeatures())
    data = dataset[0]

    # Load trained QAT model
    models_dir = cfg.get('paths.models', '../build/models')
    model_path = f'{models_dir}/reduced_graphsage_qat_no_root_best.pth'

    if not os.path.exists(model_path):
        print(f"[ERROR] QAT model not found at {model_path}")
        print("Please run train_qat.py first!")
        return

    checkpoint = torch.load(model_path, weights_only=False)

    # Reconstruct model
    model = ReducedGraphSAGEQAT(
        in_channels=dataset.num_features,
        in_channels_reduced=checkpoint['in_channels_reduced'],
        hidden_channels=checkpoint['hidden_channels'],
        out_channels=checkpoint['out_channels'],
        root_weight=checkpoint['root_weight'],
        use_projection=True,
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded QAT model from {model_path}")
    print(f"  Validation accuracy: {checkpoint['val_acc']:.4f}")

    # Extract scales from fake quantizers
    print("\nExtracting quantization scales...")
    scales = extract_qat_scales(model)

    print("\nQuantization scales:")
    for key, value in scales.items():
        if not key.endswith('_per_channel'):
            print(f"  {key}: {value}")

    # Create output directories from config
    weights_dir = cfg.get('quantization.output_dir_qat', '../build/quantized_qat')
    vectors_dir = cfg.get('quantization.test_vectors_dir_qat', '../build/test_vectors_qat')
    test_nodes = cfg.get('quantization.test_subgraph_nodes', 8)

    print(f"\nOutput directories:")
    print(f"  Weights: {weights_dir}")
    print(f"  Test vectors: {vectors_dir}")
    print(f"  Test subgraph nodes: {test_nodes}")

    # Export weights
    print("\nExporting quantized weights...")
    export_qat_weights(model, scales, weights_dir)

    # Export test vectors (using SAME parameters as PTQ for fair comparison)
    print("\nGenerating test vectors...")
    print("  Using same subgraph as PTQ: center_node=0, num_hops=2")
    export_qat_test_vectors(model, data, scales, vectors_dir, num_nodes=test_nodes, center_node=0, num_hops=2)

    # Export scales and parameters
    print("\nExporting scales and parameters...")
    export_scales_and_params(scales, vectors_dir)

    # Summary
    print("\n" + "="*60)
    print("QAT Export Complete!")
    print("="*60)
    print(f"\nQuantized weights saved to: {weights_dir}/")
    print(f"  - weights_layer1_qat.txt")
    print(f"  - bias_layer1_qat.txt")
    print(f"  - weights_layer2_qat.txt")
    print(f"  - bias_layer2_qat.txt")
    print(f"\nTest vectors saved to: {vectors_dir}/")
    print(f"  - network_input_qat.txt")
    print(f"  - network_output_reference_qat.txt")
    print(f"  - edge_index_qat.txt")
    print(f"  - adj_matrix_qat.txt")
    print(f"  - scales_qat.txt")
    print(f"  - quant_params_qat.json")
    print("\nThese files are ready for HLS implementation!")
    print("="*60)


if __name__ == '__main__':
    main()
