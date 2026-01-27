#!/usr/bin/env python3
"""
Brevitas quantization script for GraphSAGE models.
Loads a trained float model, converts to Brevitas quantized model,
runs calibration, and exports quantized weights and configuration.
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path

try:
    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures
except ImportError:
    print("Error: torch_geometric not installed")
    sys.exit(1)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_base import ReducedGraphSAGE
from brevitas_models import (
    BrevitasQuantConfig, 
    check_brevitas_available,
    BrevitasReducedGraphSAGE, 
    load_from_float_model
)
from config import get_config


def load_cora_dataset(root='data'):
    """Load and prepare Cora dataset."""
    dataset = Planetoid(root=root, name='Cora', transform=NormalizeFeatures())
    data = dataset[0]
    return dataset, data


def calibrate_model(model, data, edge_index, num_batches=10, device='cpu'):
    """
    Run calibration/inference passes to collect quantization statistics.
    
    Args:
        model: Brevitas quantized model
        data: Data object with node features
        edge_index: Edge connectivity
        num_batches: Number of calibration passes (for statistics)
        device: Device to run on
    
    Returns:
        model: Calibrated model
    """
    model.eval()
    model.to(device)
    
    x = data.x.to(device)
    edge_index = edge_index.to(device)
    
    print(f"\nRunning calibration with {num_batches} passes...")
    
    with torch.no_grad():
        for i in range(num_batches):
            _ = model(x, edge_index)
            if (i + 1) % 5 == 0:
                print(f"  Calibration pass {i + 1}/{num_batches}")
    
    print("✓ Calibration complete")
    return model


def extract_quant_weight(layer, layer_name):
    """
    Extract quantized integer weights and scale from a Brevitas layer.
    
    Uses the built-in .int property from Brevitas QuantTensor.
    For fixed-point quantizers, the scale is power-of-two.
    
    Args:
        layer: Brevitas QuantLinear layer
        layer_name: Name for logging
    
    Returns:
        dict with 'int_weight', 'scale', 'zero_point', 'float_weight', 'fractional_bits'
    """
    result = {}
    
    try:
        with torch.no_grad():
            quant_weight = layer.quant_weight()
            
            # Use built-in .int property (returns integer representation)
            if hasattr(quant_weight, 'int'):
                int_weight = quant_weight.int().detach().cpu()
                # Cast to int8 if quantized to 8 bits
                result['int_weight'] = int_weight.to(torch.int8).numpy()
            else:
                result['int_weight'] = None
            
            # Get scale from QuantTensor
            if hasattr(quant_weight, 'scale') and quant_weight.scale is not None:
                scale_value = quant_weight.scale.detach().cpu()
                result['scale'] = scale_value.numpy()
                
                # For fixed-point quantizers, scale is power-of-two
                # Fractional bits = -log2(scale)
                if scale_value.numel() == 1 and scale_value.item() > 0:
                    fractional_bits = -torch.log2(scale_value)
                    result['fractional_bits'] = fractional_bits.item()
                else:
                    result['fractional_bits'] = None
            else:
                result['scale'] = None
                result['fractional_bits'] = None
            
            # Zero point from QuantTensor
            if hasattr(quant_weight, 'zero_point') and quant_weight.zero_point is not None:
                zp_value = quant_weight.zero_point.detach().cpu()
                result['zero_point'] = zp_value.numpy()
            else:
                result['zero_point'] = None
            
            # Float weight for reference
            result['float_weight'] = layer.weight.data.detach().cpu().numpy()
            
            # Verify quantization roundtrip if we have both int and scale
            if result['int_weight'] is not None and result['scale'] is not None:
                dequant = result['int_weight'].astype(np.float32) * result['scale']
                max_diff = np.abs(dequant - result['float_weight']).max()
                if max_diff > 0.5:
                    print(f"  ⚠ Warning: {layer_name} quantization roundtrip error (max_diff={max_diff:.4f})")
        
    except Exception as e:
        print(f"  ⚠ Warning: Could not extract quantized weight from {layer_name}: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to float weight
        result['float_weight'] = layer.weight.data.detach().cpu().numpy()
        result['int_weight'] = None
        result['scale'] = None
        result['zero_point'] = None
        result['fractional_bits'] = None
    
    return result


def export_quantized_weights(model, output_dir, quant_config):
    """
    Export quantized integer weights and biases from Brevitas model.
    
    Exports both:
    - Integer quantized weights (via quant_weight())
    - Float weights and scales for reference
    
    Args:
        model: Calibrated Brevitas model
        output_dir: Directory to save weights
        quant_config: Quantization configuration
    """
    weights_dir = Path(output_dir) / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    print(f"\nExporting quantized weights to {weights_dir}/")
    
    # Export projection weights if present
    if model.use_projection:
        proj_data = extract_quant_weight(model.projection, 'projection')
        
        if proj_data['int_weight'] is not None:
            np.savetxt(weights_dir / 'projection_weight_int.txt', proj_data['int_weight'].flatten(), fmt='%d')
            with open(weights_dir / 'projection_weight_int.shape', 'w') as f:
                f.write(','.join(map(str, proj_data['int_weight'].shape)))
            if proj_data['scale'] is not None:
                np.savetxt(weights_dir / 'projection_weight_scale.txt', proj_data['scale'].flatten(), fmt='%.10f')
            if proj_data.get('fractional_bits') is not None:
                with open(weights_dir / 'projection_weight_frac_bits.txt', 'w') as f:
                    f.write(f"{proj_data['fractional_bits']:.1f}")
        
        # Float reference
        np.savetxt(weights_dir / 'projection_weight_float.txt', proj_data['float_weight'].flatten(), fmt='%.6f')
        
        # Bias (typically not quantized)
        proj_bias = model.projection.bias.data.cpu().numpy()
        np.savetxt(weights_dir / 'projection_bias.txt', proj_bias.flatten(), fmt='%.6f')
        with open(weights_dir / 'projection_bias.shape', 'w') as f:
            f.write(','.join(map(str, proj_bias.shape)))
        
        if proj_data['int_weight'] is not None:
            print(f"  ✓ projection: int_weight {proj_data['int_weight'].shape}, scale {proj_data['scale'].shape if proj_data['scale'] is not None else 'N/A'}, bias {proj_bias.shape}")
        else:
            print(f"  ✓ projection: float_weight {proj_data['float_weight'].shape}, bias {proj_bias.shape}")
    
    # Export conv1 neighbor weights
    conv1_neighbor_data = extract_quant_weight(model.conv1.lin_neighbor, 'conv1_neighbor')
    
    if conv1_neighbor_data['int_weight'] is not None:
        np.savetxt(weights_dir / 'conv1_neighbor_weight_int.txt', conv1_neighbor_data['int_weight'].flatten(), fmt='%d')
        with open(weights_dir / 'conv1_neighbor_weight_int.shape', 'w') as f:
            f.write(','.join(map(str, conv1_neighbor_data['int_weight'].shape)))
        if conv1_neighbor_data['scale'] is not None:
            np.savetxt(weights_dir / 'conv1_neighbor_weight_scale.txt', conv1_neighbor_data['scale'].flatten(), fmt='%.10f')
    
    np.savetxt(weights_dir / 'conv1_neighbor_weight_float.txt', conv1_neighbor_data['float_weight'].flatten(), fmt='%.6f')
    
    conv1_neighbor_bias = model.conv1.lin_neighbor.bias.data.cpu().numpy()
    np.savetxt(weights_dir / 'conv1_neighbor_bias.txt', conv1_neighbor_bias.flatten(), fmt='%.6f')
    with open(weights_dir / 'conv1_neighbor_bias.shape', 'w') as f:
        f.write(','.join(map(str, conv1_neighbor_bias.shape)))
    
    if conv1_neighbor_data['int_weight'] is not None:
        print(f"  ✓ conv1_neighbor: int_weight {conv1_neighbor_data['int_weight'].shape}, scale {conv1_neighbor_data['scale'].shape if conv1_neighbor_data['scale'] is not None else 'N/A'}, bias {conv1_neighbor_bias.shape}")
    else:
        print(f"  ✓ conv1_neighbor: float_weight {conv1_neighbor_data['float_weight'].shape}, bias {conv1_neighbor_bias.shape}")
    
    # Export conv1 root weights if present
    if model.root_weight and model.conv1.lin_root is not None:
        conv1_root_data = extract_quant_weight(model.conv1.lin_root, 'conv1_root')
        
        if conv1_root_data['int_weight'] is not None:
            np.savetxt(weights_dir / 'conv1_root_weight_int.txt', conv1_root_data['int_weight'].flatten(), fmt='%d')
            with open(weights_dir / 'conv1_root_weight_int.shape', 'w') as f:
                f.write(','.join(map(str, conv1_root_data['int_weight'].shape)))
            if conv1_root_data['scale'] is not None:
                np.savetxt(weights_dir / 'conv1_root_weight_scale.txt', conv1_root_data['scale'].flatten(), fmt='%.10f')
        
        np.savetxt(weights_dir / 'conv1_root_weight_float.txt', conv1_root_data['float_weight'].flatten(), fmt='%.6f')
        
        if conv1_root_data['int_weight'] is not None:
            print(f"  ✓ conv1_root: int_weight {conv1_root_data['int_weight'].shape}")
        else:
            print(f"  ✓ conv1_root: float_weight {conv1_root_data['float_weight'].shape}")
    
    # Export conv2 neighbor weights
    conv2_neighbor_data = extract_quant_weight(model.conv2.lin_neighbor, 'conv2_neighbor')
    
    if conv2_neighbor_data['int_weight'] is not None:
        np.savetxt(weights_dir / 'conv2_neighbor_weight_int.txt', conv2_neighbor_data['int_weight'].flatten(), fmt='%d')
        with open(weights_dir / 'conv2_neighbor_weight_int.shape', 'w') as f:
            f.write(','.join(map(str, conv2_neighbor_data['int_weight'].shape)))
        if conv2_neighbor_data['scale'] is not None:
            np.savetxt(weights_dir / 'conv2_neighbor_weight_scale.txt', conv2_neighbor_data['scale'].flatten(), fmt='%.10f')
    
    np.savetxt(weights_dir / 'conv2_neighbor_weight_float.txt', conv2_neighbor_data['float_weight'].flatten(), fmt='%.6f')
    
    conv2_neighbor_bias = model.conv2.lin_neighbor.bias.data.cpu().numpy()
    np.savetxt(weights_dir / 'conv2_neighbor_bias.txt', conv2_neighbor_bias.flatten(), fmt='%.6f')
    with open(weights_dir / 'conv2_neighbor_bias.shape', 'w') as f:
        f.write(','.join(map(str, conv2_neighbor_bias.shape)))
    
    if conv2_neighbor_data['int_weight'] is not None:
        print(f"  ✓ conv2_neighbor: int_weight {conv2_neighbor_data['int_weight'].shape}, scale {conv2_neighbor_data['scale'].shape if conv2_neighbor_data['scale'] is not None else 'N/A'}, bias {conv2_neighbor_bias.shape}")
    else:
        print(f"  ✓ conv2_neighbor: float_weight {conv2_neighbor_data['float_weight'].shape}, bias {conv2_neighbor_bias.shape}")
    
    # Export conv2 root weights if present
    if model.root_weight and model.conv2.lin_root is not None:
        conv2_root_data = extract_quant_weight(model.conv2.lin_root, 'conv2_root')
        
        if conv2_root_data['int_weight'] is not None:
            np.savetxt(weights_dir / 'conv2_root_weight_int.txt', conv2_root_data['int_weight'].flatten(), fmt='%d')
            with open(weights_dir / 'conv2_root_weight_int.shape', 'w') as f:
                f.write(','.join(map(str, conv2_root_data['int_weight'].shape)))
            if conv2_root_data['scale'] is not None:
                np.savetxt(weights_dir / 'conv2_root_weight_scale.txt', conv2_root_data['scale'].flatten(), fmt='%.10f')
        
        np.savetxt(weights_dir / 'conv2_root_weight_float.txt', conv2_root_data['float_weight'].flatten(), fmt='%.6f')
        
        if conv2_root_data['int_weight'] is not None:
            print(f"  ✓ conv2_root: int_weight {conv2_root_data['int_weight'].shape}")
        else:
            print(f"  ✓ conv2_root: float_weight {conv2_root_data['float_weight'].shape}")
    
    print(f"\n✓ Exported quantized weights to {weights_dir}/")
    print(f"   Format: *_int.txt (integer weights), *_scale.txt (scales), *_float.txt (reference)")


def export_quantization_scales(model, output_dir):
    """
    Export quantization scales and zero points from Brevitas model.
    
    Args:
        model: Calibrated Brevitas model
        output_dir: Directory to save scales
    """
    debug_dir = Path(output_dir) / 'debug'
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    scales_info = {}
    
    print(f"\nExporting quantization scales to {debug_dir}/")
    
    model.eval()
    
    # Helper to extract scale from QuantTensor or quantizer
    def get_scale_info(module, prefix):
        info = {}
        try:
            # Try to get weight quantizer info
            if hasattr(module, 'quant_weight') and hasattr(module.quant_weight, 'scale'):
                scale = module.quant_weight.scale()
                if scale is not None:
                    info[f'{prefix}_weight_scale'] = float(scale.item()) if scale.numel() == 1 else scale.detach().cpu().numpy().tolist()
            
            # Try to get output quantizer info
            if hasattr(module, 'quant_output') and hasattr(module.quant_output, 'scale'):
                scale = module.quant_output.scale()
                if scale is not None:
                    info[f'{prefix}_output_scale'] = float(scale.item()) if scale.numel() == 1 else scale.detach().cpu().numpy().tolist()
        except Exception as e:
            print(f"  Warning: Could not extract scale from {prefix}: {e}")
        
        return info
    
    # Extract scales from each layer
    if model.use_projection:
        scales_info.update(get_scale_info(model.projection, 'projection'))
        scales_info.update(get_scale_info(model.projection_act, 'projection_act'))
    
    scales_info.update(get_scale_info(model.conv1.lin_neighbor, 'conv1_neighbor'))
    if model.root_weight and model.conv1.lin_root is not None:
        scales_info.update(get_scale_info(model.conv1.lin_root, 'conv1_root'))
    scales_info.update(get_scale_info(model.conv1_act, 'conv1_act'))
    
    scales_info.update(get_scale_info(model.conv2.lin_neighbor, 'conv2_neighbor'))
    if model.root_weight and model.conv2.lin_root is not None:
        scales_info.update(get_scale_info(model.conv2.lin_root, 'conv2_root'))
    
    # Save scales
    scales_file = debug_dir / 'quantization_scales.json'
    with open(scales_file, 'w') as f:
        json.dump(scales_info, f, indent=2)
    
    print(f"✓ Exported {len(scales_info)} scale parameters to {scales_file}")


def evaluate_model(model, data, edge_index, device='cpu'):
    """
    Evaluate model accuracy on test set.
    
    Args:
        model: Model to evaluate
        data: Data object
        edge_index: Edge connectivity
        device: Device to run on
    
    Returns:
        test_acc: Test accuracy
    """
    model.eval()
    model.to(device)
    
    x = data.x.to(device)
    edge_index = edge_index.to(device)
    y = data.y.to(device)
    test_mask = data.test_mask.to(device)
    
    with torch.no_grad():
        out = model(x, edge_index)
        pred = out.argmax(dim=1)
        test_acc = (pred[test_mask] == y[test_mask]).float().mean().item()
    
    return test_acc


def main():
    parser = argparse.ArgumentParser(description='Brevitas Quantization for GraphSAGE')
    
    # Model configuration
    parser.add_argument('--root-weight', action='store_true',
                       help='Use model with root_weight=True (default is False for HLS)')
    parser.add_argument('--no-projection', dest='use_projection', action='store_false',
                       help='Disable projection layer')
    parser.set_defaults(use_projection=True)
    
    # Quantization bit widths
    parser.add_argument('--w-bits', type=int, default=8,
                       help='Weight bit width (default: 8)')
    parser.add_argument('--a-bits', type=int, default=8,
                       help='Activation bit width (default: 8)')
    parser.add_argument('--bias-bits', type=int, default=32,
                       help='Bias bit width (default: 32)')
    parser.add_argument('--acc-bits', type=int, default=32,
                       help='Accumulator bit width (default: 32)')
    
    # Calibration
    parser.add_argument('--calibration-batches', type=int, default=10,
                       help='Number of calibration passes (default: 10)')
    
    # Output
    parser.add_argument('--export-dir', type=str, default='build/brevitas',
                       help='Export directory (default: build/brevitas)')
    
    # Device
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (default: cpu)')
    
    args = parser.parse_args()
    
    # Check Brevitas availability
    if not check_brevitas_available():
        print("ERROR: Brevitas is not installed. Install with: pip install brevitas")
        return 1
    
    print("="*60)
    print("Brevitas Quantization for GraphSAGE")
    print("="*60)
    
    # Load configuration
    cfg = get_config()
    
    # Model selection
    root_weight = args.root_weight
    use_projection = args.use_projection
    suffix = "" if root_weight else "_no_root"
    
    # Use Path to find models relative to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    model_path = project_root / 'build' / 'models' / f'reduced_graphsage{suffix}_best.pth'
    
    print(f"\n📋 Configuration:")
    print(f"   Float model: {model_path}")
    print(f"   root_weight: {root_weight}")
    print(f"   use_projection: {use_projection}")
    print(f"   Architecture: {cfg.num_features} → {cfg.reduced_in_channels} → {cfg.reduced_hidden_channels} → {cfg.num_classes}")
    print(f"\n📊 Quantization Settings:")
    print(f"   Weight bits: {args.w_bits}")
    print(f"   Activation bits: {args.a_bits}")
    print(f"   Bias bits: {args.bias_bits}")
    print(f"   Accumulator bits: {args.acc_bits}")
    print(f"   Calibration batches: {args.calibration_batches}")
    
    # Create output directory (resolve relative to project root)
    output_dir = Path(args.export_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    if root_weight:
        output_dir = output_dir / 'with_root'
    else:
        output_dir = output_dir / 'no_root'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Export directory: {output_dir}")
    
    # Load dataset
    print("\n📦 Loading Cora dataset...")
    dataset, data = load_cora_dataset()
    print(f"   Nodes: {data.x.shape[0]}, Edges: {data.edge_index.shape[1]}")
    print(f"   Features: {data.x.shape[1]}, Classes: {dataset.num_classes}")
    
    # Load float model
    print(f"\n🔧 Loading float model from {model_path}...")
    float_model = ReducedGraphSAGE(
        in_channels=cfg.num_features,
        in_channels_reduced=cfg.reduced_in_channels,
        hidden_channels=cfg.reduced_hidden_channels,
        out_channels=cfg.num_classes,
        dropout=cfg.reduced_dropout,
        use_projection=use_projection,
        root_weight=root_weight
    )
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        # Handle both dict-wrapped and raw state_dict formats
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        float_model.load_state_dict(state_dict)
        print(f"✓ Loaded trained weights from checkpoint")
        
        # Evaluate float model
        float_acc = evaluate_model(float_model, data, data.edge_index, device=args.device)
        print(f"   Float model test accuracy: {float_acc:.4f}")
    except Exception as e:
        print(f"Warning: Could not load trained model: {e}")
        print("Using random weights for demonstration.")
        float_acc = None
    
    # Create Brevitas quantization config
    print(f"\n⚙️ Creating Brevitas quantized model...")
    quant_config = BrevitasQuantConfig(
        weight_bit_width=args.w_bits,
        act_bit_width=args.a_bits,
        enable_bias_quant=False,  # Keep biases in higher precision
        return_quant_tensor=True,
    )
    
    # Create Brevitas model
    brevitas_model = BrevitasReducedGraphSAGE(
        in_channels=cfg.num_features,
        in_channels_reduced=cfg.reduced_in_channels,
        hidden_channels=cfg.reduced_hidden_channels,
        out_channels=cfg.num_classes,
        dropout=cfg.reduced_dropout,
        use_projection=use_projection,
        root_weight=root_weight,
        quant_config=quant_config
    )
    
    # Load weights from float model
    print(f"\n📥 Loading weights from float model into Brevitas model...")
    load_from_float_model(brevitas_model, float_model)
    
    # Run calibration
    print(f"\n🎯 Calibrating quantized model...")
    brevitas_model = calibrate_model(
        brevitas_model, data, data.edge_index,
        num_batches=args.calibration_batches,
        device=args.device
    )
    
    # Evaluate quantized model
    print(f"\n📊 Evaluating quantized model...")
    quant_acc = evaluate_model(brevitas_model, data, data.edge_index, device=args.device)
    print(f"   Quantized model test accuracy: {quant_acc:.4f}")
    
    if float_acc is not None:
        acc_drop = float_acc - quant_acc
        print(f"   Accuracy drop: {acc_drop:.4f} ({acc_drop * 100:.2f}%)")
    
    # Save model state dict
    print(f"\n💾 Saving quantized model...")
    model_file = output_dir / 'model_brevitas.pth'
    torch.save({
        'model_state_dict': brevitas_model.state_dict(),
        'quant_config': quant_config.to_dict(),
        'architecture': {
            'in_channels': cfg.num_features,
            'in_channels_reduced': cfg.reduced_in_channels,
            'hidden_channels': cfg.reduced_hidden_channels,
            'out_channels': cfg.num_classes,
            'use_projection': use_projection,
            'root_weight': root_weight,
        },
        'test_accuracy': quant_acc,
    }, model_file)
    print(f"✓ Saved model to {model_file}")
    
    # Export quantization config
    config_file = output_dir / 'quant_config.json'
    with open(config_file, 'w') as f:
        json.dump({
            'quant_config': quant_config.to_dict(),
            'architecture': {
                'in_channels': cfg.num_features,
                'in_channels_reduced': cfg.reduced_in_channels,
                'hidden_channels': cfg.reduced_hidden_channels,
                'out_channels': cfg.num_classes,
                'use_projection': use_projection,
                'root_weight': root_weight,
            },
            'test_accuracy': quant_acc,
            'float_accuracy': float_acc,
        }, f, indent=2)
    print(f"✓ Saved config to {config_file}")
    
    # Export weights
    export_quantized_weights(brevitas_model, output_dir, quant_config)
    
    # Export scales
    export_quantization_scales(brevitas_model, output_dir)
    
    print("\n" + "="*60)
    print("✓ Brevitas Quantization Complete!")
    print("="*60)
    print(f"\nGenerated artifacts in {output_dir}:")
    print(f"  - model_brevitas.pth       (Brevitas model state dict)")
    print(f"  - quant_config.json        (Quantization configuration)")
    print(f"  - weights/                 (Quantized integer weights + float reference)")
    print(f"  - debug/                   (Quantization scales)")
    print("\nWeight files:")
    print(f"  - *_weight_int.txt         INT{args.w_bits} quantized weights")
    print(f"  - *_weight_scale.txt       Quantization scales")
    print(f"  - *_weight_float.txt       Float reference weights")
    print(f"  - *_bias.txt               Biases (not quantized)")
    print("\nQuantization scope:")
    print(f"  - Linear transforms: INT{args.w_bits} weights, INT{args.a_bits} activations")
    print(f"  - Graph aggregation: Float (mean aggregation not quantized)")
    print("\nNext steps:")
    print(f"  - Review accuracy: {quant_acc:.4f}")
    print(f"  - Use *_weight_int.txt and *_weight_scale.txt for HLS")
    print(f"  - Generate test vectors for HLS validation")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
