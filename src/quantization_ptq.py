"""
Quantization utilities for FPGA implementation.
Implements INT8 quantization for weights and activations.
"""

import torch
import numpy as np
import json
from model_base import ReducedGraphSAGE
from config import get_config


class QuantizationParams:
    """Store quantization parameters (scale and zero_point)."""

    def __init__(self):
        self.scales = {}
        self.zero_points = {}

    def add_param(self, name, scale, zero_point):
        self.scales[name] = float(scale)
        self.zero_points[name] = int(zero_point)

    def save(self, filepath):
        data = {
            'scales': self.scales,
            'zero_points': self.zero_points
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


def quantize_tensor(tensor, num_bits=8, symmetric=True):
    """
    Quantize a tensor to int representation.

    Args:
        tensor: Input tensor to quantize
        num_bits: Number of bits for quantization (default 8 for int8)
        symmetric: Use symmetric quantization (zero_point=0)

    Returns:
        quantized: Quantized tensor (int representation)
        scale: Quantization scale
        zero_point: Quantization zero point
    """
    if symmetric:
        # Symmetric quantization: zero_point = 0
        max_val = max(abs(tensor.min().item()), abs(tensor.max().item()))
        qmax = 2 ** (num_bits - 1) - 1  # 127 for int8
        scale = max_val / qmax if max_val != 0 else 1.0
        zero_point = 0
    else:
        # Asymmetric quantization
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        qmin = -(2 ** (num_bits - 1))  # -128 for int8
        qmax = 2 ** (num_bits - 1) - 1  # 127 for int8
        scale = (max_val - min_val) / (qmax - qmin) if max_val != min_val else 1.0
        zero_point = qmin - int(min_val / scale)

    # Quantize
    quantized = torch.clamp(
        torch.round(tensor / scale) + zero_point,
        -(2 ** (num_bits - 1)),
        2 ** (num_bits - 1) - 1
    )

    return quantized.to(torch.int8), scale, zero_point


def dequantize_tensor(quantized, scale, zero_point):
    """Dequantize a tensor back to float representation."""
    return (quantized.to(torch.float32) - zero_point) * scale


def quantize_model_weights(model, num_bits=8):
    """
    Quantize all weights in the model.

    Args:
        model: PyTorch model to quantize
        num_bits: Number of bits for quantization

    Returns:
        quantized_weights: Dictionary of quantized weights
        quant_params: Quantization parameters
    """
    quantized_weights = {}
    quant_params = QuantizationParams()

    model.eval()

    for name, param in model.named_parameters():
        if len(param.shape) == 0:  # Skip scalar parameters
            continue

        # Quantize weight
        quantized, scale, zero_point = quantize_tensor(
            param.data, num_bits=num_bits, symmetric=True
        )

        quantized_weights[name] = quantized.numpy()
        quant_params.add_param(name, scale, zero_point)

        print(f"Quantized {name}: shape={param.shape}, scale={scale:.6f}")

    return quantized_weights, quant_params


def save_quantized_weights(quantized_weights, quant_params, output_dir='../build/quantized'):
    """Save quantized weights and parameters to files."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Save each weight as separate file
    for name, weight in quantized_weights.items():
        # Replace dots with underscores for filename
        filename = name.replace('.', '_') + '.txt'
        np.savetxt(f'{output_dir}/{filename}', weight.flatten(), fmt='%d')

        # Also save shape info
        with open(f'{output_dir}/{filename}.shape', 'w') as f:
            f.write(','.join(map(str, weight.shape)))

    # Save quantization parameters
    quant_params.save(f'{output_dir}/quant_params.json')

    print(f"\nQuantized weights saved to {output_dir}/")


def export_weights_as_c_header(quantized_weights, quant_params, header_file='../build/hls/weights.h', model_dims=None):
    """Export quantized weights as C/C++ header file for HLS."""
    import os
    os.makedirs(os.path.dirname(header_file), exist_ok=True)

    with open(header_file, 'w') as f:
        f.write("// Auto-generated quantized weights for FPGA implementation\n")
        f.write("#ifndef WEIGHTS_H\n")
        f.write("#define WEIGHTS_H\n\n")
        f.write("#include <stdint.h>\n\n")

        # Write model dimensions if provided
        if model_dims:
            f.write("// Model architecture dimensions\n")
            f.write(f"#define MODEL_IN_FEATURES {model_dims['in_features']}\n")
            f.write(f"#define MODEL_HIDDEN_FEATURES {model_dims['hidden_features']}\n")
            f.write(f"#define MODEL_OUT_FEATURES {model_dims['out_features']}\n")
            if 'num_nodes' in model_dims:
                f.write(f"#define MODEL_NUM_NODES {model_dims['num_nodes']}\n")
            f.write("\n")

        for name, weight in quantized_weights.items():
            # Create valid C identifier
            c_name = name.replace('.', '_').upper()

            # Write array
            f.write(f"// {name}\n")
            f.write(f"const int8_t {c_name}[{weight.size}] = {{\n")

            # Write values (16 per line)
            flat_weight = weight.flatten()
            for i in range(0, len(flat_weight), 16):
                chunk = flat_weight[i:i+16]
                f.write("    " + ", ".join(f"{int(x):4d}" for x in chunk))
                if i + 16 < len(flat_weight):
                    f.write(",")
                f.write("\n")

            f.write("};\n\n")

            # Write shape constants
            if len(weight.shape) == 2:
                f.write(f"const int {c_name}_ROWS = {weight.shape[0]};\n")
                f.write(f"const int {c_name}_COLS = {weight.shape[1]};\n\n")

        # Write scales as float constants
        f.write("// Quantization scales\n")
        for name, scale in quant_params.scales.items():
            c_name = name.replace('.', '_').upper()
            f.write(f"const float {c_name}_SCALE = {scale}f;\n")

        f.write("\n#endif // WEIGHTS_H\n")

    print(f"\nC header file saved to {header_file}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Quantize GraphSAGE model')
    parser.add_argument('--use-root-weight', action='store_true', 
                       help='Use model with root_weight=True (default is False for HLS)')
    args = parser.parse_args()
    
    root_weight = args.use_root_weight
    suffix = "" if root_weight else "_no_root"
    model_path = f'../build/models/reduced_graphsage{suffix}_best.pth'
    
    print(f"Using model: {model_path}")
    print(f"root_weight = {root_weight}")
    if not root_weight:
        print("NOTE: This is the HLS-compatible version (simpler formula)")
    
    # Load configuration
    cfg = get_config()

    # Load reduced model
    model = ReducedGraphSAGE(
        in_channels=cfg.num_features,
        in_channels_reduced=cfg.reduced_in_channels,
        hidden_channels=cfg.reduced_hidden_channels,
        out_channels=cfg.num_classes,
        dropout=cfg.reduced_dropout,
        use_projection=True,
        root_weight=root_weight
    )

    print(f"📋 Using config: reduced_model")
    print(f"   Architecture: {cfg.reduced_in_channels} → {cfg.reduced_hidden_channels} → {cfg.num_classes}")

    # Load trained weights
    try:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded trained model from {model_path}")
    except Exception as e:
        print(f"Warning: Could not load trained model: {e}")
        print("Using random weights.")

    # Quantize model
    print("\nQuantizing model to INT8...")
    quantized_weights, quant_params = quantize_model_weights(model, num_bits=8)

    # Save quantized weights with new naming convention
    if root_weight:
        output_dir = '../build/weights_ptq_float_with_root'
    else:
        output_dir = '../build/weights_ptq_float'
    save_quantized_weights(quantized_weights, quant_params, output_dir=output_dir)

    # Prepare model dimensions
    model_dims = {
        'in_features': 16,          # After projection
        'hidden_features': 24,       # Conv1 output
        'out_features': 7,           # Number of classes
        'num_nodes': 32             # Max nodes for FPGA
    }

    # Export as C header with dimensions
    header_path = f'../build/hls/weights{suffix}.h'
    export_weights_as_c_header(quantized_weights, quant_params, 
                               header_file=header_path, model_dims=model_dims)

    print("\nQuantization complete!")
    print(f"Output directory: {output_dir}")
    print(f"C header: {header_path}")
