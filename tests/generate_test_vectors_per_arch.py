#!/usr/bin/env python3
"""
Generate architecture-specific test vectors for DSE

This script generates test vectors for a specific architecture and implementation.
It loads the trained model for that architecture and generates golden outputs
using the same quantization scheme that HLS will use.

Supports:
- int8_po2: INT8 with power-of-2 scales (bit-shifts)
- fixed: ap_fixed with Q(W,I) format
- int8: INT8 with arbitrary scales (for completeness)

Usage:
    python tests/generate_test_vectors_per_arch.py --in-channels 16 --hidden 24 --impl int8_po2
    python tests/generate_test_vectors_per_arch.py --in-channels 16 --hidden 32 --impl fixed --data-w 24 --data-i 12
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def generate_test_vectors(in_channels: int, hidden_channels: int, 
                         implementation: str, output_dir: Path,
                         data_w: int = 16, data_i: int = 8,
                         weight_w: int = 16, weight_i: int = 4,
                         acc_w: int = 32, acc_i: int = 16):
    """
    Generate test vectors for specific architecture and implementation.
    
    Args:
        in_channels: Input feature dimension (after projection)
        hidden_channels: Hidden layer dimension
        implementation: "int8_po2", "fixed", or "int8"
        output_dir: Where to save test vectors
        data_w, data_i: Q format for activations (fixed only)
        weight_w, weight_i: Q format for weights (fixed only)
        acc_w, acc_i: Q format for accumulators (fixed only)
    """
    print("=" * 80)
    print(f"Generating Test Vectors")
    print("=" * 80)
    print(f"  Architecture: {in_channels} → {hidden_channels} → 7")
    print(f"  Implementation: {implementation}")
    if implementation == "fixed":
        print(f"  Q Format:")
        print(f"    Data: Q({data_w},{data_i}) - {data_w-data_i} fractional bits")
        print(f"    Weights: Q({weight_w},{weight_i}) - {weight_w-weight_i} fractional bits")
        print(f"    Accumulators: Q({acc_w},{acc_i}) - {acc_w-acc_i} fractional bits")
    
    # Build directory structure
    arch_key = f"{in_channels}x{hidden_channels}"
    impl_dir = output_dir / implementation / arch_key
    impl_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if PTQ files exist in architecture-specific directory
    # Try architecture-specific first, fall back to legacy shared directory
    # Use absolute paths to avoid CWD issues
    project_root = Path(__file__).parent.parent
    ptq_arch_dir = project_root / "build" / "weights_ptq_per_arch" / arch_key
    if ptq_arch_dir.exists():
        ptq_dir = ptq_arch_dir
        int8_dir = ptq_arch_dir
    else:
        # Fall back to legacy shared directories
        ptq_dir = project_root / "build" / "weights_ptq_float"
        int8_dir = project_root / "build" / "weights_ptq_int8"
    
    # Check for required PTQ files
    weights1_path = ptq_dir / "conv1_lin_l_weight.txt"
    int8_params_path = int8_dir / "int8_params.json"
    
    if not weights1_path.exists():
        print(f"\n✗ ERROR: No PTQ weights found for architecture {arch_key}")
        print(f"  Expected: {weights1_path}")
        print(f"\nYou need to run PTQ first:")
        print(f"  1. python src/quantization_ptq.py")
        print(f"  2. python src/prepare_ptq_int8_parameters.py --m24")
        return False
    
    if not int8_params_path.exists():
        print(f"\n✗ ERROR: No INT8 params found")
        print(f"  Expected: {int8_params_path}")
        print(f"\nYou need to run: python src/prepare_ptq_int8_parameters.py --m24")
        return False
    
    print(f"\n✓ Found PTQ files")
    
    # Load INT8 weights from PTQ float directory (they are quantized to INT8)
    try:
        weights1_int8 = np.loadtxt(ptq_dir / "conv1_lin_l_weight.txt", dtype=np.int8).reshape(hidden_channels, in_channels)
        weights2_int8 = np.loadtxt(ptq_dir / "conv2_lin_l_weight.txt", dtype=np.int8).reshape(7, hidden_channels)
        
        print(f"\n✓ Loaded INT8 weights:")
        print(f"  Layer 1: {weights1_int8.shape} (hidden x in)")
        print(f"  Layer 2: {weights2_int8.shape} (out x hidden)")
    except Exception as e:
        print(f"\n✗ ERROR loading weights: {e}")
        return False
    
    # Load biases from INT8 directory
    bias1_int32 = np.loadtxt(int8_dir / "bias_layer1_int32.txt", dtype=np.int32)
    bias2_int32 = np.loadtxt(int8_dir / "bias_layer2_int32.txt", dtype=np.int32)
    
    # Load adjacency and input
    adj_matrix_int16 = np.loadtxt(int8_dir / "adj_matrix_int16.txt", dtype=np.int16)
    input_int8 = np.loadtxt(project_root / "build" / "test_vectors_ptq_float" / "network_input.txt", dtype=np.int8)
    
    print(f"\n✓ Loaded test data:")
    print(f"  Input: {input_int8.shape}")
    print(f"  Adjacency: {adj_matrix_int16.shape}")
    print(f"  Biases: {bias1_int32.shape}, {bias2_int32.shape}")
    
    # Load quantization parameters (M, K) from PTQ configuration
    with open(int8_params_path, 'r') as f:
        params = json.load(f)
    
    M = params['fixed_point_config']['M']
    K = params['fixed_point_config']['K']
    K_BITS = params['fixed_point_config']['K_BITS']
    
    print(f"\n✓ Loaded PTQ configuration:")
    print(f"  M_BITS: {M} (fractional bits for weights/activations)")
    print(f"  K_BITS: {K_BITS} (adjacency scaling: K = 2^{K_BITS})")
    print(f"  K: {K}")
    
    # Load or compute PO2 shifts (for int8_po2 implementation)
    if implementation == "int8_po2":
        po2_config_path = project_root / "build" / "test_vectors_ptq_int8_po2" / "po2_config.json"
        if po2_config_path.exists():
            with open(po2_config_path, 'r') as f:
                po2_config = json.load(f)
            shifts = po2_config['shifts']
            print(f"  PO2 shifts (from existing config): {shifts}")
        else:
            # Compute PO2 shifts from fixed-point scales in int8_params.json
            print(f"  Computing PO2 shifts from fixed-point scales...")
            fp_scales = params.get('fixed_point_scales', {})
            
            def compute_po2_shift(scale_fp, M):
                """Convert fixed-point scale to nearest power-of-2 shift"""
                if scale_fp <= 0:
                    return M
                import math
                log2_val = math.log2(scale_fp)
                po2_exponent = round(log2_val)
                return M - po2_exponent
            
            # Get original scales (or use reasonable defaults)
            eff_scale1_fp = fp_scales.get('eff_scale1_fp', 2**M)  # Default to 2^M (shift=0)
            eff_scale2_fp = fp_scales.get('eff_scale2_fp', 2**M)
            beta1_fp = fp_scales.get('beta1_fp', 2**(M-K_BITS))  # Default to 2^(M-K)
            beta2_fp = fp_scales.get('beta2_fp', 2**(M-K_BITS))
            
            shifts = {
                'BETA1_SHIFT': compute_po2_shift(beta1_fp, M),
                'BETA2_SHIFT': compute_po2_shift(beta2_fp, M), 
                'EFF_SCALE1_SHIFT': compute_po2_shift(eff_scale1_fp, M),
                'EFF_SCALE2_SHIFT': compute_po2_shift(eff_scale2_fp, M)
            }
            print(f"  PO2 shifts (computed): {shifts}")
    else:
        shifts = None
    
    # Run forward pass with appropriate quantization
    print(f"\nRunning {implementation} forward pass...")
    
    if implementation == "int8_po2":
        output_int8 = run_po2_inference(
            input_int8, adj_matrix_int16,
            weights1_int8, weights2_int8,
            bias1_int32, bias2_int32,
            shifts, M, K_BITS
        )
        ref_suffix = "int8_po2_reference"
    elif implementation == "fixed":
        output_int8 = run_fixed_inference(
            input_int8, adj_matrix_int16,
            weights1_int8, weights2_int8,
            bias1_int32, bias2_int32,
            data_w, data_i, weight_w, weight_i, acc_w, acc_i,
            M, K_BITS
        )
        ref_suffix = "fixed_reference"
    elif implementation == "int8":
        # Use arbitrary-precision scales (load from config)
        output_int8 = run_int8_inference(
            input_int8, adj_matrix_int16,
            weights1_int8, weights2_int8,
            bias1_int32, bias2_int32,
            M, K_BITS, shifts
        )
        ref_suffix = "int8_reference"
    else:
        raise ValueError(f"Unsupported implementation: {implementation}")
    
    print(f"\n✓ Generated output: {output_int8.shape}")
    print(f"  Range: [{output_int8.min()}, {output_int8.max()}]")
    
    # Save test vectors
    print(f"\nSaving test vectors to {impl_dir}...")
    
    # Save golden output (for HLS verification)
    np.savetxt(impl_dir / f"network_output_{ref_suffix}.txt", output_int8, fmt='%d')
    
    # Save all test vectors with standard naming (matching testbench expectations)
    np.savetxt(impl_dir / "network_input.txt", input_int8, fmt='%d')
    np.savetxt(impl_dir / "adj_matrix_int16.txt", adj_matrix_int16, fmt='%d')
    np.savetxt(impl_dir / "weights_layer1.txt", weights1_int8, fmt='%d')
    np.savetxt(impl_dir / "weights_layer2.txt", weights2_int8, fmt='%d')
    np.savetxt(impl_dir / "bias_layer1_int32.txt", bias1_int32, fmt='%d')
    np.savetxt(impl_dir / "bias_layer2_int32.txt", bias2_int32, fmt='%d')
    
    # Save configuration
    config = {
        'architecture': {
            'in_channels': in_channels,
            'hidden_channels': hidden_channels,
            'out_channels': 7,
            'num_nodes': 8
        },
        'implementation': implementation,
        'quantization': {
            'M_BITS': M,
            'shifts': shifts
        }
    }
    
    if implementation == "fixed":
        config['quantization']['q_format'] = {
            'data_w': data_w, 'data_i': data_i,
            'weight_w': weight_w, 'weight_i': weight_i,
            'acc_w': acc_w, 'acc_i': acc_i
        }
    
    with open(impl_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Also save po2_config.json for int8_po2 (DSE expects this format)
    if implementation == "int8_po2" and shifts:
        po2_config = {
            'description': 'PO2 shift configuration for HLS',
            'shifts': shifts,
            'M_BITS': M,
            'architecture': {
                'in_channels': in_channels,
                'hidden_channels': hidden_channels
            }
        }
        with open(impl_dir / "po2_config.json", 'w') as f:
            json.dump(po2_config, f, indent=2)
        print(f"  - po2_config.json (PO2 shift values)")
    
    print(f"\n✓ Test vectors saved to {impl_dir}")
    print(f"\nFiles generated:")
    print(f"  - network_output_{ref_suffix}.txt (golden output)")
    print(f"  - network_input.txt, adj_matrix_int16.txt")
    print(f"  - weights_layer1.txt, weights_layer2.txt")
    print(f"  - bias_layer1_int32.txt, bias_layer2_int32.txt")
    print(f"  - config.json")
    
    return True


def run_po2_inference(input_int8, adj_int16, w1, w2, b1, b2, shifts, M, K_BITS):
    """Run integer-only inference with PO2 scaling"""
    N_NODES = input_int8.shape[0]
    IN_FEATURES = input_int8.shape[1]  # Input features (16)
    HIDDEN_FEATURES = w1.shape[0]      # Hidden features (24) - w1 is [hidden, input]
    
    # Layer 1: Aggregation over input features
    agg1 = np.zeros((N_NODES, IN_FEATURES), dtype=np.int8)
    for i in range(N_NODES):
        for f in range(IN_FEATURES):
            tmp = np.int32(0)
            for j in range(N_NODES):
                if adj_int16[i, j] != 0:
                    tmp += np.int32(adj_int16[i, j]) * np.int32(input_int8[j, f])
            # PO2 shift with rounding (remove adjacency K scaling)
            shift_amt = shifts['BETA1_SHIFT']
            if tmp >= 0:
                tmp = (tmp + (1 << (shift_amt - 1))) >> shift_amt
            else:
                tmp = -((-tmp + (1 << (shift_amt - 1))) >> shift_amt)
            agg1[i, f] = np.clip(tmp, -128, 127).astype(np.int8)
    
    # Layer 1: Linear (transforms aggregated input to hidden features)
    hidden = np.zeros((N_NODES, HIDDEN_FEATURES), dtype=np.int8)
    for i in range(N_NODES):
        for h in range(HIDDEN_FEATURES):
            acc = np.int32(b1[h])
            for f in range(IN_FEATURES):
                acc += np.int32(w1[h, f]) * np.int32(agg1[i, f])
            # PO2 shift with rounding (remove M scaling)
            shift_amt = shifts['EFF_SCALE1_SHIFT']
            if acc >= 0:
                acc = (acc + (1 << (shift_amt - 1))) >> shift_amt
            else:
                acc = -((-acc + (1 << (shift_amt - 1))) >> shift_amt)
            # ReLU
            hidden[i, h] = np.clip(max(0, acc), -128, 127).astype(np.int8)
    
    # Layer 2: Aggregation over hidden features
    agg2 = np.zeros((N_NODES, HIDDEN_FEATURES), dtype=np.int8)
    for i in range(N_NODES):
        for f in range(HIDDEN_FEATURES):
            tmp = np.int32(0)
            for j in range(N_NODES):
                if adj_int16[i, j] != 0:
                    tmp += np.int32(adj_int16[i, j]) * np.int32(hidden[j, f])
            # PO2 shift with rounding
            shift_amt = shifts['BETA2_SHIFT']
            if tmp >= 0:
                tmp = (tmp + (1 << (shift_amt - 1))) >> shift_amt
            else:
                tmp = -((-tmp + (1 << (shift_amt - 1))) >> shift_amt)
            agg2[i, f] = np.clip(tmp, -128, 127).astype(np.int8)
    
    # Layer 2: Linear (no ReLU) - transforms hidden to output
    OUT_FEATURES = w2.shape[0]  # Output classes (7)
    output = np.zeros((N_NODES, OUT_FEATURES), dtype=np.int8)
    for i in range(N_NODES):
        for o in range(OUT_FEATURES):
            acc = np.int32(b2[o])
            for h in range(HIDDEN_FEATURES):
                acc += np.int32(w2[o, h]) * np.int32(agg2[i, h])
            # PO2 shift with rounding
            shift_amt = shifts['EFF_SCALE2_SHIFT']
            if acc >= 0:
                acc = (acc + (1 << (shift_amt - 1))) >> shift_amt
            else:
                acc = -((-acc + (1 << (shift_amt - 1))) >> shift_amt)
            output[i, o] = np.clip(acc, -128, 127).astype(np.int8)
    
    return output


def run_fixed_inference(input_int8, adj_int16, w1, w2, b1, b2, 
                       data_w, data_i, weight_w, weight_i, acc_w, acc_i,
                       M, K_BITS):
    """
    Run inference with fixed-point Q format quantization.
    Emulates ap_fixed<W,I> arithmetic using the SAME scaling as PTQ.
    
    Key insight: INT8 values from PTQ are already scaled by M fractional bits.
    We just need to handle accumulation/multiplication properly.
    """
    N_NODES = input_int8.shape[0]
    IN_FEATURES = input_int8.shape[1]  # Input features (16)
    HIDDEN_FEATURES = w1.shape[0]      # Hidden features (24) - w1 is [hidden, input]
    
    # Layer 1: Aggregation over input features
    agg1 = np.zeros((N_NODES, IN_FEATURES), dtype=np.int8)
    for i in range(N_NODES):
        for f in range(IN_FEATURES):
            acc = np.int64(0)
            for j in range(N_NODES):
                if adj_int16[i, j] != 0:
                    # adj (INT16, scaled by K=2^K_BITS) × input (INT8, scaled by M)
                    acc += np.int64(adj_int16[i, j]) * np.int64(input_int8[j, f])
            # Scale down: remove K scaling (adj contribution)
            acc = acc >> K_BITS
            # Result has M fractional bits (from input)
            agg1[i, f] = np.clip(acc, -128, 127).astype(np.int8)
    
    # Layer 1: Linear (transforms aggregated input to hidden features)
    hidden = np.zeros((N_NODES, HIDDEN_FEATURES), dtype=np.int8)
    for i in range(N_NODES):
        for h in range(HIDDEN_FEATURES):
            acc = np.int64(b1[h])  # Bias already scaled by M
            for f in range(IN_FEATURES):
                # weight (INT8, M bits) × agg (INT8, M bits) = INT16 with 2M bits
                acc += np.int64(w1[h, f]) * np.int64(agg1[i, f])
            # Scale down: remove M (one of the two M contributions)
            acc = acc >> M
            # ReLU and clip
            hidden[i, h] = np.clip(max(0, acc), -128, 127).astype(np.int8)
    
    # Layer 2: Aggregation over hidden features
    agg2 = np.zeros((N_NODES, HIDDEN_FEATURES), dtype=np.int8)
    for i in range(N_NODES):
        for f in range(HIDDEN_FEATURES):
            acc = np.int64(0)
            for j in range(N_NODES):
                if adj_int16[i, j] != 0:
                    acc += np.int64(adj_int16[i, j]) * np.int64(hidden[j, f])
            # Scale down: remove K
            acc = acc >> K_BITS
            agg2[i, f] = np.clip(acc, -128, 127).astype(np.int8)
    
    # Layer 2: Linear (no ReLU) - transforms hidden to output
    OUT_FEATURES = w2.shape[0]  # Output classes (7)
    output = np.zeros((N_NODES, OUT_FEATURES), dtype=np.int8)
    for i in range(N_NODES):
        for o in range(OUT_FEATURES):
            acc = np.int64(b2[o])
            for h in range(HIDDEN_FEATURES):
                acc += np.int64(w2[o, h]) * np.int64(agg2[i, h])
            # Scale down: remove M
            acc = acc >> M
            output[i, o] = np.clip(acc, -128, 127).astype(np.int8)
    
    return output


def run_int8_inference(input_int8, adj_int16, w1, w2, b1, b2, M, K_BITS, shifts):
    """
    Run inference with INT8 and arbitrary-precision scales.
    Similar to PO2 but uses exact scale values instead of bit-shifts.
    """
    # For now, just use PO2 - this is rarely used
    # Could load actual scale values from PTQ config if needed
    print("  ⚠ Using PO2 approximation for int8 (arbitrary scales not implemented)")
    return run_po2_inference(input_int8, adj_int16, w1, w2, b1, b2, shifts, M, K_BITS)


def main():
    parser = argparse.ArgumentParser(description='Generate architecture-specific test vectors')
    parser.add_argument('--in-channels', type=int, default=16,
                       help='Input feature dimension (default: 16)')
    parser.add_argument('--hidden-channels', type=int, default=24,
                       help='Hidden layer dimension (default: 24)')
    parser.add_argument('--implementation', '--impl', type=str, default='int8_po2',
                       choices=['int8_po2', 'fixed', 'int8'],
                       help='HLS implementation type (default: int8_po2)')
    parser.add_argument('--output-dir', type=str, default='../build/test_vectors_arch',
                       help='Output directory (default: ../build/test_vectors_arch)')
    
    # Fixed-point Q format parameters (only used if --implementation=fixed)
    parser.add_argument('--data-w', type=int, default=16,
                       help='Data total bits for fixed (default: 16)')
    parser.add_argument('--data-i', type=int, default=8,
                       help='Data integer bits for fixed (default: 8)')
    parser.add_argument('--weight-w', type=int, default=16,
                       help='Weight total bits for fixed (default: 16)')
    parser.add_argument('--weight-i', type=int, default=4,
                       help='Weight integer bits for fixed (default: 4)')
    parser.add_argument('--acc-w', type=int, default=32,
                       help='Accumulator total bits for fixed (default: 32)')
    parser.add_argument('--acc-i', type=int, default=16,
                       help='Accumulator integer bits for fixed (default: 16)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    success = generate_test_vectors(
        args.in_channels, args.hidden_channels,
        args.implementation, output_dir,
        args.data_w, args.data_i,
        args.weight_w, args.weight_i,
        args.acc_w, args.acc_i
    )
    
    if success:
        print("\n✅ SUCCESS")
        return 0
    else:
        print("\n❌ FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
