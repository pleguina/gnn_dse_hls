"""
Evaluate Integer-Only PTQ Accuracy

Compares the integer-only PTQ implementation against:
1. Float model (ground truth)
2. Float PTQ (dequant->float ops->quant)

This helps assess the accuracy impact of using fixed-point arithmetic
vs floating-point dequantization in the PTQ implementation.
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_base import ReducedGraphSAGE
from subgraph_extraction import extract_fixed_subgraph
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from quantization_ptq import quantize_tensor


def load_int8_output():
    """Load integer-only PTQ output"""
    output_path = Path("../build/test_vectors_ptq_int8/network_output_int8_reference.txt")
    if not output_path.exists():
        print(f"ERROR: Integer-only output not found at {output_path}")
        print("Please run: cd tests && python3 integer_ptq_emulator.py")
        return None
    
    return np.loadtxt(output_path, dtype=np.int8)


def load_float_ptq_output():
    """Load float PTQ output"""
    output_path = Path("../build/test_vectors_ptq_float/network_output_reference.txt")
    if not output_path.exists():
        print(f"ERROR: Float PTQ output not found at {output_path}")
        print("Please run: cd tests && python3 generate_test_vectors_ptq_float.py")
        return None
    
    return np.loadtxt(output_path, dtype=np.int8)


def run_float_model(model, subgraph_data):
    """Run float model and quantize output"""
    model.eval()
    
    # Get features
    features = subgraph_data['x']
    
    # Process through projection if exists
    if hasattr(model, 'projection') and model.use_projection:
        features_tensor = torch.from_numpy(features).float()
        with torch.no_grad():
            features_tensor = model.projection(features_tensor)
            features_tensor = torch.relu(features_tensor)
        features = features_tensor.numpy()
    
    # Run through network
    with torch.no_grad():
        edge_index = torch.from_numpy(subgraph_data['edge_index']).long()
        x = torch.from_numpy(features).float()
        
        # Layer 1
        out = model.conv1(x, edge_index)
        out = torch.relu(out)
        
        # Layer 2
        out = model.conv2(out, edge_index)
        
        # Quantize output
        out_quant, scale_out, _ = quantize_tensor(out, num_bits=8)
        
    return out_quant.numpy(), out.numpy()


def compute_accuracy_metrics(float_output, int8_output, output_name="INT8-PTQ"):
    """Compute accuracy metrics between float and int8 outputs"""
    
    # Compute differences
    diff = int8_output.astype(np.int16) - float_output.astype(np.int16)
    
    # Metrics
    exact_match = np.sum(diff == 0)
    total = diff.size
    max_error = np.abs(diff).max()
    mean_error = np.abs(diff).mean()
    
    # LSB error distribution
    error_dist = {}
    for err in range(int(max_error) + 1):
        count = np.sum(np.abs(diff) == err)
        if count > 0:
            error_dist[err] = count
    
    print(f"\n{output_name} vs Float PTQ:")
    print(f"  Exact matches: {exact_match}/{total} ({100*exact_match/total:.1f}%)")
    print(f"  Max error: {max_error} LSB")
    print(f"  Mean error: {mean_error:.2f} LSB")
    
    print(f"\n  Error distribution:")
    for err in sorted(error_dist.keys())[:10]:  # Show first 10
        pct = 100 * error_dist[err] / total
        print(f"    {err} LSB: {error_dist[err]:4d} ({pct:5.1f}%)")
    if len(error_dist) > 10:
        print(f"    ... ({len(error_dist) - 10} more error levels)")
    
    return {
        'exact_match_pct': 100 * exact_match / total,
        'max_error': int(max_error),
        'mean_error': float(mean_error)
    }


def dequantize_and_compare(int8_output, scale):
    """Dequantize INT8 output and show float values"""
    float_output = int8_output.astype(np.float32) * scale
    
    print(f"\nDequantized output (first node, scale={scale}):")
    print(f"  INT8:  {int8_output[0]}")
    print(f"  Float: {float_output[0]}")
    
    return float_output


if __name__ == '__main__':
    print("="*80)
    print("INTEGER-ONLY PTQ ACCURACY EVALUATION")
    print("="*80)
    
    # Load dataset and subgraph
    print("\nLoading dataset and subgraph...")
    dataset = Planetoid(root='../data', name='Cora', transform=NormalizeFeatures())
    data = dataset[0]
    subgraph_data = extract_fixed_subgraph(data, num_nodes=8, center_node=0, num_hops=2)
    
    # Load trained model
    print("Loading trained float model...")
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
    print(f"✓ Loaded model with test accuracy: {checkpoint.get('test_acc', 'N/A'):.4f}")
    
    # Run float model
    print("\nRunning float model...")
    float_output_quant, float_output_raw = run_float_model(model, subgraph_data)
    
    # Get scale for dequantization
    scale_out = 0.1  # From test vectors
    
    # Load PTQ outputs
    print("\nLoading PTQ outputs...")
    float_ptq_output = load_float_ptq_output()
    int8_ptq_output = load_int8_output()
    
    if float_ptq_output is None or int8_ptq_output is None:
        sys.exit(1)
    
    print(f"✓ Float PTQ output shape: {float_ptq_output.shape}")
    print(f"✓ INT8 PTQ output shape: {int8_ptq_output.shape}")
    
    # Compare outputs
    print("\n" + "="*80)
    print("ACCURACY COMPARISON")
    print("="*80)
    
    # 1. INT8-PTQ vs Float PTQ
    metrics_int8 = compute_accuracy_metrics(float_ptq_output, int8_ptq_output, "INT8-PTQ (fixed-point)")
    
    # 2. Show sample outputs
    print("\n" + "="*80)
    print("SAMPLE OUTPUTS (Node 0)")
    print("="*80)
    
    print(f"\nFloat model (raw logits):")
    print(f"  {float_output_raw[0]}")
    
    print(f"\nFloat model (quantized INT8):")
    print(f"  {float_output_quant[0]}")
    
    print(f"\nFloat PTQ (dequant→float ops→quant):")
    print(f"  {float_ptq_output[0]}")
    
    print(f"\nINT8 PTQ (fixed-point, M=20, K=4096):")
    print(f"  {int8_ptq_output[0]}")
    
    # Dequantize for float comparison
    int8_dequant = dequantize_and_compare(int8_ptq_output, scale_out)
    float_ptq_dequant = dequantize_and_compare(float_ptq_output, scale_out)
    
    # Compare dequantized outputs
    print("\n" + "="*80)
    print("DEQUANTIZED COMPARISON (in original logit scale)")
    print("="*80)
    
    float_diff = np.abs(int8_dequant - float_ptq_dequant)
    print(f"\nMax difference: {float_diff.max():.6f}")
    print(f"Mean difference: {float_diff.mean():.6f}")
    print(f"RMS difference: {np.sqrt((float_diff**2).mean()):.6f}")
    
    # Final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nInteger-Only PTQ Accuracy:")
    print(f"  Exact match with Float PTQ: {metrics_int8['exact_match_pct']:.1f}%")
    print(f"  Max quantization error: {metrics_int8['max_error']} LSB")
    print(f"  Mean quantization error: {metrics_int8['mean_error']:.2f} LSB")
    
    print(f"\nInterpretation:")
    if metrics_int8['exact_match_pct'] > 90:
        print(f"  ✓ EXCELLENT: >90% exact match - negligible accuracy impact")
    elif metrics_int8['exact_match_pct'] > 70:
        print(f"  ✓ GOOD: >70% exact match - acceptable for HLS")
    elif metrics_int8['exact_match_pct'] > 50:
        print(f"  ⚠ FAIR: >50% exact match - may need tuning")
    else:
        print(f"  ✗ POOR: <50% exact match - fixed-point parameters may need adjustment")
    
    print(f"\nThe differences come from:")
    print(f"  1. Float PTQ uses: dequantize → float aggregate → quantize")
    print(f"  2. INT8 PTQ uses: INT16 fixed-point aggregate (K={4096})")
    print(f"  3. Fixed-point scale approximation (M={20} fractional bits)")
    
    print(f"\nFor HLS implementation:")
    print(f"  - Use INT8 PTQ for pure integer datapath")
    print(f"  - No floating-point operations required")
    print(f"  - Significant resource savings (DSP, area)")
    print(f"  - Accuracy trade-off: ~{metrics_int8['mean_error']:.1f} LSB average error")
    
    print("\n" + "="*80)
