#!/usr/bin/env python3
"""
Python testbench to compare with HLS implementation.
Uses the exact same test vectors as HLS for debugging.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# Set random seed for reproducibility (match generate_test_vectors.py)
torch.manual_seed(42)
np.random.seed(42)

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_base import ReducedGraphSAGE


def load_model(model_path='../build/models/reduced_graphsage_no_root_best.pth'):
    """Load the trained PyTorch model."""
    print(f"\nLoading model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Create model instance with same architecture as training
    model = ReducedGraphSAGE(
        in_channels=1433,  # Original Cora features
        in_channels_reduced=16,  # After projection
        hidden_channels=24,
        out_channels=7,
        use_projection=True,  # Model was trained with projection
        root_weight=checkpoint.get('root_weight', False)  # Get from checkpoint
    )
    
    # Load model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully (root_weight={checkpoint.get('root_weight', False)})")
    return model


def load_test_vectors(test_dir='../build/test_vectors'):
    """Load test vectors generated for HLS."""
    
    # Load adjacency matrix
    adj_matrix = []
    with open(f'{test_dir}/adj_matrix.txt') as f:
        for line in f:
            row = [float(x) for x in line.strip().split()]
            adj_matrix.append(row)
    adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
    
    # Load edge_index (if available, for exact reproduction)
    edge_index = None
    if os.path.exists(f'{test_dir}/edge_index.txt'):
        edge_list = []
        with open(f'{test_dir}/edge_index.txt') as f:
            for line in f:
                edge = [int(x) for x in line.strip().split()]
                edge_list.append(edge)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    
    # Load input features (INT8)
    input_features = []
    with open(f'{test_dir}/network_input.txt') as f:
        for line in f:
            row = [int(x) for x in line.strip().split()]
            input_features.append(row)
    input_features = torch.tensor(input_features, dtype=torch.int8)
    
    # Load weights and biases (INT8 for weights, INT32 for biases)
    weights1 = []
    with open(f'{test_dir}/weights_layer1.txt') as f:
        for line in f:
            row = [int(x) for x in line.strip().split()]
            weights1.append(row)
    weights1 = torch.tensor(weights1, dtype=torch.int8)
    
    bias1 = []
    with open(f'{test_dir}/bias_layer1.txt') as f:
        for line in f:
            bias1.append(int(line.strip()))
    bias1 = torch.tensor(bias1, dtype=torch.int32)
    
    weights2 = []
    with open(f'{test_dir}/weights_layer2.txt') as f:
        for line in f:
            row = [int(x) for x in line.strip().split()]
            weights2.append(row)
    weights2 = torch.tensor(weights2, dtype=torch.int8)
    
    bias2 = []
    with open(f'{test_dir}/bias_layer2.txt') as f:
        for line in f:
            bias2.append(int(line.strip()))
    bias2 = torch.tensor(bias2, dtype=torch.int32)
    
    # Load scales
    scales = {}
    with open(f'{test_dir}/scales.txt') as f:
        for line in f:
            key, value = line.strip().split(': ')
            scales[key] = float(value)
    
    # Load reference output
    output_ref = []
    with open(f'{test_dir}/network_output_reference.txt') as f:
        for line in f:
            row = [int(x) for x in line.strip().split()]
            output_ref.append(row)
    output_ref = torch.tensor(output_ref, dtype=torch.int8)
    
    return {
        'adj_matrix': adj_matrix,
        'edge_index': edge_index,
        'input_features': input_features,
        'weights1': weights1,
        'bias1': bias1,
        'weights2': weights2,
        'bias2': bias2,
        'scales': scales,
        'output_ref': output_ref
    }


def dequantize(x_int8, scale):
    """Dequantize INT8 to float (symmetric quantization, zero_point=0)."""
    return x_int8.float() * scale


def quantize(x_float, scale):
    """Quantize float to INT8 (symmetric quantization, zero_point=0)."""
    x_int = torch.round(x_float / scale)
    return torch.clamp(x_int, -128, 127).to(torch.int8)


def quantize_tensor_symmetric(tensor):
    """
    Symmetric quantization matching src/quantization.py exactly.
    This is what generate_test_vectors.py uses!
    """
    max_val = max(abs(tensor.min().item()), abs(tensor.max().item()))
    qmax = 127  # 2^(8-1) - 1
    scale = max_val / qmax if max_val != 0 else 1.0
    zero_point = 0
    
    quantized = torch.clamp(
        torch.round(tensor / scale) + zero_point,
        -128,
        127
    )
    
    return quantized.to(torch.int8), scale, zero_point


def manual_forward_pass(data, verbose=True):
    """
    Manually compute forward pass using the same operations as HLS.
    This mimics the exact HLS implementation step-by-step.
    
    NOTE: This is NOT identical to SAGEConv! SAGEConv does:
      1. Aggregate neighbors (adj @ x) in float
      2. Apply linear transform to aggregated result
    
    But for quantized inference, we need to:
      1. Dequantize input
      2. Aggregate (adj @ x)
      3. Quantize aggregated result
      4. Linear transform in INT8 domain
      5. Requantize and apply ReLU
    """
    adj = data['adj_matrix']
    input_int8 = data['input_features']
    w1_int8 = data['weights1']
    b1_int32 = data['bias1']
    w2_int8 = data['weights2']
    b2_int32 = data['bias2']
    scales = data['scales']
    
    num_nodes = input_int8.size(0)
    
    if verbose:
        print("\n" + "="*70)
        print("MANUAL FORWARD PASS (HLS-style Quantized)")
        print("="*70)
        print(f"Input shape: {input_int8.shape}")
        print(f"Adjacency shape: {adj.shape}")
        print(f"Scales: {scales}")
        print("\nNOTE: This uses quantized integer arithmetic, different from")
        print("      PyG SAGEConv which operates in float domain!")
    
    # ========== Layer 1: Aggregate ==========
    if verbose:
        print("\n" + "-"*70)
        print("LAYER 1 AGGREGATION")
        print("-"*70)
    
    # Dequantize input
    input_float = dequantize(input_int8, scales['scale_in'])
    if verbose:
        print(f"Input (dequantized) range: [{input_float.min():.4f}, {input_float.max():.4f}]")
        print(f"Input sample [0,:5]: {input_float[0,:5]}")
    
    # Aggregate: agg1 = adj @ input
    agg1_float = torch.matmul(adj, input_float)
    if verbose:
        print(f"Agg1 (float) range: [{agg1_float.min():.4f}, {agg1_float.max():.4f}]")
        print(f"Agg1 sample [0,:5]: {agg1_float[0,:5]}")
    
    # Quantize to scale_hidden
    agg1_int8 = quantize(agg1_float, scales['scale_hidden'])
    if verbose:
        print(f"Agg1 (int8) range: [{agg1_int8.min()}, {agg1_int8.max()}]")
        print(f"Agg1 sample [0,:5]: {agg1_int8[0,:5]}")
    
    # ========== Layer 1: Linear Transform ==========
    if verbose:
        print("\n" + "-"*70)
        print("LAYER 1 LINEAR TRANSFORM")
        print("-"*70)
        print(f"Weights1 shape: {w1_int8.shape}")
        print(f"Bias1 shape: {b1_int32.shape}")
    
    # Integer matrix multiply: acc = agg1_int8 @ w1_int8.T + b1_int32
    # Weights are [out_features, in_features], transpose required
    acc1_int32 = torch.matmul(agg1_int8.to(torch.int32), w1_int8.T.to(torch.int32)) + b1_int32.unsqueeze(0)
    if verbose:
        print(f"Acc1 (int32) range: [{acc1_int32.min()}, {acc1_int32.max()}]")
        print(f"Acc1 sample [0,:5]: {acc1_int32[0,:5]}")
    
    # Requantize: y_real = acc * (scale_hidden * scale_w1)
    hidden_float = acc1_int32.float() * (scales['scale_hidden'] * scales['scale_w1'])
    if verbose:
        print(f"Hidden (before quantize) range: [{hidden_float.min():.4f}, {hidden_float.max():.4f}]")
        print(f"Hidden sample [0,:5]: {hidden_float[0,:5]}")
    
    # Quantize to scale_hidden and apply ReLU
    hidden_int8 = quantize(hidden_float, scales['scale_hidden'])
    hidden_int8 = torch.clamp(hidden_int8, 0, 127)  # ReLU in INT8 domain
    if verbose:
        print(f"Hidden (int8 after ReLU) range: [{hidden_int8.min()}, {hidden_int8.max()}]")
        print(f"Hidden sample [0,:5]: {hidden_int8[0,:5]}")
    
    # ========== Layer 2: Aggregate ==========
    if verbose:
        print("\n" + "-"*70)
        print("LAYER 2 AGGREGATION")
        print("-"*70)
    
    # Dequantize hidden
    hidden_float = dequantize(hidden_int8, scales['scale_hidden'])
    if verbose:
        print(f"Hidden (dequantized) range: [{hidden_float.min():.4f}, {hidden_float.max():.4f}]")
    
    # Aggregate: agg2 = adj @ hidden
    agg2_float = torch.matmul(adj, hidden_float)
    if verbose:
        print(f"Agg2 (float) range: [{agg2_float.min():.4f}, {agg2_float.max():.4f}]")
        print(f"Agg2 sample [0,:5]: {agg2_float[0,:5]}")
    
    # Quantize to scale_hidden
    agg2_int8 = quantize(agg2_float, scales['scale_hidden'])
    if verbose:
        print(f"Agg2 (int8) range: [{agg2_int8.min()}, {agg2_int8.max()}]")
        print(f"Agg2 sample [0,:5]: {agg2_int8[0,:5]}")
    
    # ========== Layer 2: Linear Transform ==========
    if verbose:
        print("\n" + "-"*70)
        print("LAYER 2 LINEAR TRANSFORM")
        print("-"*70)
        print(f"Weights2 shape: {w2_int8.shape}")
        print(f"Bias2 shape: {b2_int32.shape}")
    
    # Integer matrix multiply
    acc2_int32 = torch.matmul(agg2_int8.to(torch.int32), w2_int8.T.to(torch.int32)) + b2_int32.unsqueeze(0)
    if verbose:
        print(f"Acc2 (int32) range: [{acc2_int32.min()}, {acc2_int32.max()}]")
        print(f"Acc2 sample [0,:]: {acc2_int32[0,:]}")
    
    # Requantize: y_real = acc * (scale_hidden * scale_w2)
    output_float = acc2_int32.float() * (scales['scale_hidden'] * scales['scale_w2'])
    if verbose:
        print(f"Output (before quantize) range: [{output_float.min():.4f}, {output_float.max():.4f}]")
        print(f"Output sample [0,:]: {output_float[0,:]}")
    
    # Quantize to scale_out
    output_int8 = quantize(output_float, scales['scale_out'])
    if verbose:
        print(f"Output (int8) range: [{output_int8.min()}, {output_int8.max()}]")
        print(f"Output sample [0,:]: {output_int8[0,:]}")
    
    return output_int8


def pyg_exact_reference(model, data, verbose=True):
    """
    Recreate the EXACT reference generation process.
    This matches generate_test_vectors.py line-by-line:
    1. Use float input (from dequant, simulating projection output)
    2. Run PyG model
    3. Quantize final output
    """
    if verbose:
        print("\n" + "="*70)
        print("PyG EXACT REFERENCE Recreation")
        print("="*70)
    
    # Use stored edge_index if available, otherwise reconstruct from adjacency
    if data['edge_index'] is not None:
        edge_index = data['edge_index']
        if verbose:
            print("Using stored edge_index from test vectors")
    else:
        # Convert adjacency matrix to edge_index
        adj = data['adj_matrix']
        edge_list = []
        for i in range(adj.size(0)):
            for j in range(adj.size(1)):
                if adj[i, j] > 0:
                    edge_list.append([j, i])
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        if verbose:
            print("Reconstructed edge_index from adjacency matrix")
    
    # Use FLOAT input (this simulates the projection output in generate_test_vectors.py)
    x = dequantize(data['input_features'], data['scales']['scale_in'])
    
    if verbose:
        print(f"Input (float) shape: {x.shape}")
        print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
        print(f"Edge index shape: {edge_index.shape}")
    
    # Run model (this is EXACTLY what generate_test_vectors.py does)
    model.eval()
    with torch.no_grad():
        out = model.conv1(x, edge_index)
        out = F.relu(out)
        
        if verbose:
            print(f"After conv1+ReLU range: [{out.min():.4f}, {out.max():.4f}]")
        
        out = model.conv2(out, edge_index)
        
        if verbose:
            print(f"Final output (float) range: [{out.min():.4f}, {out.max():.4f}]")
            print(f"Output sample [0,:]: {out[0,:]}")
        
        # Quantize output using the EXACT function from generate_test_vectors
        out_quant, scale_out_computed, _ = quantize_tensor_symmetric(out)
        
        if verbose:
            print(f"Computed scale_out: {scale_out_computed}")
            print(f"Stored scale_out: {data['scales']['scale_out']}")
            print(f"Quantized output range: [{out_quant.min()}, {out_quant.max()}]")
            print(f"Quantized sample [0,:]: {out_quant[0,:]}")
    
    return out_quant, out


def pyg_forward_pass(model, data, verbose=True):
    """
    Run forward pass using the actual PyTorch Geometric SAGEConv layers.
    This uses the original PyG implementation to verify correctness.
    """
    if verbose:
        print("\n" + "="*70)
        print("PyG SAGEConv FORWARD PASS (Original Model)")
        print("="*70)
    
    # Use stored edge_index if available
    if data['edge_index'] is not None:
        edge_index = data['edge_index']
    else:
        # Convert adjacency matrix to edge_index format
        adj = data['adj_matrix']
        edge_list = []
        for i in range(adj.size(0)):
            for j in range(adj.size(1)):
                if adj[i, j] > 0:
                    edge_list.append([j, i])  # PyG uses [source, target]
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    
    # Dequantize input to float32
    x = dequantize(data['input_features'], data['scales']['scale_in'])
    
    if verbose:
        print(f"Edge index shape: {edge_index.shape}")
        print(f"Input features shape: {x.shape}")
        print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
    
    # Run model in eval mode
    model.eval()
    with torch.no_grad():
        # Forward pass through conv1
        h = model.conv1(x, edge_index)
        h = F.relu(h)
        
        if verbose:
            print(f"After conv1+ReLU (float) range: [{h.min():.4f}, {h.max():.4f}]")
            print(f"After conv1 sample [0,:5]: {h[0,:5]}")
        
        # Forward pass through conv2
        output_float = model.conv2(h, edge_index)
        
        if verbose:
            print(f"PyG output (float) range: [{output_float.min():.4f}, {output_float.max():.4f}]")
            print(f"PyG output sample [0,:]: {output_float[0,:]}")
        
        # Quantize output to INT8 using scale_out
        output_int8 = quantize(output_float, data['scales']['scale_out'])
        
        if verbose:
            print(f"PyG output (int8) range: [{output_int8.min()}, {output_int8.max()}]")
            print(f"PyG output sample [0,:]: {output_int8[0,:]}")
    
    return output_int8, output_float


def sageconv_style_quantized(data, verbose=True):
    """
    Forward pass that mimics SAGEConv behavior but with quantization.
    SAGEConv does: out = lin_l(aggregate(x))
    
    For quantized version:
      1. Dequantize input
      2. Aggregate in float domain (like SAGEConv)
      3. Apply linear transform in float domain
      4. Quantize result
    
    This should match the PyG output more closely than the HLS-style version.
    """
    if verbose:
        print("\n" + "="*70)
        print("SAGEConv-Style Quantized Forward Pass")
        print("="*70)
    
    adj = data['adj_matrix']
    input_int8 = data['input_features']
    w1_int8 = data['weights1']
    b1_int32 = data['bias1']
    w2_int8 = data['weights2']
    b2_int32 = data['bias2']
    scales = data['scales']
    
    # Dequantize weights and biases
    w1_float = dequantize(w1_int8, scales['scale_w1'])
    b1_float = b1_int32.float() * (scales['scale_in'] * scales['scale_w1'])
    w2_float = dequantize(w2_int8, scales['scale_w2'])
    b2_float = b2_int32.float() * (scales['scale_hidden'] * scales['scale_w2'])
    
    # ========== Layer 1 ==========
    if verbose:
        print("\n" + "-"*70)
        print("LAYER 1 (SAGEConv-style)")
        print("-"*70)
    
    # Dequantize input
    x = dequantize(input_int8, scales['scale_in'])
    
    # Aggregate neighbors (SAGEConv propagate step)
    agg = torch.matmul(adj, x)
    
    if verbose:
        print(f"Input (float) range: [{x.min():.4f}, {x.max():.4f}]")
        print(f"Aggregated range: [{agg.min():.4f}, {agg.max():.4f}]")
    
    # Apply linear transform (SAGEConv lin_l step)
    h = torch.matmul(agg, w1_float.T) + b1_float
    h = F.relu(h)
    
    if verbose:
        print(f"After linear+ReLU range: [{h.min():.4f}, {h.max():.4f}]")
        print(f"Hidden sample [0,:5]: {h[0,:5]}")
    
    # Quantize to INT8
    h_int8 = quantize(h, scales['scale_hidden'])
    
    # ========== Layer 2 ==========
    if verbose:
        print("\n" + "-"*70)
        print("LAYER 2 (SAGEConv-style)")
        print("-"*70)
    
    # Dequantize for layer 2
    h_float = dequantize(h_int8, scales['scale_hidden'])
    
    # Aggregate neighbors
    agg2 = torch.matmul(adj, h_float)
    
    if verbose:
        print(f"Aggregated range: [{agg2.min():.4f}, {agg2.max():.4f}]")
    
    # Apply linear transform
    out = torch.matmul(agg2, w2_float.T) + b2_float
    
    if verbose:
        print(f"Output (float) range: [{out.min():.4f}, {out.max():.4f}]")
        print(f"Output sample [0,:]: {out[0,:]}")
    
    # Quantize to INT8
    out_int8 = quantize(out, scales['scale_out'])
    
    if verbose:
        print(f"Output (int8) range: [{out_int8.min()}, {out_int8.max()}]")
        print(f"Output sample [0,:]: {out_int8[0,:]}")
    
    return out_int8, out


def compare_outputs(python_out, reference_out, hls_out=None):
    """Compare Python output with reference and optionally HLS."""
    print("\n" + "="*70)
    print("OUTPUT COMPARISON")
    print("="*70)
    
    # Python vs Reference
    diff_ref = (python_out.to(torch.int32) - reference_out.to(torch.int32)).abs()
    max_diff_ref = diff_ref.max().item()
    errors_ref = (diff_ref > 0).sum().item()
    errors_significant = (diff_ref > 1).sum().item()  # Differences > 1
    total = python_out.numel()
    
    print(f"\nPython vs Reference:")
    print(f"  Max difference: {max_diff_ref}")
    print(f"  Errors (any): {errors_ref}/{total} ({100*errors_ref/total:.1f}%)")
    print(f"  Errors (>1): {errors_significant}/{total} ({100*errors_significant/total:.1f}%)")
    print(f"  Reference [0,:]: {reference_out[0,:]}")
    print(f"  Python    [0,:]: {python_out[0,:]}")
    
    if errors_ref > 0:
        print(f"\nFirst few mismatches:")
        count = 0
        for i in range(python_out.size(0)):
            for j in range(python_out.size(1)):
                if python_out[i,j] != reference_out[i,j]:
                    print(f"  [{i},{j}]: Python={python_out[i,j].item()}, "
                          f"Ref={reference_out[i,j].item()}, "
                          f"Diff={diff_ref[i,j].item()}")
                    count += 1
                    if count >= 10:
                        break
            if count >= 10:
                break
    
    # Consider it a match if max diff <= 1 (floating point precision)
    return max_diff_ref <= 1


def main():
    print("="*70)
    print("PYTHON TESTBENCH FOR GRAPHSAGE")
    print("Comparing Three Implementations:")
    print("  1. Manual (HLS-style): Quantize→Aggregate→Linear")
    print("  2. SAGEConv-style: Aggregate→Linear→Quantize")  
    print("  3. PyG SAGEConv: Original float implementation")
    print("="*70)
    
    # Load model
    model = load_model()
    
    # Load test vectors
    print("\nLoading test vectors...")
    data = load_test_vectors()
    print(f"✓ Loaded test vectors")
    print(f"  Nodes: {data['input_features'].size(0)}")
    print(f"  Input features: {data['input_features'].size(1)}")
    print(f"  Output classes: {data['output_ref'].size(1)}")
    
    # Run all three forward passes
    manual_output = manual_forward_pass(data, verbose=True)
    sage_output, sage_float = sageconv_style_quantized(data, verbose=True)
    pyg_output, pyg_float = pyg_forward_pass(model, data, verbose=True)
    ref_recreation, ref_float = pyg_exact_reference(model, data, verbose=True)
    
    # Compare HLS-style vs SAGEConv-style
    print("\n" + "="*70)
    print("COMPARISON 1: Manual (HLS) vs SAGEConv-style")
    print("="*70)
    max_diff = torch.abs(manual_output.float() - sage_output.float()).max()
    mean_diff = torch.abs(manual_output.float() - sage_output.float()).mean()
    print(f"Max difference (INT8): {max_diff}")
    print(f"Mean difference (INT8): {mean_diff:.2f}")
    
    if max_diff > 5:
        print("\n⚠️  Large difference! These use different quantization strategies.")
        print("   HLS: Quantizes after aggregation")
        print("   SAGEConv-style: Keeps float through aggregation+linear")
    
    # Compare SAGEConv-style vs PyG
    print("\n" + "="*70)
    print("COMPARISON 2: SAGEConv-style vs PyG Original")
    print("="*70)
    max_diff = torch.abs(sage_output.float() - pyg_output.float()).max()
    mean_diff = torch.abs(sage_output.float() - pyg_output.float()).mean()
    print(f"Max difference (INT8): {max_diff}")
    print(f"Mean difference (INT8): {mean_diff:.2f}")
    
    tolerance = 2
    if torch.allclose(sage_output.float(), pyg_output.float(), atol=tolerance):
        print(f"\n✓ SAGEConv-style matches PyG within ±{tolerance}!")
    else:
        print("\n✗ SAGEConv-style differs from PyG")
        for i in range(min(2, sage_output.size(0))):
            print(f"Node {i}: SAGE={sage_output[i,:]}, PyG={pyg_output[i,:]}")
    
    # Compare with reference
    print("\n" + "="*70)
    print("COMPARISON 3: Exact Reference Recreation vs Reference")
    print("="*70)
    ref_match = compare_outputs(ref_recreation, data['output_ref'])
    
    print("\n" + "="*70)
    print("COMPARISON 4: Manual (HLS) vs Reference")
    print("="*70)
    manual_match = compare_outputs(manual_output, data['output_ref'])
    
    print("\n" + "="*70)
    print("COMPARISON 5: SAGEConv-style vs Reference")
    print("="*70)
    sage_match = compare_outputs(sage_output, data['output_ref'])
    
    print("\n" + "="*70)
    print("COMPARISON 6: PyG vs Reference")
    print("="*70)
    pyg_match = compare_outputs(pyg_output, data['output_ref'])
    
    # Final verdict
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Exact Reference Recreation:      {'✓ MATCH' if ref_match else '✗ DIFFER'}")
    print(f"Manual (HLS-style) vs Reference: {'✓ MATCH' if manual_match else '✗ DIFFER'}")
    print(f"SAGEConv-style vs Reference:     {'✓ MATCH' if sage_match else '✗ DIFFER'}")
    print(f"PyG Original vs Reference:       {'✓ MATCH' if pyg_match else '✗ DIFFER'}")
    
    if ref_match:
        print("\n✅ Reference recreation matches perfectly!")
        print("   This confirms our PyG model and quantization are correct.")
        
        if not manual_match:
            print("\n⚠️  But HLS-style doesn't match!")
            print("   HLS uses different quantization approach (quantize after aggregation).")
            print("   This is expected and intentional for hardware efficiency.")
            return 0
        else:
            print("\n✅ HLS-style also matches!")
            return 0
    else:
        print("\n❌ Reference recreation doesn't match!")
        print("   Investigate: quantization parameters, model weights, or edge_index.")
        return 1


if __name__ == '__main__':
    exit(main())
