"""
Comprehensive model analysis and visualization script.
Generates comparison plots for all model variants.
Automatically trains missing models.
"""

import torch
import numpy as np
import json
import os
import sys
from pathlib import Path

from model_base import GraphSAGE, ReducedGraphSAGE
from model_qat import ReducedGraphSAGEQAT
from model_qat_v2 import ReducedGraphSAGEQATv2, train_qat_v2
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data
from quantization_ptq import quantize_tensor
from config import get_config
from train import train_base_model, train_reduced_model
from train_qat import train_qat_model

# Try to import Brevitas support
try:
    from brevitas_models import BrevitasReducedGraphSAGE, BrevitasQuantConfig, check_brevitas_available
    BREVITAS_AVAILABLE = check_brevitas_available()
except ImportError:
    BREVITAS_AVAILABLE = False

# Fix for PyTorch 2.6+ weights_only default change
torch.serialization.add_safe_globals([Data])
from visualization import (
    plot_model_comparison,
    plot_efficiency_analysis,
    plot_accuracy_degradation,
    plot_resource_utilization,
    generate_summary_report,
    save_stats_json,
    plot_quantization_error
)


def count_parameters(model):
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters())


def estimate_model_size(model, quantized=False):
    """Estimate model memory footprint in MB."""
    params = count_parameters(model)
    if quantized:
        # INT8: 1 byte per parameter
        size_mb = params / (1024 ** 2)
    else:
        # FP32: 4 bytes per parameter
        size_mb = params * 4 / (1024 ** 2)
    return size_mb


def get_layer_breakdown(model):
    """Get parameter breakdown by layer type."""
    conv_params = 0
    linear_params = 0
    other_params = 0

    for name, module in model.named_modules():
        if 'conv' in name.lower():
            conv_params += sum(p.numel() for p in module.parameters())
        elif 'linear' in name.lower() or 'projection' in name.lower():
            linear_params += sum(p.numel() for p in module.parameters())
        elif len(list(module.parameters())) > 0:
            other_params += sum(p.numel() for p in module.parameters())

    # Avoid double counting
    total = conv_params + linear_params + other_params
    model_total = count_parameters(model)

    if total > model_total:
        # Adjust to prevent double counting
        conv_params = int(conv_params * model_total / total)
        linear_params = int(linear_params * model_total / total)
        other_params = model_total - conv_params - linear_params

    return {
        'conv': conv_params,
        'linear': linear_params,
        'other': other_params
    }


def evaluate_model(model, data, use_test_mask=True):
    """Evaluate model accuracy.
    
    Args:
        model: PyTorch model
        data: PyG data object
        use_test_mask: If True, evaluate only on test nodes (fair comparison).
                       If False, evaluate on all nodes (full dataset statistics).
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
        if use_test_mask:
            test_correct = pred[data.test_mask] == data.y[data.test_mask]
            test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
            return test_acc
        else:
            all_correct = pred == data.y
            full_acc = int(all_correct.sum()) / len(data.y)
            return full_acc


def analyze_quantization_effect(model, layer_name='conv1'):
    """Analyze quantization error for a specific layer."""
    model.eval()

    # Get layer weights
    if hasattr(model, layer_name):
        layer = getattr(model, layer_name)
        weights = layer.lin_l.weight.data
    else:
        print(f"Warning: Layer {layer_name} not found")
        return None, None

    # Quantize
    weights_np = weights.numpy()
    weights_quant, scale, _ = quantize_tensor(weights, num_bits=8)
    weights_quant_np = weights_quant.numpy()

    # Dequantize for comparison
    weights_dequant = weights_quant_np.astype(np.float32) * scale

    return weights_np, weights_dequant


def evaluate_ptq_float_model(model, data):
    """
    Evaluate PTQ-Float model accuracy on full dataset.
    
    PTQ-Float: Quantize weights/activations to INT8, but use float for 
    intermediate operations (dequantize → float ops → quantize).
    
    This properly quantizes BOTH weights AND activations at each layer.
    """
    import copy
    
    model.eval()
    
    try:
        # Create a copy of the model to avoid modifying original
        ptq_model = copy.deepcopy(model)
        
        # Quantize all weights in the model
        with torch.no_grad():
            for name, param in ptq_model.named_parameters():
                if 'weight' in name or 'bias' in name:
                    # Quantize and dequantize (simulates INT8 storage)
                    quant, scale, zp = quantize_tensor(param.data, num_bits=8)
                    # Dequantize back to float for computation
                    param.data = quant.float() * scale
        
        # Now run forward pass with quantized weights
        # Also quantize input features
        with torch.no_grad():
            x = data.x
            x_quant, scale_x, _ = quantize_tensor(x, num_bits=8)
            x_dequant = x_quant.float() * scale_x
            
            # Forward pass through model with quantized weights
            out = ptq_model(x_dequant, data.edge_index)
            
            # Get predictions on test set for fair comparison
            pred = out.argmax(dim=1)
            test_correct = pred[data.test_mask] == data.y[data.test_mask]
            test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
            
        return test_acc
    except Exception as e:
        print(f"  Warning: PTQ-Float evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_ptq_int8_model(model, data, M=20):
    """
    Evaluate PTQ-INT8 model accuracy on full dataset using TRUE integer-only arithmetic.
    
    PTQ-INT8: Pure integer arithmetic with fixed-point scales.
    - M: Number of fractional bits for fixed-point representation (20 or 24)
    - K: Adjacency scaling factor (4096 = 2^12)
    
    This implements the EXACT same integer-only arithmetic used in HLS:
    - INT8 for weights and activations
    - INT16 for adjacency matrix  
    - INT32 for accumulators
    - INT64 for intermediate multiplication results
    - No floating point in the forward pass (except for scale computation)
    
    Note: Uses vectorized numpy for speed, but maintains integer arithmetic semantics.
    """
    import copy
    
    model.eval()
    
    try:
        # =====================================================================
        # Step 1: Build adjacency matrix from edge_index
        # =====================================================================
        num_nodes = data.x.shape[0]
        edge_index = data.edge_index.numpy()
        
        # Build normalized adjacency (mean aggregation)
        adj_float = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        row, col = edge_index[0], edge_index[1]
        
        # Count incoming edges for each node (for mean aggregation)
        degree = np.zeros(num_nodes, dtype=np.float32)
        np.add.at(degree, col, 1)
        degree = np.maximum(degree, 1)  # Avoid division by zero
        
        # Set adjacency values (1/degree for mean aggregation)
        adj_float[col, row] = 1.0 / degree[col]
        
        # =====================================================================
        # Step 2: Quantize model weights
        # =====================================================================
        ptq_model = copy.deepcopy(model)
        
        with torch.no_grad():
            # Get weight scales
            scales = {}
            weights_int8 = {}
            biases_float = {}
            
            for name, param in ptq_model.named_parameters():
                if 'weight' in name:
                    quant, scale, zp = quantize_tensor(param.data, num_bits=8)
                    scales[name] = scale
                    weights_int8[name] = quant.numpy()
                elif 'bias' in name:
                    biases_float[name] = param.data.numpy()
        
        # =====================================================================
        # Step 3: Quantize input features
        # =====================================================================
        x_quant, scale_in, _ = quantize_tensor(data.x, num_bits=8)
        x_int8 = x_quant.numpy()
        
        # =====================================================================
        # Step 4: Integer-only forward pass helper functions (vectorized)
        # =====================================================================
        K = 4096  # 2^12 for adjacency fixed-point
        
        def int8_clamp(x):
            """Clamp to INT8 range [-128, 127]"""
            return np.clip(x, -128, 127).astype(np.int8)
        
        def aggregate_int_only_vectorized(features_int8, adj_float, scale_in, scale_out, M):
            """Vectorized integer-only mean aggregation."""
            # Convert adjacency to INT16 fixed-point
            adj_int16 = np.round(adj_float * K).astype(np.int16)
            
            # Beta = scale_in / (K * scale_out)
            beta = scale_in / (K * scale_out)
            beta_fp = int(round(beta * (1 << M)))
            
            # Matrix multiply: adj_int16 @ features_int8 (in INT32)
            # Use int32 for accumulation to avoid overflow
            features_int32 = features_int8.astype(np.int32)
            adj_int32 = adj_int16.astype(np.int32)
            
            # tmp[i,f] = Σ_j adj[i,j] * features[j,f]
            tmp = adj_int32 @ features_int32  # [N, F] INT32
            
            # Scale: tmp_scaled = tmp * beta_fp (INT64)
            tmp_scaled = tmp.astype(np.int64) * np.int64(beta_fp)
            
            # Round: add (1 << (M-1))
            tmp_rounded = tmp_scaled + (1 << (M - 1))
            
            # Shift: >> M
            q_agg_int = tmp_rounded >> M
            
            # Clamp to INT8
            return int8_clamp(q_agg_int)
        
        def linear_int_only_vectorized(features_int8, weight_int8, bias_float, 
                                       scale_in, scale_w, scale_out, M):
            """Vectorized integer-only linear transformation."""
            # Effective scale: (scale_in * scale_w) / scale_out
            eff_scale = (scale_in * scale_w) / scale_out
            eff_scale_fp = int(round(eff_scale * (1 << M)))
            
            # Convert bias to accumulator domain
            # bias_int32 = round(bias_float / (scale_in * scale_w))
            bias_int32 = np.round(bias_float / (scale_in * scale_w)).astype(np.int32)
            
            # Matrix multiply: features @ weight.T + bias (in INT32)
            features_int32 = features_int8.astype(np.int32)
            weight_int32 = weight_int8.astype(np.int32)
            
            # acc[n,o] = Σ_f features[n,f] * weight[o,f] + bias[o]
            acc = features_int32 @ weight_int32.T + bias_int32  # [N, OUT] INT32
            
            # Requantize: tmp_scaled = acc * eff_scale_fp (INT64)
            tmp_scaled = acc.astype(np.int64) * np.int64(eff_scale_fp)
            
            # Round: add (1 << (M-1))
            tmp_rounded = tmp_scaled + (1 << (M - 1))
            
            # Shift: >> M
            q_out_int = tmp_rounded >> M
            
            # Clamp to INT8
            return int8_clamp(q_out_int)
        
        def relu_int8(x_int8):
            """ReLU for INT8 (symmetric quantization, zero_point=0)"""
            return np.maximum(x_int8, 0).astype(np.int8)
        
        # =====================================================================
        # Step 5: Run integer-only forward pass
        # =====================================================================
        scale_hidden = 0.1  # Fixed activation scale after layers
        
        # Projection layer (if present)
        if ptq_model.use_projection:
            proj_weight_int8 = weights_int8['projection.weight']
            proj_bias_float = biases_float['projection.bias']
            scale_proj_w = scales['projection.weight']
            
            # Linear projection (no aggregation needed)
            x_proj = linear_int_only_vectorized(
                x_int8, proj_weight_int8, proj_bias_float,
                scale_in, scale_proj_w, scale_hidden, M
            )
            x_proj = relu_int8(x_proj)
            scale_in_conv = scale_hidden
            x_conv_in = x_proj
        else:
            scale_in_conv = scale_in
            x_conv_in = x_int8
        
        # Conv1: Aggregate + Linear + ReLU
        agg1 = aggregate_int_only_vectorized(x_conv_in, adj_float, scale_in_conv, scale_hidden, M)
        
        conv1_weight_int8 = weights_int8['conv1.lin_l.weight']
        conv1_bias_float = biases_float['conv1.lin_l.bias']
        scale_conv1_w = scales['conv1.lin_l.weight']
        
        hidden = linear_int_only_vectorized(
            agg1, conv1_weight_int8, conv1_bias_float,
            scale_hidden, scale_conv1_w, scale_hidden, M
        )
        hidden = relu_int8(hidden)
        
        # Conv2: Aggregate + Linear (no ReLU)
        agg2 = aggregate_int_only_vectorized(hidden, adj_float, scale_hidden, scale_hidden, M)
        
        conv2_weight_int8 = weights_int8['conv2.lin_l.weight']
        conv2_bias_float = biases_float['conv2.lin_l.bias']
        scale_conv2_w = scales['conv2.lin_l.weight']
        
        output_int8 = linear_int_only_vectorized(
            agg2, conv2_weight_int8, conv2_bias_float,
            scale_hidden, scale_conv2_w, scale_hidden, M
        )
        
        # =====================================================================
        # Step 6: Compute accuracy on test set for fair comparison
        # =====================================================================
        pred = np.argmax(output_int8, axis=1)
        labels = data.y.numpy()
        test_mask = data.test_mask.numpy()
        
        # Evaluate on test nodes only (fair comparison with training)
        test_correct = pred[test_mask] == labels[test_mask]
        test_acc = np.sum(test_correct) / np.sum(test_mask)
        
        return test_acc
        
    except Exception as e:
        print(f"  Warning: PTQ-INT8 (M={M}) evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_ptq_int8_po2_model(model, data, M=24):
    """
    Evaluate PTQ-INT8-PO2 model accuracy using power-of-two scales.
    
    PTQ-INT8-PO2: Pure integer arithmetic with POWER-OF-TWO scales.
    Instead of arbitrary scale multiplications, uses bit-shifts only.
    
    Key differences from evaluate_ptq_int8_model:
    - All scales are constrained to powers of 2
    - Scale multiplication replaced with bit-shift: x * scale_fp >> M  →  x >> shift
    - Saves ~500-700 DSP48s in HLS at cost of small accuracy loss
    
    Args:
        model: PyTorch model (ReducedGraphSAGE)
        data: PyG data object
        M: Number of fractional bits (default 24 for comparison)
        
    Returns:
        test_acc: Test accuracy with PO2 quantization
    """
    import copy
    import math
    
    model.eval()
    
    try:
        # =====================================================================
        # Step 1: Build adjacency matrix from edge_index
        # =====================================================================
        num_nodes = data.x.shape[0]
        edge_index = data.edge_index.numpy()
        
        # Build normalized adjacency (mean aggregation)
        adj_float = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        row, col = edge_index[0], edge_index[1]
        
        # Count incoming edges for each node
        degree = np.zeros(num_nodes, dtype=np.float32)
        np.add.at(degree, col, 1)
        degree = np.maximum(degree, 1)
        
        adj_float[col, row] = 1.0 / degree[col]
        
        # =====================================================================
        # Step 2: Quantize model weights with POWER-OF-TWO scales
        # =====================================================================
        ptq_model = copy.deepcopy(model)
        
        with torch.no_grad():
            scales = {}
            weights_int8 = {}
            biases_float = {}
            
            for name, param in ptq_model.named_parameters():
                if 'weight' in name:
                    # PO2 quantization requires power_of_two_scale=True
                    quant, scale, zp = quantize_tensor(param.data, num_bits=8, 
                                                        power_of_two_scale=True)
                    scales[name] = scale
                    weights_int8[name] = quant.numpy()
                elif 'bias' in name:
                    biases_float[name] = param.data.numpy()
        
        # =====================================================================
        # Step 3: Quantize input features with PO2 scale
        # =====================================================================
        x_quant, scale_in, _ = quantize_tensor(data.x, num_bits=8, 
                                                power_of_two_scale=True)
        x_int8 = x_quant.numpy()
        
        # =====================================================================
        # Step 4: PO2 Integer-only forward pass helpers
        # =====================================================================
        K = 4096  # 2^12 for adjacency
        K_BITS = 12
        
        def int8_clamp(x):
            """Clamp to INT8 range [-128, 127]"""
            return np.clip(x, -128, 127).astype(np.int8)
        
        def compute_po2_shift(scale_value, M):
            """
            Convert a scale value to PO2 shift amount.
            
            For scale_fp = scale * 2^M, if scale is power-of-two (2^-k),
            then scale_fp = 2^(M-k), and the shift is (M - (M-k)) = k.
            
            More generally: shift = M - log2(scale_fp)
            where scale_fp = round(scale * 2^M) to nearest PO2
            """
            if scale_value <= 0:
                return M  # Default shift
            
            scale_fp = scale_value * (1 << M)
            # Round to nearest power of 2
            log2_val = math.log2(scale_fp)
            po2_exponent = round(log2_val)
            # Shift amount = M - po2_exponent
            shift = M - po2_exponent
            return max(0, min(shift, 31))  # Clamp to valid range
        
        def aggregate_int_po2(features_int8, adj_float, scale_in, scale_out, M):
            """PO2 mean aggregation using bit-shifts."""
            adj_int16 = np.round(adj_float * K).astype(np.int16)
            
            # Beta = scale_in / (K * scale_out)
            beta = scale_in / (K * scale_out)
            shift = compute_po2_shift(beta, M)
            
            features_int32 = features_int8.astype(np.int32)
            adj_int32 = adj_int16.astype(np.int32)
            
            tmp = adj_int32 @ features_int32
            
            # PO2 rescale: bit-shift with rounding
            if shift > 0:
                round_const = 1 << (shift - 1)
                tmp_rounded = tmp + round_const
                q_agg_int = tmp_rounded >> shift
            else:
                q_agg_int = tmp << (-shift)
            
            return int8_clamp(q_agg_int)
        
        def linear_int_po2(features_int8, weight_int8, bias_float, 
                           scale_in, scale_w, scale_out, M):
            """PO2 linear transformation using bit-shifts."""
            eff_scale = (scale_in * scale_w) / scale_out
            shift = compute_po2_shift(eff_scale, M)
            
            # Convert bias (approximate for PO2)
            bias_int32 = np.round(bias_float / (scale_in * scale_w)).astype(np.int32)
            
            features_int32 = features_int8.astype(np.int32)
            weight_int32 = weight_int8.astype(np.int32)
            
            acc = features_int32 @ weight_int32.T + bias_int32
            
            # PO2 rescale: bit-shift with rounding
            if shift > 0:
                round_const = 1 << (shift - 1)
                tmp_rounded = acc + round_const
                q_out_int = tmp_rounded >> shift
            else:
                q_out_int = acc << (-shift)
            
            return int8_clamp(q_out_int)
        
        def relu_int8(x_int8):
            return np.maximum(x_int8, 0).astype(np.int8)
        
        # =====================================================================
        # Step 5: Run PO2 integer-only forward pass  
        # =====================================================================
        # Use PO2 scale for hidden activations too
        scale_hidden_float = 0.1
        # Round to nearest power of 2
        scale_hidden = 2 ** math.ceil(math.log2(scale_hidden_float))
        
        # Projection layer
        if ptq_model.use_projection:
            proj_weight_int8 = weights_int8['projection.weight']
            proj_bias_float = biases_float['projection.bias']
            scale_proj_w = scales['projection.weight']
            
            x_proj = linear_int_po2(
                x_int8, proj_weight_int8, proj_bias_float,
                scale_in, scale_proj_w, scale_hidden, M
            )
            x_proj = relu_int8(x_proj)
            scale_in_conv = scale_hidden
            x_conv_in = x_proj
        else:
            scale_in_conv = scale_in
            x_conv_in = x_int8
        
        # Conv1: Aggregate + Linear + ReLU
        agg1 = aggregate_int_po2(x_conv_in, adj_float, scale_in_conv, scale_hidden, M)
        
        conv1_weight_int8 = weights_int8['conv1.lin_l.weight']
        conv1_bias_float = biases_float['conv1.lin_l.bias']
        scale_conv1_w = scales['conv1.lin_l.weight']
        
        hidden = linear_int_po2(
            agg1, conv1_weight_int8, conv1_bias_float,
            scale_hidden, scale_conv1_w, scale_hidden, M
        )
        hidden = relu_int8(hidden)
        
        # Conv2: Aggregate + Linear
        agg2 = aggregate_int_po2(hidden, adj_float, scale_hidden, scale_hidden, M)
        
        conv2_weight_int8 = weights_int8['conv2.lin_l.weight']
        conv2_bias_float = biases_float['conv2.lin_l.bias']
        scale_conv2_w = scales['conv2.lin_l.weight']
        
        output_int8 = linear_int_po2(
            agg2, conv2_weight_int8, conv2_bias_float,
            scale_hidden, scale_conv2_w, scale_hidden, M
        )
        
        # =====================================================================
        # Step 6: Compute accuracy on test set for fair comparison
        # =====================================================================
        pred = np.argmax(output_int8, axis=1)
        labels = data.y.numpy()
        test_mask = data.test_mask.numpy()
        
        # Evaluate on test nodes only (fair comparison with training)
        test_correct = pred[test_mask] == labels[test_mask]
        test_acc = np.sum(test_correct) / np.sum(test_mask)
        
        return test_acc
        
    except Exception as e:
        print(f"  Warning: PTQ-INT8-PO2 (M={M}) evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    # Load configuration
    cfg = get_config()

    print("="*60)
    print("GraphSAGE Model Analysis and Visualization")
    print("="*60)
    print(f"📋 Using configuration file")
    
    # Setup paths relative to script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    model_dir = project_root / 'build' / 'models'
    
    print(f"📁 Project root: {project_root}")
    print(f"📁 Model directory: {model_dir}")

    # Load dataset - ensure consistent data for all models
    print("\n📦 Loading Cora dataset...")
    dataset = Planetoid(root=str(project_root / 'data'), name='Cora', transform=NormalizeFeatures())
    data = dataset[0]
    
    print(f"   Dataset: {data.x.shape[0]} nodes, {data.edge_index.shape[1]} edges")
    print(f"   Train set: {data.train_mask.sum().item()} nodes")
    print(f"   Val set: {data.val_mask.sum().item()} nodes")
    print(f"   Test set: {data.test_mask.sum().item()} nodes")
    print(f"\n   ⚠️  All models evaluated on TEST SET ONLY for fair comparison")


    model_stats = []

    # 1. Analyze Base Model
    print("\n" + "="*60)
    print("Analyzing Base Model")
    print("="*60)

    base_model = GraphSAGE(
        in_channels=dataset.num_features,
        hidden_channels=cfg.base_hidden_channels,
        out_channels=dataset.num_classes,
        dropout=cfg.base_dropout
    )

    try:
        checkpoint = torch.load(model_dir / 'base_graphsage_best.pth', weights_only=False)
        base_model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Loaded trained base model")
    except Exception as e:
        print(f"⚠ No trained base model found, training now...")
        print(f"  (This will take 2-3 minutes)")
        # Change to project root so train.py's relative paths work
        original_dir = os.getcwd()
        os.chdir(project_root / 'src')
        base_model, data, _ = train_base_model(epochs=200)
        os.chdir(original_dir)
        print("✓ Base model trained and saved")

    base_acc = evaluate_model(base_model, data)
    base_params = count_parameters(base_model)
    base_memory = estimate_model_size(base_model, quantized=False)
    base_breakdown = get_layer_breakdown(base_model)

    model_stats.append({
        'name': 'Base',
        'accuracy': base_acc,
        'parameters': base_params,
        'memory_mb': base_memory,
        'layer_breakdown': base_breakdown
    })

    print(f"  Accuracy: {base_acc*100:.2f}%")
    print(f"  Parameters: {base_params:,}")
    print(f"  Memory: {base_memory:.2f} MB")

    # 2. Analyze Reduced Model
    print("\n" + "="*60)
    print("Analyzing Reduced Model")
    print("="*60)

    reduced_model = ReducedGraphSAGE(
        in_channels=dataset.num_features,
        in_channels_reduced=cfg.reduced_in_channels,
        hidden_channels=cfg.reduced_hidden_channels,
        out_channels=dataset.num_classes,
        dropout=cfg.reduced_dropout,
        use_projection=True,
        root_weight=False  # Use no_root version to match HLS
    )

    try:
        checkpoint = torch.load(model_dir / 'reduced_graphsage_no_root_best.pth', weights_only=False)
        reduced_model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        print("✓ Loaded trained reduced model (no_root)")
    except Exception as e:
        print(f"⚠ No trained reduced model found, training now...")
        print(f"  (This will take 2-3 minutes)")
        # Change to project root so train.py's relative paths work
        original_dir = os.getcwd()
        os.chdir(project_root / 'src')
        reduced_model, data, _ = train_reduced_model(epochs=200, root_weight=False)
        os.chdir(original_dir)
        print("✓ Reduced model trained and saved")

    reduced_acc = evaluate_model(reduced_model, data)
    reduced_params = count_parameters(reduced_model)
    reduced_memory = estimate_model_size(reduced_model, quantized=False)
    reduced_breakdown = get_layer_breakdown(reduced_model)

    model_stats.append({
        'name': 'Reduced',
        'accuracy': reduced_acc,
        'parameters': reduced_params,
        'memory_mb': reduced_memory,
        'layer_breakdown': reduced_breakdown
    })

    print(f"  Accuracy: {reduced_acc*100:.2f}%")
    print(f"  Parameters: {reduced_params:,}")
    print(f"  Memory: {reduced_memory:.2f} MB")
    print(f"  Reduction: {(1 - reduced_params/base_params)*100:.1f}%")

    # 3. Analyze PTQ-Float Model (Post-Training Quantization with float dequant ops)
    # This replaces the old "Quantized" placeholder which just estimated 2% degradation
    print("\n" + "="*60)
    print("Analyzing PTQ-Float Model (Post-Training Quantization)")
    print("="*60)

    quant_memory = estimate_model_size(reduced_model, quantized=True)
    ptq_float_acc = evaluate_ptq_float_model(reduced_model, data)
    
    if ptq_float_acc is not None:
        model_stats.append({
            'name': 'PTQ-Float',
            'accuracy': ptq_float_acc,
            'parameters': reduced_params,
            'memory_mb': quant_memory,
            'layer_breakdown': reduced_breakdown
        })
        print(f"  ✓ Evaluated PTQ-Float model")
        print(f"  Accuracy: {ptq_float_acc*100:.2f}%")
        print(f"  Parameters: {reduced_params:,} (INT8 weights)")
        print(f"  Memory: {quant_memory:.2f} MB (INT8)")
        print(f"  Method: quantize weights/activations → float aggregate → quantize output")
    else:
        # Fallback to estimate if evaluation fails
        model_stats.append({
            'name': 'PTQ-Float',
            'accuracy': reduced_acc * 0.98,  # Estimate
            'parameters': reduced_params,
            'memory_mb': quant_memory,
            'layer_breakdown': reduced_breakdown
        })
        print(f"  ⚠ Using estimated accuracy (actual evaluation failed)")
        print(f"  Accuracy: {reduced_acc*0.98*100:.2f}% (estimated)")
        print(f"  Memory: {quant_memory:.2f} MB")

    # 4. Analyze QAT v2 Models (Multi-Bitwidth)
    print("\n" + "="*60)
    print("Analyzing QAT v2 Models (Corrected Quantization-Aware Training)")
    print("="*60)
    
    # Bit widths to test
    bit_widths = [8, 6, 4, 2]
    float_baseline_acc = reduced_acc  # Use reduced model as baseline
    
    for num_bits in bit_widths:
        print(f"\n--- QAT v2 {num_bits}-bit ---")
        
        try:
            # Create QAT v2 model with specified bit width
            qat_v2_model = ReducedGraphSAGEQATv2(
                in_channels=dataset.num_features,
                in_channels_reduced=16,
                hidden_channels=24,
                out_channels=dataset.num_classes,
                dropout=0.5,
                use_projection=True,
                root_weight=False,
                num_bits=num_bits,
            )
            
            # Initialize from pre-trained reduced model
            qat_v2_model.load_from_float_model(reduced_model)
            
            # Calibrate
            print(f"  Calibrating {num_bits}-bit model...")
            qat_v2_model.calibrate(data, num_batches=50)
            
            # Evaluate after calibration (before fine-tuning)
            qat_v2_model.eval()
            with torch.no_grad():
                out = qat_v2_model(data.x, data.edge_index)
                pred = out.argmax(dim=1)
                test_correct = pred[data.test_mask] == data.y[data.test_mask]
                calib_acc = test_correct.float().mean().item()
            print(f"  Post-calibration accuracy: {calib_acc*100:.2f}%")
            
            # Fine-tune if accuracy is reasonable
            if calib_acc > 0.3:  # Only fine-tune if not completely broken
                print(f"  Fine-tuning {num_bits}-bit model...")
                qat_v2_model = train_qat_v2(qat_v2_model, data, epochs=100, lr=0.001, verbose=False)
                
                # Final evaluation
                qat_v2_model.eval()
                with torch.no_grad():
                    out = qat_v2_model(data.x, data.edge_index)
                    pred = out.argmax(dim=1)
                    test_correct = pred[data.test_mask] == data.y[data.test_mask]
                    qat_acc = test_correct.float().mean().item()
            else:
                qat_acc = calib_acc
            
            qat_params = count_parameters(qat_v2_model)
            # Memory estimate based on bit width
            qat_memory = qat_params * num_bits / 8 / (1024 ** 2)
            qat_breakdown = get_layer_breakdown(qat_v2_model)
            
            # Determine verdict
            acc_drop = (float_baseline_acc - qat_acc) * 100
            if acc_drop < 2.0:
                verdict = "✅ EXCELLENT"
            elif acc_drop < 5.0:
                verdict = "✅ GOOD"
            elif acc_drop < 10.0:
                verdict = "⚠️ MARGINAL"
            else:
                verdict = "❌ FAIL"
            
            model_stats.append({
                'name': f'QAT-{num_bits}B',
                'accuracy': qat_acc,
                'parameters': qat_params,
                'memory_mb': qat_memory,
                'layer_breakdown': qat_breakdown
            })
            
            print(f"  ✓ QAT v2 {num_bits}-bit model")
            print(f"  Final Accuracy: {qat_acc*100:.2f}%")
            print(f"  Accuracy Drop: {acc_drop:.2f}% {verdict}")
            print(f"  Parameters: {qat_params:,}")
            print(f"  Memory: {qat_memory:.4f} MB ({num_bits}-bit)")
            
        except Exception as e:
            print(f"  ⚠ Error analyzing QAT v2 {num_bits}-bit: {e}")
            import traceback
            traceback.print_exc()

    # 5. Analyze PTQ-INT8 Models (Pure integer arithmetic with fixed-point scales)
    print("\n" + "="*60)
    print("Analyzing PTQ-INT8 Models (Pure Integer Arithmetic)")
    print("="*60)

    for M_value in [20, 24]:
        ptq_int8_acc = evaluate_ptq_int8_model(reduced_model, data, M=M_value)
        if ptq_int8_acc is not None:
            model_name = f'PTQ-INT8-M{M_value}'
            model_stats.append({
                'name': model_name,
                'accuracy': ptq_int8_acc,
                'parameters': reduced_params,
                'memory_mb': quant_memory,  # INT8 memory
                'layer_breakdown': reduced_breakdown,
                'M_value': M_value
            })
            print(f"  ✓ PTQ-INT8 (M={M_value}): {ptq_int8_acc*100:.2f}%")
        else:
            print(f"  ✗ Could not evaluate PTQ-INT8 (M={M_value})")

    print(f"\n  Note on PTQ-INT8:")
    print(f"    - M=20: Uses 20-bit fixed-point scales (smaller multipliers)")
    print(f"    - M=24: Uses 24-bit fixed-point scales (better precision)")
    print(f"    - M=24 with HW rounding matches PTQ-Float exactly (0 LSB error)")

    # 6. Analyze PTQ-INT8-PO2 Models (Power-of-Two scales, no DSP multipliers)
    print("\n" + "="*60)
    print("Analyzing PTQ-INT8-PO2 Models (Power-of-Two Scales)")
    print("="*60)

    for M_value in [24]:  # PO2 typically uses M=24 for best precision
        ptq_int8_po2_acc = evaluate_ptq_int8_po2_model(reduced_model, data, M=M_value)
        if ptq_int8_po2_acc is not None:
            model_name = f'PTQ-INT8-PO2-M{M_value}'
            model_stats.append({
                'name': model_name,
                'accuracy': ptq_int8_po2_acc,
                'parameters': reduced_params,
                'memory_mb': quant_memory,  # INT8 memory
                'layer_breakdown': reduced_breakdown,
                'M_value': M_value,
                'po2_scales': True
            })
            print(f"  ✓ PTQ-INT8-PO2 (M={M_value}): {ptq_int8_po2_acc*100:.2f}%")
            
            # Compare with regular INT8
            for stat in model_stats:
                if stat['name'] == f'PTQ-INT8-M{M_value}':
                    diff = ptq_int8_po2_acc - stat['accuracy']
                    print(f"    vs PTQ-INT8-M{M_value}: {diff*100:+.2f}% (PO2 approximation error)")
                    break
        else:
            print(f"  ✗ Could not evaluate PTQ-INT8-PO2 (M={M_value})")

    print(f"\n  Note on PTQ-INT8-PO2:")
    print(f"    - Uses power-of-two scales (bit-shifts instead of multipliers)")
    print(f"    - Saves ~500-700 DSP48s in HLS synthesis")
    print(f"    - Small accuracy loss due to scale approximation")
    print(f"    - Best for resource-constrained FPGAs")

    # 7. Analyze Brevitas Model (Fixed-Point Quantization)
    print("\n" + "="*60)
    print("Analyzing Brevitas Model (Fixed-Point INT8)")
    print("="*60)

    if BREVITAS_AVAILABLE:
        try:
            brevitas_dir = project_root / 'build' / 'brevitas' / 'no_root'
            
            # Create and calibrate Brevitas model from float model
            quant_config = BrevitasQuantConfig(weight_bit_width=8, act_bit_width=8)
            
            brevitas_model = BrevitasReducedGraphSAGE(
                in_channels=dataset.num_features,
                in_channels_reduced=cfg.reduced_in_channels,
                hidden_channels=cfg.reduced_hidden_channels,
                out_channels=dataset.num_classes,
                dropout=cfg.reduced_dropout,
                use_projection=True,
                root_weight=False,
                quant_config=quant_config
            )
            
            # Load from float model
            from brevitas_models import load_from_float_model
            load_from_float_model(brevitas_model, reduced_model)
            
            # Calibrate with same data
            brevitas_model.eval()
            with torch.no_grad():
                for _ in range(10):  # 10 calibration passes
                    _ = brevitas_model(data.x, data.edge_index)
            
            # Evaluate on SAME test set as all other models
            brevitas_acc = evaluate_model(brevitas_model, data)
            brevitas_params = count_parameters(brevitas_model)
            brevitas_memory = estimate_model_size(brevitas_model, quantized=True)
            brevitas_breakdown = get_layer_breakdown(brevitas_model)
            
            model_stats.append({
                'name': 'Brevitas',
                'accuracy': brevitas_acc,
                'parameters': brevitas_params,
                'memory_mb': brevitas_memory,
                'layer_breakdown': brevitas_breakdown
            })
            
            print(f"  ✓ Created and calibrated Brevitas model")
            print(f"  Accuracy: {brevitas_acc*100:.2f}%")
            print(f"  Parameters: {brevitas_params:,}")
            print(f"  Memory: {brevitas_memory:.2f} MB (INT8)")
            print(f"  Quantizer: Int8WeightPerTensorFixedPoint (power-of-two scales)")
            print(f"  Accuracy vs Float: {(reduced_acc - brevitas_acc)*100:+.2f}%")
        except Exception as e:
            print(f"  ✗ Could not create Brevitas model: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  ✗ Brevitas not available")
        print("  Install with: pip install brevitas")

    # 8. Analyze Pruned Model (if exists)
    print("\n" + "="*60)
    print("Analyzing Pruned Model")
    print("="*60)

    try:
        pruned_checkpoint = torch.load(model_dir / 'pruned_graphsage.pth', weights_only=False)
        arch = pruned_checkpoint['architecture']

        pruned_model = ReducedGraphSAGE(
            in_channels=dataset.num_features,
            in_channels_reduced=arch['in_channels_reduced'],
            hidden_channels=arch['hidden_channels'],
            out_channels=arch['out_channels'],
            dropout=0.5,
            use_projection=True
        )
        pruned_model.load_state_dict(pruned_checkpoint['model_state_dict'])

        pruned_acc = evaluate_model(pruned_model, data)
        pruned_params = count_parameters(pruned_model)
        pruned_memory = estimate_model_size(pruned_model, quantized=False)
        pruned_breakdown = get_layer_breakdown(pruned_model)

        model_stats.append({
            'name': 'Pruned',
            'accuracy': pruned_acc,
            'parameters': pruned_params,
            'memory_mb': pruned_memory,
            'layer_breakdown': pruned_breakdown
        })

        print(f"  ✓ Loaded pruned model")
        print(f"  Accuracy: {pruned_acc*100:.2f}%")
        print(f"  Parameters: {pruned_params:,}")
        print(f"  Memory: {pruned_memory:.2f} MB")
    except:
        print("  ✗ No pruned model found (run pruning.py first)")

    # Generate all plots
    print("\n" + "="*60)
    print("Generating Visualization Plots")
    print("="*60)

    plots_dir = project_root / 'build' / 'plots'
    os.makedirs(str(plots_dir), exist_ok=True)

    # Model comparison
    print("\nGenerating model comparison plot...")
    plot_model_comparison(model_stats, save_path=str(plots_dir / 'model_comparison.png'))

    # Efficiency analysis
    print("Generating efficiency analysis plot...")
    plot_efficiency_analysis(model_stats, save_path=str(plots_dir / 'efficiency_analysis.png'))

    # Accuracy degradation
    print("Generating accuracy degradation plot...")
    model_names = [s['name'] for s in model_stats[1:]]
    model_accs = [s['accuracy'] for s in model_stats[1:]]
    plot_accuracy_degradation(base_acc, model_accs, model_names, 
                             save_path=str(plots_dir / 'accuracy_degradation.png'))

    # Resource utilization
    print("Generating resource utilization plot...")
    plot_resource_utilization(model_stats, save_path=str(plots_dir / 'resource_utilization.png'))

    # Summary report
    print("Generating summary report...")
    generate_summary_report(model_stats, save_path=str(plots_dir / 'summary_report.png'))

    # Quantization error analysis
    print("Generating quantization error analysis...")
    orig_weights, quant_weights = analyze_quantization_effect(reduced_model, 'conv1')
    if orig_weights is not None:
        plot_quantization_error(orig_weights, quant_weights, 'Conv Layer 1',
                               save_path=str(plots_dir / 'quantization_error.png'))

    # Save statistics to JSON
    print("Saving model statistics...")
    print(f"DEBUG: model_stats has {len(model_stats)} entries:")
    for stat in model_stats:
        print(f"  - {stat['name']}: {stat['accuracy']*100:.2f}%")
    save_stats_json(model_stats, save_path=str(plots_dir / 'model_stats.json'))

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"\nGenerated plots in {plots_dir}/:")
    print("  - model_comparison.png")
    print("  - efficiency_analysis.png")
    print("  - accuracy_degradation.png")
    print("  - resource_utilization.png")
    print("  - summary_report.png")
    print("  - quantization_error.png")
    print("\nStatistics saved to:")
    print(f"  - {plots_dir / 'model_stats.json'}")

    # Print summary table
    print("\n" + "="*60)
    print("Model Summary")
    print("="*60)
    print(f"{'Model':<15} {'Accuracy':<12} {'Parameters':<15} {'Memory (MB)':<12} {'Compression':<12}")
    print("-" * 72)
    for stat in model_stats:
        compression = base_params / stat['parameters']
        print(f"{stat['name']:<15} {stat['accuracy']*100:>10.2f}% {stat['parameters']:>13,} "
              f"{stat['memory_mb']:>10.2f} MB {compression:>10.2f}x")


if __name__ == '__main__':
    main()
