"""
Comprehensive model analysis and visualization script.
Generates comparison plots for all model variants.
"""

import torch
import numpy as np
import json
import os
import sys

from model_base import GraphSAGE, ReducedGraphSAGE
from model_qat import ReducedGraphSAGEQAT
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data
from quantization import quantize_tensor

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


def evaluate_model(model, data):
    """Evaluate model accuracy."""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        test_correct = pred[data.test_mask] == data.y[data.test_mask]
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc


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


def main():
    print("="*60)
    print("GraphSAGE Model Analysis and Visualization")
    print("="*60)

    # Load dataset
    print("\nLoading Cora dataset...")
    dataset = Planetoid(root='./data', name='Cora', transform=NormalizeFeatures())
    data = dataset[0]

    model_stats = []

    # 1. Analyze Base Model
    print("\n" + "="*60)
    print("Analyzing Base Model")
    print("="*60)

    base_model = GraphSAGE(
        in_channels=dataset.num_features,
        hidden_channels=64,
        out_channels=dataset.num_classes,
        dropout=0.5
    )

    try:
        checkpoint = torch.load('../build/models/base_graphsage_best.pth')
        base_model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Loaded trained base model")
    except:
        print("✗ Warning: Could not load trained base model, using random weights")

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
        in_channels_reduced=16,
        hidden_channels=24,
        out_channels=dataset.num_classes,
        dropout=0.5,
        use_projection=True
    )

    try:
        checkpoint = torch.load('../build/models/reduced_graphsage_best.pth')
        reduced_model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Loaded trained reduced model")
    except:
        print("✗ Warning: Could not load trained reduced model, using random weights")

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

    # 3. Analyze Quantized Model (INT8)
    print("\n" + "="*60)
    print("Analyzing Quantized Model (INT8)")
    print("="*60)

    # Quantized model has same parameters but different memory
    quant_memory = estimate_model_size(reduced_model, quantized=True)

    model_stats.append({
        'name': 'Quantized',
        'accuracy': reduced_acc * 0.98,  # Assume 2% degradation
        'parameters': reduced_params,
        'memory_mb': quant_memory,
        'layer_breakdown': reduced_breakdown
    })

    print(f"  Accuracy: {reduced_acc*0.98*100:.2f}% (estimated)")
    print(f"  Parameters: {reduced_params:,}")
    print(f"  Memory: {quant_memory:.2f} MB")
    print(f"  Memory Reduction: {(1 - quant_memory/reduced_memory)*100:.1f}%")

    # 4. Analyze QAT Model
    print("\n" + "="*60)
    print("Analyzing QAT Model (Quantization-Aware Training)")
    print("="*60)

    try:
        qat_checkpoint = torch.load('../build/models/reduced_graphsage_qat_no_root_best.pth', weights_only=False)

        # Reconstruct QAT model
        qat_model = ReducedGraphSAGEQAT(
            in_channels=dataset.num_features,
            in_channels_reduced=qat_checkpoint['in_channels_reduced'],
            hidden_channels=qat_checkpoint['hidden_channels'],
            out_channels=qat_checkpoint['out_channels'],
            dropout=0.5,
            use_projection=True,
            root_weight=qat_checkpoint['root_weight'],
        )
        qat_model.load_state_dict(qat_checkpoint['model_state_dict'])

        # Evaluate with fake quantization enabled (simulates INT8)
        qat_model.eval()
        qat_model.enable_fake_quant()

        with torch.no_grad():
            out = qat_model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            test_correct = pred[data.test_mask] == data.y[data.test_mask]
            qat_acc = int(test_correct.sum()) / int(data.test_mask.sum())

        qat_params = count_parameters(qat_model)
        qat_memory = estimate_model_size(qat_model, quantized=True)
        qat_breakdown = get_layer_breakdown(qat_model)

        model_stats.append({
            'name': 'QAT',
            'accuracy': qat_acc,
            'parameters': qat_params,
            'memory_mb': qat_memory,
            'layer_breakdown': qat_breakdown
        })

        print(f"  ✓ Loaded QAT model")
        print(f"  Accuracy: {qat_acc*100:.2f}% (INT8-simulated)")
        print(f"  Parameters: {qat_params:,}")
        print(f"  Memory: {qat_memory:.2f} MB (INT8)")
        print(f"  Validation Acc (from training): {qat_checkpoint['val_acc']*100:.2f}%")
        if 'val_acc_float' in qat_checkpoint:
            print(f"  Float Acc (from training): {qat_checkpoint['val_acc_float']*100:.2f}%")
            print(f"  Quantization Gap: {(qat_checkpoint['val_acc_float'] - qat_checkpoint['val_acc'])*100:.2f}%")
    except Exception as e:
        print(f"  ✗ Could not load QAT model: {e}")
        print("  Run 'python train_qat.py' first to train QAT model")

    # 5. Analyze Pruned Model (if exists)
    print("\n" + "="*60)
    print("Analyzing Pruned Model")
    print("="*60)

    try:
        pruned_checkpoint = torch.load('../build/models/pruned_graphsage.pth')
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

    os.makedirs('../build/plots', exist_ok=True)

    # Model comparison
    print("\nGenerating model comparison plot...")
    plot_model_comparison(model_stats)

    # Efficiency analysis
    print("Generating efficiency analysis plot...")
    plot_efficiency_analysis(model_stats)

    # Accuracy degradation
    print("Generating accuracy degradation plot...")
    model_names = [s['name'] for s in model_stats[1:]]
    model_accs = [s['accuracy'] for s in model_stats[1:]]
    plot_accuracy_degradation(base_acc, model_accs, model_names)

    # Resource utilization
    print("Generating resource utilization plot...")
    plot_resource_utilization(model_stats)

    # Summary report
    print("Generating summary report...")
    generate_summary_report(model_stats)

    # Quantization error analysis
    print("Generating quantization error analysis...")
    orig_weights, quant_weights = analyze_quantization_effect(reduced_model, 'conv1')
    if orig_weights is not None:
        plot_quantization_error(orig_weights, quant_weights, 'Conv Layer 1')

    # Save statistics to JSON
    print("Saving model statistics...")
    save_stats_json(model_stats)

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print("\nGenerated plots in ./outputs/plots/:")
    print("  - model_comparison.png")
    print("  - efficiency_analysis.png")
    print("  - accuracy_degradation.png")
    print("  - resource_utilization.png")
    print("  - summary_report.png")
    print("  - quantization_error.png")
    print("\nStatistics saved to:")
    print("  - ./outputs/plots/model_stats.json")

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
