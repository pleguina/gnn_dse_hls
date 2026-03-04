"""
Test QAT v2 with different bit widths (16, 8, 6, 4, 2).
This script evaluates the accuracy vs. bit width trade-off for GNN quantization.
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import numpy as np
import json
from pathlib import Path

from model_base import ReducedGraphSAGE
from model_qat_v2 import ReducedGraphSAGEQATv2, train_qat_v2


def evaluate_model(model, data):
    """Evaluate model accuracy on all nodes."""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        all_correct = pred == data.y
        full_acc = int(all_correct.sum()) / len(data.y)
    return full_acc


def test_qat_bitwidth(float_model, data, num_bits, fine_tune_epochs=50):
    """
    Test QAT v2 with a specific bit width.
    
    Args:
        float_model: Pre-trained float model
        data: PyG data object
        num_bits: Quantization bit width
        fine_tune_epochs: Number of fine-tuning epochs
        
    Returns:
        dict with accuracy metrics
    """
    print(f"\n{'='*60}")
    print(f"Testing QAT v2 with {num_bits}-bit quantization")
    print(f"{'='*60}")
    
    # Create QAT model with specified bit width
    qat_model = ReducedGraphSAGEQATv2(
        in_channels=1433,  # Cora features
        in_channels_reduced=16,
        hidden_channels=24,
        out_channels=7,
        dropout=0.5,
        use_projection=True,
        root_weight=False,
        num_bits=num_bits
    )
    
    # Initialize from float model
    qat_model.load_from_float_model(float_model)
    
    # Evaluate before calibration (should match float)
    qat_model.eval()
    qat_model.disable_fake_quant()
    pre_calib_acc = evaluate_model(qat_model, data)
    print(f"  Pre-calibration (float path): {pre_calib_acc*100:.2f}%")
    
    # Calibrate
    print(f"  Calibrating...")
    qat_model.calibrate(data, num_batches=50)
    
    # Evaluate after calibration
    qat_model.eval()
    qat_model.enable_fake_quant()
    post_calib_acc = evaluate_model(qat_model, data)
    print(f"  Post-calibration (quantized): {post_calib_acc*100:.2f}%")
    
    # Fine-tune
    if fine_tune_epochs > 0:
        print(f"  Fine-tuning for {fine_tune_epochs} epochs...")
        qat_model = train_qat_v2(
            qat_model, data, 
            epochs=fine_tune_epochs, 
            lr=0.001 if num_bits >= 4 else 0.0005  # Lower LR for very low bits
        )
    
    # Final evaluation
    qat_model.eval()
    qat_model.enable_fake_quant()
    final_acc = evaluate_model(qat_model, data)
    print(f"  Final accuracy: {final_acc*100:.2f}%")
    
    # Calculate theoretical memory savings
    # INT8 baseline = 1 byte per param
    # For lower bits, we calculate equivalent size
    params = sum(p.numel() for p in qat_model.parameters())
    memory_mb = (params * num_bits / 8) / (1024 ** 2)
    
    return {
        'num_bits': num_bits,
        'pre_calibration_acc': pre_calib_acc,
        'post_calibration_acc': post_calib_acc,
        'final_acc': final_acc,
        'parameters': params,
        'memory_mb': memory_mb,
        'quantization_levels': 2 ** num_bits,
    }


def main():
    print("="*70)
    print("QAT v2: Multi-Bitwidth Analysis")
    print("="*70)
    
    # Load data
    print("\n📦 Loading Cora dataset...")
    dataset = Planetoid(root='../data', name='Cora', transform=NormalizeFeatures())
    data = dataset[0]
    print(f"   {data.x.shape[0]} nodes, {data.edge_index.shape[1]} edges")
    
    # Load pre-trained float model
    print("\n📂 Loading pre-trained float model...")
    float_model = ReducedGraphSAGE(
        in_channels=dataset.num_features,
        in_channels_reduced=16,
        hidden_channels=24,
        out_channels=dataset.num_classes,
        dropout=0.5,
        use_projection=True,
        root_weight=False
    )
    
    checkpoint = torch.load('../build/models/reduced_graphsage_no_root_best.pth', weights_only=False)
    float_model.load_state_dict(checkpoint['model_state_dict'])
    
    float_acc = evaluate_model(float_model, data)
    print(f"   Float model accuracy: {float_acc*100:.2f}%")
    
    # Test different bit widths
    # Note: PyTorch FakeQuantize is limited to 8 bits max for qint8 dtype
    # For >8 bits, it effectively behaves like 8-bit
    bit_widths = [8, 6, 4, 3, 2]
    results = []
    
    print("\n⚠️  Note: PyTorch FakeQuantize is limited to 8-bit qint8 dtype.")
    print("   Testing bit widths: 8, 6, 4, 3, 2")
    
    for num_bits in bit_widths:
        result = test_qat_bitwidth(float_model, data, num_bits, fine_tune_epochs=50)
        result['float_acc'] = float_acc
        result['accuracy_drop'] = float_acc - result['final_acc']
        results.append(result)
    
    # Print summary table
    print("\n" + "="*80)
    print("RESULTS SUMMARY: QAT v2 Accuracy vs Bit Width")
    print("="*80)
    print(f"{'Bits':>6} | {'Levels':>8} | {'Pre-Cal':>8} | {'Post-Cal':>8} | {'Final':>8} | {'Drop':>8} | {'Memory':>8}")
    print("-"*80)
    
    for r in results:
        print(f"{r['num_bits']:>6} | {r['quantization_levels']:>8} | "
              f"{r['pre_calibration_acc']*100:>7.2f}% | "
              f"{r['post_calibration_acc']*100:>7.2f}% | "
              f"{r['final_acc']*100:>7.2f}% | "
              f"{r['accuracy_drop']*100:>7.2f}% | "
              f"{r['memory_mb']:>6.3f}MB")
    
    print("-"*80)
    print(f"{'Float':>6} | {'∞':>8} | {float_acc*100:>7.2f}% | {float_acc*100:>7.2f}% | "
          f"{float_acc*100:>7.2f}% | {'0.00':>7}% | "
          f"{sum(p.numel() for p in float_model.parameters()) * 4 / (1024**2):>6.3f}MB")
    print("="*80)
    
    # Analysis
    print("\n📊 ANALYSIS:")
    print("-"*60)
    
    # Find the minimum bits that maintain >1% accuracy drop
    for r in results:
        if r['accuracy_drop'] <= 0.01:
            print(f"✓ {r['num_bits']}-bit: Only {r['accuracy_drop']*100:.2f}% drop - EXCELLENT")
        elif r['accuracy_drop'] <= 0.02:
            print(f"✓ {r['num_bits']}-bit: {r['accuracy_drop']*100:.2f}% drop - GOOD")
        elif r['accuracy_drop'] <= 0.05:
            print(f"○ {r['num_bits']}-bit: {r['accuracy_drop']*100:.2f}% drop - ACCEPTABLE")
        else:
            print(f"✗ {r['num_bits']}-bit: {r['accuracy_drop']*100:.2f}% drop - SIGNIFICANT DEGRADATION")
    
    # Memory efficiency
    print("\n💾 MEMORY EFFICIENCY:")
    print("-"*60)
    float_memory = sum(p.numel() for p in float_model.parameters()) * 4 / (1024**2)
    for r in results:
        compression = float_memory / r['memory_mb']
        print(f"  {r['num_bits']:>2}-bit: {r['memory_mb']:.3f} MB ({compression:.1f}x compression vs FP32)")
    
    # Save results
    output_path = Path('../build/plots/qat_v2_bitwidth_analysis.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'float_accuracy': float_acc,
            'results': results
        }, f, indent=2)
    print(f"\n📁 Results saved to {output_path}")
    
    # Create a simple ASCII chart
    print("\n📈 ACCURACY vs BIT WIDTH:")
    print("-"*60)
    max_bar = 40
    for r in results:
        bar_len = int(r['final_acc'] * max_bar / float_acc)
        bar = "█" * bar_len + "░" * (max_bar - bar_len)
        print(f"  {r['num_bits']:>2}-bit: [{bar}] {r['final_acc']*100:.2f}%")
    print(f"  Float: [{'█' * max_bar}] {float_acc*100:.2f}%")
    
    return results


if __name__ == "__main__":
    results = main()
