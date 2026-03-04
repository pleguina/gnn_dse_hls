#!/usr/bin/env python3
"""
Fixed-Point Q Format Evaluation

This tool helps you find the optimal Q(W,I) format for ap_fixed implementation:
- Q(W,I): W = total bits, I = integer bits, F = W-I fractional bits
- Trade-off: More integer bits → larger range, fewer fractional bits → less precision

Usage:
    # Evaluate different Q formats for a specific model
    python src/evaluate_qformat.py --model-path pretrained_models/reduced_24.pth
    
    # Evaluate specific Q formats
    python src/evaluate_qformat.py --data-w 16 24 32 --data-i 8 12 16
    
    # Quick comparison of a few configs
    python src/evaluate_qformat.py --quick
"""

import argparse
import json
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from itertools import product

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model_base import ReducedGraphSAGE
from train import load_cora_dataset
from config import get_config


def quantize_fixed_point(value: torch.Tensor, total_bits: int, int_bits: int) -> torch.Tensor:
    """
    Quantize tensor to fixed-point Q(total_bits, int_bits) format
    
    Args:
        value: Input tensor
        total_bits: Total bit-width (W)
        int_bits: Integer bit-width (I), fractional = W - I
        
    Returns:
        Quantized tensor (still in float, but limited to Q format precision)
    """
    frac_bits = total_bits - int_bits
    
    # Range: [-2^(I-1), 2^(I-1) - 2^(-F)]
    max_val = 2 ** (int_bits - 1) - 2 ** (-frac_bits)
    min_val = -2 ** (int_bits - 1)
    
    # Clip to range
    value_clipped = torch.clamp(value, min_val, max_val)
    
    # Quantize to fixed-point precision (2^(-F) step size)
    scale = 2 ** frac_bits
    value_quantized = torch.round(value_clipped * scale) / scale
    
    return value_quantized


def evaluate_qformat_config(
    model: ReducedGraphSAGE,
    data, 
    mask,
    data_w: int,
    data_i: int,
    weight_w: int,
    weight_i: int,
    acc_w: int = None,
    acc_i: int = None,
    verbose: bool = False
) -> Dict:
    """
    Evaluate model accuracy with specific Q format configuration
    
    Args:
        model: Trained PyTorch model
        data: Graph data
        mask: Test mask
        data_w, data_i: Q format for activations
        weight_w, weight_i: Q format for weights
        acc_w, acc_i: Q format for accumulators (optional, uses higher precision)
        verbose: Print detailed info
        
    Returns:
        Dict with accuracy, range statistics, and overflow info
    """
    model.eval()
    
    # Quantize model weights
    quantized_state = {}
    weight_overflow_count = 0
    weight_values = []
    
    for name, param in model.state_dict().items():
        if 'weight' in name or 'bias' in name:
            quantized = quantize_fixed_point(param, weight_w, weight_i)
            quantized_state[name] = quantized
            
            # Track overflow (clipping)
            frac_bits = weight_w - weight_i
            max_val = 2 ** (weight_i - 1) - 2 ** (-frac_bits)
            min_val = -2 ** (weight_i - 1)
            overflow = ((param > max_val) | (param < min_val)).sum().item()
            weight_overflow_count += overflow
            weight_values.extend(param.flatten().tolist())
        else:
            quantized_state[name] = param
    
    model.load_state_dict(quantized_state)
    
    # Forward pass with quantized activations
    with torch.no_grad():
        x = data.x
        edge_index = data.edge_index
        
        # Track activation ranges per layer
        activation_stats = []
        
        # Layer 1
        x = model.conv1(x, edge_index)
        x_quantized = quantize_fixed_point(x, data_w, data_i)
        
        # Track overflow
        frac_bits = data_w - data_i
        max_val = 2 ** (data_i - 1) - 2 ** (-frac_bits)
        min_val = -2 ** (data_i - 1)
        overflow_l1 = ((x > max_val) | (x < min_val)).sum().item()
        activation_stats.append({
            'layer': 'conv1',
            'min': x.min().item(),
            'max': x.max().item(),
            'mean': x.mean().item(),
            'std': x.std().item(),
            'overflow_count': overflow_l1,
            'total_elements': x.numel(),
            'qformat': f"Q({data_w},{data_i})"
        })
        
        x = F.relu(x_quantized)
        x = F.dropout(x, p=model.dropout, training=False)
        
        # Layer 2
        x = model.conv2(x, edge_index)
        x_quantized = quantize_fixed_point(x, data_w, data_i)
        
        overflow_l2 = ((x > max_val) | (x < min_val)).sum().item()
        activation_stats.append({
            'layer': 'conv2',
            'min': x.min().item(),
            'max': x.max().item(),
            'mean': x.mean().item(),
            'std': x.std().item(),
            'overflow_count': overflow_l2,
            'total_elements': x.numel(),
            'qformat': f"Q({data_w},{data_i})"
        })
        
        logits = F.log_softmax(x_quantized, dim=1)
        pred = logits[mask].max(1)[1]
        correct = pred.eq(data.y[mask]).sum().item()
        total = mask.sum().item()
        accuracy = correct / total
    
    result = {
        'config': {
            'data': f"Q({data_w},{data_i})",
            'weight': f"Q({weight_w},{weight_i})",
            'data_w': data_w,
            'data_i': data_i,
            'weight_w': weight_w,
            'weight_i': weight_i,
        },
        'accuracy': accuracy,
        'weight_stats': {
            'min': min(weight_values),
            'max': max(weight_values),
            'overflow_count': weight_overflow_count,
            'total_params': len(weight_values),
        },
        'activation_stats': activation_stats,
        'total_overflow': weight_overflow_count + overflow_l1 + overflow_l2,
    }
    
    if verbose:
        print(f"\nQ Format: Data={result['config']['data']}, Weight={result['config']['weight']}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Weight range: [{result['weight_stats']['min']:.4f}, {result['weight_stats']['max']:.4f}]")
        print(f"  Weight overflows: {weight_overflow_count}")
        for stat in activation_stats:
            print(f"  {stat['layer']}: range=[{stat['min']:.2f}, {stat['max']:.2f}], "
                  f"overflows={stat['overflow_count']}/{stat['total_elements']}")
    
    return result


def sweep_qformats(
    model: ReducedGraphSAGE,
    data,
    mask,
    data_w_list: List[int],
    data_i_list: List[int],
    weight_w_list: List[int] = None,
    weight_i_list: List[int] = None,
    output_json: str = None
) -> List[Dict]:
    """
    Sweep multiple Q format configurations and compare
    
    Returns:
        List of results sorted by accuracy
    """
    # Default weight configs if not specified
    if weight_w_list is None:
        weight_w_list = data_w_list
    if weight_i_list is None:
        weight_i_list = data_i_list
    
    results = []
    total_configs = len(data_w_list) * len(data_i_list) * len(weight_w_list) * len(weight_i_list)
    
    print(f"Evaluating {total_configs} Q format configurations...")
    print(f"  Data: W={data_w_list}, I={data_i_list}")
    print(f"  Weight: W={weight_w_list}, I={weight_i_list}")
    print()
    
    for i, (dw, di, ww, wi) in enumerate(product(data_w_list, data_i_list, weight_w_list, weight_i_list)):
        # Skip invalid configs (integer bits must be less than total bits)
        if di >= dw or wi >= ww:
            continue
        
        # Skip configs with too few fractional bits (< 4 bits)
        if (dw - di) < 4 or (ww - wi) < 4:
            continue
        
        print(f"[{i+1}/{total_configs}] Evaluating Data=Q({dw},{di}), Weight=Q({ww},{wi})...")
        result = evaluate_qformat_config(model, data, mask, dw, di, ww, wi, verbose=False)
        results.append(result)
        
        print(f"  Accuracy: {result['accuracy']:.4f}, Overflows: {result['total_overflow']}")
    
    # Sort by accuracy (descending)
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    # Save to JSON if requested
    if output_json:
        output_path = PROJECT_ROOT / output_json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    return results


def print_summary(results: List[Dict], top_n: int = 10):
    """Print summary of top configurations"""
    print("\n" + "="*80)
    print(f"TOP {top_n} CONFIGURATIONS (by accuracy)")
    print("="*80)
    print(f"{'Rank':<6}{'Data Q':<12}{'Weight Q':<12}{'Accuracy':<12}{'Overflows':<12}{'Total Bits':<12}")
    print("-"*80)
    
    for i, result in enumerate(results[:top_n]):
        cfg = result['config']
        total_bits = cfg['data_w'] + cfg['weight_w']
        print(f"{i+1:<6}{cfg['data']:<12}{cfg['weight']:<12}"
              f"{result['accuracy']:<12.4f}{result['total_overflow']:<12}"
              f"{total_bits:<12}")
    
    print("\n" + "="*80)
    print("PARETO OPTIMAL CONFIGURATIONS (accuracy vs size)")
    print("="*80)
    
    # Find Pareto front: maximize accuracy, minimize total bits
    pareto = []
    for r in results:
        total_bits = r['config']['data_w'] + r['config']['weight_w']
        is_pareto = True
        for other in results:
            other_bits = other['config']['data_w'] + other['config']['weight_w']
            # Dominated if: other has better accuracy AND smaller/equal size
            # OR other has equal accuracy AND strictly smaller size
            if (other['accuracy'] > r['accuracy'] and other_bits <= total_bits) or \
               (other['accuracy'] == r['accuracy'] and other_bits < total_bits):
                is_pareto = False
                break
        if is_pareto:
            r['total_bits'] = total_bits
            pareto.append(r)
    
    # Sort Pareto front by total bits
    pareto.sort(key=lambda x: x['total_bits'])
    
    print(f"{'Config':<25}{'Accuracy':<12}{'Overflows':<12}{'Total Bits':<12}")
    print("-"*80)
    for r in pareto:
        cfg = r['config']
        config_str = f"D={cfg['data']}, W={cfg['weight']}"
        print(f"{config_str:<25}{r['accuracy']:<12.4f}{r['total_overflow']:<12}{r['total_bits']:<12}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate fixed-point Q format configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--model-path', type=str,
                        default='pretrained_models/reduced_24.pth',
                        help='Path to trained model')
    parser.add_argument('--data-w', type=int, nargs='+',
                        default=[16, 24, 32],
                        help='Total bits for activations')
    parser.add_argument('--data-i', type=int, nargs='+',
                        default=[8, 12, 16],
                        help='Integer bits for activations')
    parser.add_argument('--weight-w', type=int, nargs='+',
                        default=None,
                        help='Total bits for weights (default: same as data-w)')
    parser.add_argument('--weight-i', type=int, nargs='+',
                        default=None,
                        help='Integer bits for weights (default: same as data-i)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with 3 configs only')
    parser.add_argument('--output', '-o', type=str,
                        default='build/qformat_evaluation.json',
                        help='Output JSON file for results')
    parser.add_argument('--top-n', type=int, default=10,
                        help='Number of top configurations to show')
    
    args = parser.parse_args()
    
    # Quick mode: just test a few representative configs
    if args.quick:
        args.data_w = [16, 24, 32]
        args.data_i = [8, 12, 16]
        args.weight_w = [16, 24]
        args.weight_i = [4, 8]
    
    # Load model
    config = get_config()
    model_path = PROJECT_ROOT / args.model_path
    
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        print("Train a model first:")
        print("  python src/train.py --in-channels 16 --hidden 24")
        return 1
    
    print(f"Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Get model architecture from checkpoint
    in_channels = checkpoint.get('in_channels_reduced', 16)
    hidden_channels = checkpoint.get('hidden_channels', 24)
    
    model = ReducedGraphSAGE(
        in_channels_reduced=in_channels,
        hidden_channels=hidden_channels,
        num_classes=7,
        dropout=0.5
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  Architecture: {in_channels} → {hidden_channels} → 7")
    print(f"  Baseline accuracy: {checkpoint.get('test_accuracy', 'N/A')}")
    
    # Load data
    print("\nLoading Cora dataset...")
    data = load_cora_dataset(root=str(PROJECT_ROOT / 'data'))
    
    # Sweep Q formats
    results = sweep_qformats(
        model=model,
        data=data,
        mask=data.test_mask,
        data_w_list=args.data_w,
        data_i_list=args.data_i,
        weight_w_list=args.weight_w,
        weight_i_list=args.weight_i,
        output_json=args.output
    )
    
    # Print summary
    print_summary(results, top_n=args.top_n)
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    # Find best configurations for different objectives
    best_accuracy = results[0]
    best_size = min(results, key=lambda x: x['config']['data_w'] + x['config']['weight_w'])
    
    # Find best "balanced" config: high accuracy, low size
    # Use weighted score: 70% accuracy, 30% normalized size
    max_bits = max(r['config']['data_w'] + r['config']['weight_w'] for r in results)
    for r in results:
        total_bits = r['config']['data_w'] + r['config']['weight_w']
        r['score'] = 0.7 * r['accuracy'] + 0.3 * (1 - total_bits / max_bits)
    best_balanced = max(results, key=lambda x: x['score'])
    
    print("\n1. Best Accuracy:")
    cfg = best_accuracy['config']
    print(f"   Data={cfg['data']}, Weight={cfg['weight']}")
    print(f"   Accuracy: {best_accuracy['accuracy']:.4f}")
    print(f"   Total bits: {cfg['data_w'] + cfg['weight_w']}")
    
    print("\n2. Smallest Size:")
    cfg = best_size['config']
    print(f"   Data={cfg['data']}, Weight={cfg['weight']}")
    print(f"   Accuracy: {best_size['accuracy']:.4f}")
    print(f"   Total bits: {cfg['data_w'] + cfg['weight_w']}")
    
    print("\n3. Best Balanced (accuracy vs size):")
    cfg = best_balanced['config']
    print(f"   Data={cfg['data']}, Weight={cfg['weight']}")
    print(f"   Accuracy: {best_balanced['accuracy']:.4f}")
    print(f"   Total bits: {cfg['data_w'] + cfg['weight_w']}")
    print(f"   Score: {best_balanced['score']:.4f}")
    
    print("\n4. To use in design space exploration:")
    print("   Edit configs/design_space.yaml:")
    print(f"     fixed:")
    print(f"       data_w: [{cfg['data_w']}]")
    print(f"       data_i: [{cfg['data_i']}]")
    print(f"       weight_w: [{cfg['weight_w']}]")
    print(f"       weight_i: [{cfg['weight_i']}]")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
