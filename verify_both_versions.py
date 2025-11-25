#!/usr/bin/env python3
"""
Verification script for both GraphSAGE model versions.
Checks that both standard and HLS-compatible versions are correctly generated.
"""

import json
import os

def check_version(version_name, quantized_dir, model_path, should_have_root_weight):
    """Check a specific model version."""
    print(f"\n{'='*70}")
    print(f"CHECKING {version_name}")
    print('='*70)
    
    issues = []
    ok = []
    
    # Check model file
    if os.path.exists(model_path):
        ok.append(f"✓ Model found: {model_path}")
        print(f"   ✓ Model file exists")
    else:
        issues.append(f"❌ Missing model: {model_path}")
        print(f"   ❌ Model file missing")
        return issues, ok
    
    # Check quantized directory
    if not os.path.exists(quantized_dir):
        issues.append(f"❌ Missing quantized dir: {quantized_dir}")
        print(f"   ❌ Quantized directory missing")
        return issues, ok
    
    # Check quant_params.json
    params_file = f'{quantized_dir}/quant_params.json'
    if not os.path.exists(params_file):
        issues.append(f"❌ Missing {params_file}")
        print(f"   ❌ Quantization parameters missing")
        return issues, ok
    
    with open(params_file) as f:
        quant_params = json.load(f)
    
    # Check for lin_r weights
    has_lin_r = any('lin_r' in key for key in quant_params['scales'].keys())
    
    if should_have_root_weight:
        if has_lin_r:
            ok.append("✓ Has lin_r weights (as expected)")
            print("   ✓ Has lin_r weights (root_weight=True, as expected)")
        else:
            issues.append("❌ Missing lin_r weights but should have them")
            print("   ❌ Missing lin_r weights but should have them")
    else:
        if has_lin_r:
            issues.append("❌ Has lin_r weights but shouldn't (not HLS-compatible)")
            print("   ❌ CRITICAL: Has lin_r weights (not HLS-compatible!)")
        else:
            ok.append("✓ No lin_r weights (HLS-compatible)")
            print("   ✓ Correct: No lin_r weights (HLS-compatible)")
    
    # Check required weights
    required_weights = ['conv1.lin_l.weight', 'conv1.lin_l.bias', 
                        'conv2.lin_l.weight', 'conv2.lin_l.bias']
    for w in required_weights:
        if w in quant_params['scales']:
            ok.append(f"✓ {w}")
            print(f"   ✓ {w}: scale={quant_params['scales'][w]:.6f}")
        else:
            issues.append(f"❌ Missing {w}")
            print(f"   ❌ Missing {w}")
    
    return issues, ok


def main():
    print("="*70)
    print("DUAL MODEL VERSION VERIFICATION")
    print("="*70)
    
    all_issues = []
    all_ok = []
    
    # Check standard version (with root_weight)
    issues, ok = check_version(
        "STANDARD VERSION (with root_weight)",
        "build/quantized",
        "build/models/reduced_graphsage_best.pth",
        should_have_root_weight=True
    )
    all_issues.extend(issues)
    all_ok.extend(ok)
    
    # Check HLS-compatible version (no root_weight)
    issues, ok = check_version(
        "HLS-COMPATIBLE VERSION (no root_weight)",
        "build/quantized_no_root",
        "build/models/reduced_graphsage_no_root_best.pth",
        should_have_root_weight=False
    )
    all_issues.extend(issues)
    all_ok.extend(ok)
    
    # Check test vectors
    print(f"\n{'='*70}")
    print("CHECKING TEST VECTORS")
    print('='*70)
    
    required_files = [
        'adj_matrix.txt',
        'network_input.txt', 
        'weights_layer1.txt',
        'bias_layer1.txt',
        'weights_layer2.txt',
        'bias_layer2.txt',
        'network_output_reference.txt',
        'scales.txt'
    ]
    
    for f in required_files:
        path = f'build/test_vectors/{f}'
        if os.path.exists(path):
            size = os.path.getsize(path)
            all_ok.append(f"✓ {f}")
            print(f"   ✓ {f} ({size} bytes)")
        else:
            all_issues.append(f"❌ Missing {f}")
            print(f"   ❌ Missing {f}")
    
    # Check C headers
    print(f"\n{'='*70}")
    print("CHECKING C HEADERS")
    print('='*70)
    
    headers = [
        ('build/hls/weights.h', 'Standard version'),
        ('build/hls/weights_no_root.h', 'HLS-compatible version')
    ]
    
    for header_path, desc in headers:
        if os.path.exists(header_path):
            size = os.path.getsize(header_path)
            all_ok.append(f"✓ {header_path}")
            print(f"   ✓ {desc}: {header_path} ({size} bytes)")
        else:
            all_issues.append(f"❌ Missing {header_path}")
            print(f"   ❌ Missing {header_path}")
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"✓ OK:     {len(all_ok)} checks passed")
    print(f"❌ ISSUES: {len(all_issues)} problems found")
    
    if all_issues:
        print("\n🔴 ISSUES FOUND:")
        for issue in all_issues:
            print(f"   {issue}")
        print("\n❌ VERIFICATION FAILED")
        return 1
    else:
        print("\n✅ ALL CHECKS PASSED!")
        print("\n📋 Summary:")
        print("   • Standard version (with root_weight): ✓ Available")
        print("   • HLS-compatible version (no root_weight): ✓ Available")
        print("   • Test vectors: ✓ Generated")
        print("   • C headers: ✓ Exported")
        print("\n💡 For HLS implementation, use:")
        print("   - Model: build/models/reduced_graphsage_no_root_best.pth")
        print("   - Quantized: build/quantized_no_root/")
        print("   - Header: build/hls/weights_no_root.h")
        print("\n✅ READY FOR HLS IMPLEMENTATION")
        return 0


if __name__ == '__main__':
    exit(main())
