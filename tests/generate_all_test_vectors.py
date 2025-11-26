#!/usr/bin/env python3
"""
Unified Test Vector Generator

Consolidates all test vector generation into one script.
Generates test vectors for:
- float: Float baseline (no quantization)
- ptq_float: PTQ with float quant/dequant ops  
- ptq_int8: PTQ with integer-only fixed-point ops
- qat: Quantization-aware training

Usage:
    python generate_all_test_vectors.py --variant float
    python generate_all_test_vectors.py --variant ptq_float
    python generate_all_test_vectors.py --variant ptq_int8
    python generate_all_test_vectors.py --all

Output directories:
    build/test_vectors_float/
    build/test_vectors_ptq_float/
    build/test_vectors_ptq_int8/
    build/test_vectors_qat/
"""

import argparse
import sys

print("="*80)
print("UNIFIED TEST VECTOR GENERATOR")
print("="*80)
print("\nThis script consolidates:")
print("  - generate_test_vectors_float.py → --variant float")
print("  - generate_test_vectors.py       → --variant ptq_float")
print("  - integer_ptq_emulator.py         → --variant ptq_int8")
print("  - (QAT generator)                 → --variant qat")
print("\nPlease use the appropriate --variant flag")
print("="*80)

parser = argparse.ArgumentParser(description='Generate test vectors (unified)')
parser.add_argument('--variant', type=str, choices=['float', 'ptq_float', 'ptq_int8', 'qat'],
                   help='Which variant to generate')
parser.add_argument('--all', action='store_true',
                   help='Generate all variants')

args = parser.parse_args()

if not args.variant and not args.all:
    parser.print_help()
    print("\nFor now, please use the individual scripts:")
    print("  Float:     python generate_test_vectors_float.py")
    print("  PTQ Float: python generate_test_vectors.py  (or generate_ptq_test_vectors_clean.py)")
    print("  PTQ Int8:  python integer_ptq_emulator.py")
    sys.exit(1)

# TODO: Implement unified generator
print("\n⚠ Unified generator not fully implemented yet.")
print("Please use individual scripts for now.")
print("\nAfter refactoring is complete, this will be the single entry point.")
