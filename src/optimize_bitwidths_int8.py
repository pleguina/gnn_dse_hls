"""
Bitwidth Optimization for INT8 GraphSAGE HLS Implementation

This script analyzes the PTQ model and graph data to compute optimal
bit-widths for all intermediate types in the pure-INT8 HLS implementation.

Methods:
1. THEORETICAL: Worst-case bounds based on model dimensions
2. DATA-DRIVEN: Actual numerical maxima from integer simulation

Output:
- auto_generated_bitwidths.h: HLS header with optimal type definitions
- bitwidth_analysis.json: Detailed analysis report

Reference: docs/notes/bitsize_optimization_int8ptq.txt
"""

import json
import numpy as np
import argparse
from pathlib import Path
import math

# ============================================================================
# Configuration
# ============================================================================

parser = argparse.ArgumentParser(description='Optimize bit-widths for INT8 HLS')
parser.add_argument('--safety-margin', type=int, default=2,
                    help='Safety margin bits to add (default: 2)')
parser.add_argument('--method', choices=['theoretical', 'data-driven', 'both'],
                    default='both', help='Optimization method')
parser.add_argument('--m-bits', type=int, default=24,
                    help='Fixed-point fractional bits M (default: 24)')
parser.add_argument('--hidden-channels', type=int, default=24,
                    help='Hidden layer dimension (default: 24)')
parser.add_argument('--in-channels', type=int, default=16,
                    help='Input layer dimension (default: 16)')
args = parser.parse_args()

SAFETY_MARGIN = args.safety_margin
M_BITS = args.m_bits
HIDDEN_CHANNELS = args.hidden_channels
IN_CHANNELS = args.in_channels

# Get project root (parent of src directory)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Architecture-specific subdirectory
ARCH_DIR = f"{IN_CHANNELS}x{HIDDEN_CHANNELS}"

# Paths (using absolute paths) - now architecture-specific
INT8_DIR = PROJECT_ROOT / "build/weights_ptq_per_arch" / ARCH_DIR
PTQ_DIR = PROJECT_ROOT / "build/weights_ptq_per_arch" / ARCH_DIR
TEST_VECTORS_DIR = PROJECT_ROOT / "build/test_vectors_ptq_float"
OUTPUT_DIR = PROJECT_ROOT / "build/hls"
HLS_DIR = PROJECT_ROOT / "hls"  # Also copy to HLS source directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("BITWIDTH OPTIMIZATION FOR INT8 GRAPHSAGE HLS")
print("=" * 80)
print(f"\nConfiguration:")
print(f"  Architecture: {IN_CHANNELS} → {HIDDEN_CHANNELS} → 7")
print(f"  Safety margin: {SAFETY_MARGIN} bits")
print(f"  M_BITS: {M_BITS}")
print(f"  Method: {args.method}")

# ============================================================================
# Step 1: Extract Graph Statistics
# ============================================================================

print("\n" + "-" * 80)
print("Step 1: Extract Graph Statistics")
print("-" * 80)

# Load adjacency matrix
adj_float = np.loadtxt(TEST_VECTORS_DIR / "adj_matrix.txt", dtype=np.float32)
NUM_NODES = adj_float.shape[0]

# Compute degree per node
degrees = (adj_float > 0).sum(axis=1)
deg_max = int(degrees.max())

# Get max adjacency value
A_float_max = float(np.abs(adj_float).max())

# Load int8 params to get K
with open(INT8_DIR / "int8_params.json", 'r') as f:
    params = json.load(f)
K = params['fixed_point_config']['K']
K_BITS = params['fixed_point_config']['K_BITS']

# Compute integer adjacency max
adj_int16 = np.loadtxt(INT8_DIR / "adj_matrix_int16.txt", dtype=np.int16)
A_int_max = int(np.abs(adj_int16).max())

print(f"\nGraph statistics:")
print(f"  NUM_NODES: {NUM_NODES}")
print(f"  deg_max (max node degree): {deg_max}")
print(f"  A_float_max: {A_float_max:.6f}")
print(f"  K (adjacency scale): {K} (2^{K_BITS})")
print(f"  A_int_max: {A_int_max}")

# ============================================================================
# Step 2: Extract Model Parameters
# ============================================================================

print("\n" + "-" * 80)
print("Step 2: Extract Model Parameters")
print("-" * 80)

# Load INT8 weights - use architecture parameters
weights1_int8 = np.loadtxt(PTQ_DIR / "conv1_lin_l_weight.txt", dtype=np.int8).reshape(HIDDEN_CHANNELS, IN_CHANNELS)
weights2_int8 = np.loadtxt(PTQ_DIR / "conv2_lin_l_weight.txt", dtype=np.int8).reshape(7, HIDDEN_CHANNELS)

# Load INT32 biases
bias1_int32 = np.loadtxt(INT8_DIR / "bias_layer1_int32.txt", dtype=np.int32)
bias2_int32 = np.loadtxt(INT8_DIR / "bias_layer2_int32.txt", dtype=np.int32)

# Architecture dimensions
IN_FEATURES = weights1_int8.shape[1]
HIDDEN_FEATURES = weights1_int8.shape[0]
OUT_FEATURES = weights2_int8.shape[0]

# Weight/bias ranges
w1_max = int(np.abs(weights1_int8).max())
w2_max = int(np.abs(weights2_int8).max())
b1_max = int(np.abs(bias1_int32).max())
b2_max = int(np.abs(bias2_int32).max())

feat_max = 127  # INT8 max

print(f"\nModel architecture:")
print(f"  IN_FEATURES: {IN_FEATURES}")
print(f"  HIDDEN_FEATURES: {HIDDEN_FEATURES}")
print(f"  OUT_FEATURES: {OUT_FEATURES}")

print(f"\nWeight ranges (INT8):")
print(f"  Layer 1: max|w1| = {w1_max}")
print(f"  Layer 2: max|w2| = {w2_max}")

print(f"\nBias ranges (INT32):")
print(f"  Layer 1: max|b1| = {b1_max}")
print(f"  Layer 2: max|b2| = {b2_max}")

# Load fixed-point scales
eff_scale1_fp = params['fixed_point_scales']['eff_scale1_fp']
eff_scale2_fp = params['fixed_point_scales']['eff_scale2_fp']
beta1_fp = params['fixed_point_scales']['beta1_fp']
beta2_fp = params['fixed_point_scales']['beta2_fp']

scale_max = max(eff_scale1_fp, eff_scale2_fp, beta1_fp, beta2_fp)

print(f"\nFixed-point scales:")
print(f"  eff_scale1_fp = {eff_scale1_fp}")
print(f"  eff_scale2_fp = {eff_scale2_fp}")
print(f"  beta1_fp = {beta1_fp}")
print(f"  beta2_fp = {beta2_fp}")
print(f"  max scale = {scale_max}")

# ============================================================================
# Step 3: Theoretical Bit-Width Calculation
# ============================================================================

print("\n" + "-" * 80)
print("Step 3: Theoretical Bit-Width Calculation (Worst-Case)")
print("-" * 80)

def bits_required(max_val):
    """Calculate bits required to represent a signed integer with given max absolute value"""
    if max_val <= 0:
        return 1
    return int(math.ceil(math.log2(max_val + 1))) + 1  # +1 for sign

# Adjacency bit-width
# |A_int| <= 2^K_BITS, so ADJ_BITS >= K_BITS + 1 + margin
adj_bits_theory = K_BITS + 1 + SAFETY_MARGIN
print(f"\nADJ_BITS (theoretical):")
print(f"  A_int_max <= 2^{K_BITS} = {2**K_BITS}")
print(f"  ADJ_BITS = K_BITS + 1 + margin = {K_BITS} + 1 + {SAFETY_MARGIN} = {adj_bits_theory}")

# Aggregation accumulator bit-width
# |tmp| <= deg_max * A_int_max * feat_max
agg_max_theory = deg_max * A_int_max * feat_max
agg_bits_theory = bits_required(agg_max_theory) + SAFETY_MARGIN
print(f"\nAGG_ACC_BITS (theoretical):")
print(f"  |tmp| <= deg_max * A_int_max * feat_max")
print(f"        <= {deg_max} * {A_int_max} * {feat_max} = {agg_max_theory}")
print(f"  AGG_ACC_BITS = {bits_required(agg_max_theory)} + margin = {agg_bits_theory}")

# Linear accumulator bit-width
# |acc| <= max(|bias_int_max|, F_in * feat_max * w_max)
lin_mac_max1 = IN_FEATURES * feat_max * w1_max
lin_mac_max2 = HIDDEN_FEATURES * feat_max * w2_max
lin_acc_max1 = max(b1_max, lin_mac_max1)
lin_acc_max2 = max(b2_max, lin_mac_max2)
lin_acc_max_theory = max(lin_acc_max1, lin_acc_max2)
lin_bits_theory = bits_required(lin_acc_max_theory) + SAFETY_MARGIN

print(f"\nLIN_ACC_BITS (theoretical):")
print(f"  Layer 1: |acc| <= max(|b1|, F_in*feat*w)")
print(f"                 <= max({b1_max}, {IN_FEATURES}*{feat_max}*{w1_max}) = max({b1_max}, {lin_mac_max1})")
print(f"                 = {lin_acc_max1}")
print(f"  Layer 2: |acc| <= max(|b2|, F_in*feat*w)")
print(f"                 <= max({b2_max}, {HIDDEN_FEATURES}*{feat_max}*{w2_max}) = max({b2_max}, {lin_mac_max2})")
print(f"                 = {lin_acc_max2}")
print(f"  LIN_ACC_BITS = {bits_required(lin_acc_max_theory)} + margin = {lin_bits_theory}")

# Combined accumulator (use max of aggregation and linear)
acc_bits_theory = max(agg_bits_theory, lin_bits_theory)
print(f"\nACC_BITS (combined) = max({agg_bits_theory}, {lin_bits_theory}) = {acc_bits_theory}")

# Scale bit-width
# Scales are fixed-point with M fractional bits
scale_bits_theory = M_BITS + 8  # Standard headroom
print(f"\nSCALE_BITS (theoretical):")
print(f"  M_BITS = {M_BITS}")
print(f"  SCALE_BITS = M_BITS + 8 = {scale_bits_theory}")

# Multiplication bit-width
# mult = acc * scale_fp
mult_bits_theory = acc_bits_theory + scale_bits_theory
print(f"\nMULT_BITS (theoretical):")
print(f"  MULT_BITS = ACC_BITS + SCALE_BITS = {acc_bits_theory} + {scale_bits_theory} = {mult_bits_theory}")

# ============================================================================
# Step 4: Data-Driven Bit-Width Calculation
# ============================================================================

print("\n" + "-" * 80)
print("Step 4: Data-Driven Bit-Width Calculation (Actual Maxima)")
print("-" * 80)

# Load input
input_int8 = np.loadtxt(TEST_VECTORS_DIR / "network_input.txt", dtype=np.int8)

def int8_clamp(x):
    return np.clip(x, -128, 127).astype(np.int8)

# Track actual maxima during forward pass
max_agg_tmp = 0
max_lin_acc = 0
max_scaled = 0
max_bias = max(b1_max, b2_max)

# === Layer 1 Aggregation ===
print("\nRunning integer simulation to find actual maxima...")
agg1_int8 = np.zeros((NUM_NODES, IN_FEATURES), dtype=np.int8)

for i in range(NUM_NODES):
    for f in range(IN_FEATURES):
        tmp = np.int32(0)
        for j in range(NUM_NODES):
            if adj_int16[i, j] != 0:
                tmp += np.int32(adj_int16[i, j]) * np.int32(input_int8[j, f])
        max_agg_tmp = max(max_agg_tmp, abs(int(tmp)))
        
        tmp_scaled = np.int64(tmp) * np.int64(beta1_fp)
        tmp_rounded = tmp_scaled + (1 << (M_BITS - 1))
        result = tmp_rounded >> M_BITS
        max_scaled = max(max_scaled, abs(int(result)))
        
        agg1_int8[i, f] = int8_clamp(result)

print(f"  Layer 1 Agg: max|tmp| = {max_agg_tmp}")

# === Layer 1 Linear ===
hidden_int8 = np.zeros((NUM_NODES, HIDDEN_FEATURES), dtype=np.int8)

for n in range(NUM_NODES):
    for o in range(HIDDEN_FEATURES):
        acc = np.int32(bias1_int32[o])
        for f in range(IN_FEATURES):
            acc += np.int32(agg1_int8[n, f]) * np.int32(weights1_int8[o, f])
        max_lin_acc = max(max_lin_acc, abs(int(acc)))
        
        tmp_scaled = np.int64(acc) * np.int64(eff_scale1_fp)
        tmp_rounded = tmp_scaled + (1 << (M_BITS - 1))
        result = tmp_rounded >> M_BITS
        max_scaled = max(max_scaled, abs(int(result)))
        
        hidden_int8[n, o] = int8_clamp(max(0, result))  # ReLU

print(f"  Layer 1 Lin: max|acc| = {max_lin_acc}")

# === Layer 2 Aggregation ===
agg2_int8 = np.zeros((NUM_NODES, HIDDEN_FEATURES), dtype=np.int8)

for i in range(NUM_NODES):
    for f in range(HIDDEN_FEATURES):
        tmp = np.int32(0)
        for j in range(NUM_NODES):
            if adj_int16[i, j] != 0:
                tmp += np.int32(adj_int16[i, j]) * np.int32(hidden_int8[j, f])
        max_agg_tmp = max(max_agg_tmp, abs(int(tmp)))
        
        tmp_scaled = np.int64(tmp) * np.int64(beta2_fp)
        tmp_rounded = tmp_scaled + (1 << (M_BITS - 1))
        result = tmp_rounded >> M_BITS
        max_scaled = max(max_scaled, abs(int(result)))
        
        agg2_int8[i, f] = int8_clamp(result)

print(f"  Layer 2 Agg: max|tmp| = {max_agg_tmp} (cumulative)")

# === Layer 2 Linear ===
output_int8 = np.zeros((NUM_NODES, OUT_FEATURES), dtype=np.int8)

for n in range(NUM_NODES):
    for o in range(OUT_FEATURES):
        acc = np.int32(bias2_int32[o])
        for f in range(HIDDEN_FEATURES):
            acc += np.int32(agg2_int8[n, f]) * np.int32(weights2_int8[o, f])
        max_lin_acc = max(max_lin_acc, abs(int(acc)))
        
        tmp_scaled = np.int64(acc) * np.int64(eff_scale2_fp)
        tmp_rounded = tmp_scaled + (1 << (M_BITS - 1))
        result = tmp_rounded >> M_BITS
        max_scaled = max(max_scaled, abs(int(result)))
        
        output_int8[n, o] = int8_clamp(result)

print(f"  Layer 2 Lin: max|acc| = {max_lin_acc} (cumulative)")
print(f"  max|scaled| (before shift) = {max_scaled}")

# Data-driven bit-widths
adj_bits_data = bits_required(A_int_max) + SAFETY_MARGIN
agg_bits_data = bits_required(max_agg_tmp) + SAFETY_MARGIN
lin_bits_data = bits_required(max_lin_acc) + SAFETY_MARGIN
acc_bits_data = max(agg_bits_data, lin_bits_data)
scale_bits_data = bits_required(scale_max) + SAFETY_MARGIN
mult_bits_data = acc_bits_data + scale_bits_data

print(f"\nData-driven bit-widths:")
print(f"  ADJ_BITS = {adj_bits_data}")
print(f"  AGG_ACC_BITS = {agg_bits_data}")
print(f"  LIN_ACC_BITS = {lin_bits_data}")
print(f"  ACC_BITS (combined) = {acc_bits_data}")
print(f"  SCALE_BITS = {scale_bits_data}")
print(f"  MULT_BITS = {mult_bits_data}")

# ============================================================================
# Step 5: Choose Final Bit-Widths
# ============================================================================

print("\n" + "-" * 80)
print("Step 5: Final Bit-Width Selection")
print("-" * 80)

if args.method == 'theoretical':
    final_adj_bits = adj_bits_theory
    final_acc_bits = acc_bits_theory
    final_scale_bits = scale_bits_theory
    final_mult_bits = mult_bits_theory
    method_used = "theoretical"
elif args.method == 'data-driven':
    final_adj_bits = adj_bits_data
    final_acc_bits = acc_bits_data
    final_scale_bits = scale_bits_data
    final_mult_bits = mult_bits_data
    method_used = "data-driven"
else:  # both - use data-driven but validate against theoretical
    final_adj_bits = adj_bits_data
    final_acc_bits = acc_bits_data
    final_scale_bits = scale_bits_data
    final_mult_bits = mult_bits_data
    method_used = "data-driven (validated against theoretical)"
    
    # Sanity check: data-driven should not exceed theoretical by much
    if acc_bits_data > acc_bits_theory:
        print(f"  WARNING: Data-driven ACC_BITS ({acc_bits_data}) > theoretical ({acc_bits_theory})")

print(f"\nMethod used: {method_used}")
print(f"\nFinal bit-widths:")
print(f"  K_BITS     = {K_BITS}")
print(f"  M_BITS     = {M_BITS}")
print(f"  ADJ_BITS   = {final_adj_bits}")
print(f"  ACC_BITS   = {final_acc_bits}")
print(f"  SCALE_BITS = {final_scale_bits}")
print(f"  MULT_BITS  = {final_mult_bits}")

# Compare with default (conservative) values
default_adj = K_BITS + 4  # 16
default_acc = 32
default_scale = M_BITS + 8  # 32
default_mult = default_acc + default_scale  # 64

print(f"\nComparison with defaults:")
print(f"  ADJ_BITS:   {final_adj_bits} vs default {default_adj} (savings: {default_adj - final_adj_bits} bits)")
print(f"  ACC_BITS:   {final_acc_bits} vs default {default_acc} (savings: {default_acc - final_acc_bits} bits)")
print(f"  SCALE_BITS: {final_scale_bits} vs default {default_scale} (savings: {default_scale - final_scale_bits} bits)")
print(f"  MULT_BITS:  {final_mult_bits} vs default {default_mult} (savings: {default_mult - final_mult_bits} bits)")

# ============================================================================
# Step 6: Generate HLS Header
# ============================================================================

print("\n" + "-" * 80)
print("Step 6: Generate HLS Header")
print("-" * 80)

header_content = f"""/**
 * AUTO-GENERATED BITWIDTH CONFIGURATION FOR INT8 GRAPHSAGE HLS
 * 
 * Generated by: src/optimize_bitwidths_int8.py
 * Method: {method_used}
 * Safety margin: {SAFETY_MARGIN} bits
 * 
 * DO NOT EDIT MANUALLY - regenerate using the optimization script.
 * 
 * To regenerate:
 *   cd src
 *   python optimize_bitwidths_int8.py --safety-margin {SAFETY_MARGIN} --method {args.method}
 */

#ifndef AUTO_GENERATED_BITWIDTHS_H
#define AUTO_GENERATED_BITWIDTHS_H

// ============================================================================
// Model Architecture (extracted from PTQ model)
// ============================================================================

#define OPT_NUM_NODES       {NUM_NODES}
#define OPT_IN_FEATURES     {IN_FEATURES}
#define OPT_HIDDEN_FEATURES {HIDDEN_FEATURES}
#define OPT_OUT_FEATURES    {OUT_FEATURES}

// ============================================================================
// Fixed-Point Configuration
// ============================================================================

// Fractional bits for scale factors
#define OPT_M_BITS {M_BITS}

// Adjacency matrix scale: A_float * 2^K_BITS -> adj_t
#define OPT_K_BITS {K_BITS}

// ============================================================================
// Optimized Bit-Widths
// ============================================================================

// Adjacency matrix entries (scaled integers)
// Theoretical max: 2^{K_BITS} = {2**K_BITS}
// Actual max: {A_int_max}
#define OPT_ADJ_BITS {final_adj_bits}

// Accumulator for MAC operations (aggregation and linear)
// Theoretical max: {max(agg_max_theory, lin_acc_max_theory)}
// Actual max: {max(max_agg_tmp, max_lin_acc)}
#define OPT_ACC_BITS {final_acc_bits}

// Fixed-point scale factors (eff_scale_fp, beta_fp)
// Max scale value: {scale_max}
#define OPT_SCALE_BITS {final_scale_bits}

// Intermediate product (acc * scale)
#define OPT_MULT_BITS {final_mult_bits}

// ============================================================================
// Graph Statistics (for reference)
// ============================================================================

#define OPT_DEG_MAX {deg_max}  // Maximum node degree

// ============================================================================
// Fixed-Point Scale Values (for testbench reference)
// ============================================================================

#define OPT_EFF_SCALE1_FP {eff_scale1_fp}
#define OPT_EFF_SCALE2_FP {eff_scale2_fp}
#define OPT_BETA1_FP      {beta1_fp}
#define OPT_BETA2_FP      {beta2_fp}

// ============================================================================
// Analysis Report
// ============================================================================

/*
 * Theoretical Analysis (worst-case bounds):
 *   ADJ_BITS:   {adj_bits_theory}
 *   ACC_BITS:   {acc_bits_theory}
 *   SCALE_BITS: {scale_bits_theory}
 *   MULT_BITS:  {mult_bits_theory}
 *
 * Data-Driven Analysis (actual simulation):
 *   ADJ_BITS:   {adj_bits_data}
 *   ACC_BITS:   {acc_bits_data}
 *   SCALE_BITS: {scale_bits_data}
 *   MULT_BITS:  {mult_bits_data}
 *
 * Savings vs defaults (32-bit acc, 64-bit mult):
 *   ADJ:   {default_adj - final_adj_bits} bits saved
 *   ACC:   {default_acc - final_acc_bits} bits saved
 *   SCALE: {default_scale - final_scale_bits} bits saved
 *   MULT:  {default_mult - final_mult_bits} bits saved
 */

#endif // AUTO_GENERATED_BITWIDTHS_H
"""

header_path = OUTPUT_DIR / "auto_generated_bitwidths.h"
with open(header_path, 'w') as f:
    f.write(header_content)

print(f"✓ Generated: {header_path}")

# Also copy to HLS source directory for direct use
import shutil
hls_header_path = HLS_DIR / "auto_generated_bitwidths.h"
shutil.copy(header_path, hls_header_path)
print(f"✓ Copied to: {hls_header_path}")

# ============================================================================
# Step 7: Generate Analysis Report
# ============================================================================

print("\n" + "-" * 80)
print("Step 7: Generate Analysis Report")
print("-" * 80)

analysis_report = {
    "configuration": {
        "safety_margin": SAFETY_MARGIN,
        "m_bits": M_BITS,
        "k_bits": K_BITS,
        "method": args.method
    },
    "model_architecture": {
        "num_nodes": NUM_NODES,
        "in_features": IN_FEATURES,
        "hidden_features": HIDDEN_FEATURES,
        "out_features": OUT_FEATURES
    },
    "graph_statistics": {
        "deg_max": deg_max,
        "A_float_max": A_float_max,
        "A_int_max": A_int_max
    },
    "model_parameters": {
        "w1_max": w1_max,
        "w2_max": w2_max,
        "b1_max": b1_max,
        "b2_max": b2_max,
        "feat_max": feat_max
    },
    "fixed_point_scales": {
        "eff_scale1_fp": eff_scale1_fp,
        "eff_scale2_fp": eff_scale2_fp,
        "beta1_fp": beta1_fp,
        "beta2_fp": beta2_fp,
        "scale_max": scale_max
    },
    "theoretical_analysis": {
        "agg_max": agg_max_theory,
        "lin_acc_max": lin_acc_max_theory,
        "adj_bits": adj_bits_theory,
        "acc_bits": acc_bits_theory,
        "scale_bits": scale_bits_theory,
        "mult_bits": mult_bits_theory
    },
    "data_driven_analysis": {
        "max_agg_tmp": max_agg_tmp,
        "max_lin_acc": max_lin_acc,
        "max_scaled": max_scaled,
        "adj_bits": adj_bits_data,
        "acc_bits": acc_bits_data,
        "scale_bits": scale_bits_data,
        "mult_bits": mult_bits_data
    },
    "final_bitwidths": {
        "k_bits": K_BITS,
        "m_bits": M_BITS,
        "adj_bits": final_adj_bits,
        "acc_bits": final_acc_bits,
        "scale_bits": final_scale_bits,
        "mult_bits": final_mult_bits
    },
    "savings_vs_defaults": {
        "adj_bits_saved": default_adj - final_adj_bits,
        "acc_bits_saved": default_acc - final_acc_bits,
        "scale_bits_saved": default_scale - final_scale_bits,
        "mult_bits_saved": default_mult - final_mult_bits
    }
}

report_path = OUTPUT_DIR / "bitwidth_analysis.json"
with open(report_path, 'w') as f:
    json.dump(analysis_report, f, indent=2)

print(f"✓ Generated: {report_path}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\nGenerated files:")
print(f"  ✓ {header_path}")
print(f"  ✓ {report_path}")

print(f"\nOptimized bit-widths (method: {method_used}):")
print(f"  K_BITS     = {K_BITS:3d}  (adjacency fixed-point scale)")
print(f"  M_BITS     = {M_BITS:3d}  (scale factor fractional bits)")
print(f"  ADJ_BITS   = {final_adj_bits:3d}  (adjacency matrix entries)")
print(f"  ACC_BITS   = {final_acc_bits:3d}  (MAC accumulators)")
print(f"  SCALE_BITS = {final_scale_bits:3d}  (fixed-point scales)")
print(f"  MULT_BITS  = {final_mult_bits:3d}  (scaled products)")

print(f"\nTo use in HLS:")
print(f"  1. Include auto_generated_bitwidths.h in your HLS design")
print(f"  2. Use OPT_* macros instead of hardcoded values")
print(f"  3. Or pass -D flags to override defaults:")
print(f"     -DADJ_BITS={final_adj_bits} -DACC_BITS={final_acc_bits} ...")

print(f"\nEstimated resource savings:")
total_bits_saved = (default_adj - final_adj_bits) + (default_acc - final_acc_bits) + \
                   (default_scale - final_scale_bits) + (default_mult - final_mult_bits)
print(f"  Total: ~{total_bits_saved} bits narrower types")
print(f"  This can reduce DSP usage and LUT consumption significantly")

print("\n" + "=" * 80)
