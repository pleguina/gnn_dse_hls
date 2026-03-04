#!/usr/bin/env python3
"""
generate_hardcoded_sparse.py - Auto-generate hardcoded sparse HLS code

This script takes an adjacency matrix and generates optimal HLS code where
the sparsity pattern is baked into hardware at compile time.

Benefits:
- Zero wasted multipliers (only actual edges get hardware)
- No conditionals (pure combinational datapath)  
- II=1 guaranteed (fully deterministic)
- 50-90% DSP reduction for sparse graphs

Usage:
    python generate_hardcoded_sparse.py --adj-file path/to/adj.txt --features 16
"""

import numpy as np
import argparse
from pathlib import Path


def analyze_graph(adj: np.ndarray) -> dict:
    """Analyze adjacency matrix statistics."""
    N = adj.shape[0]
    edges = []
    
    for i in range(N):
        for j in range(N):
            if adj[i, j] != 0:
                edges.append((i, j, adj[i, j]))
    
    degrees = [sum(1 for e in edges if e[0] == i) for i in range(N)]
    
    return {
        'num_nodes': N,
        'num_edges': len(edges),
        'edges': edges,
        'degrees': degrees,
        'max_degree': max(degrees),
        'sparsity': 1 - len(edges) / (N * N)
    }


def generate_aggregate_function(stats: dict, feature_dim: int, 
                                  func_suffix: str = "") -> str:
    """Generate hardcoded sparse aggregation function."""
    N = stats['num_nodes']
    edges = stats['edges']
    
    # Group edges by source node
    node_edges = {i: [] for i in range(N)}
    for src, dst, weight in edges:
        node_edges[src].append((dst, weight))
    
    # Calculate DSP usage
    dense_dsps = N * N * feature_dim
    sparse_dsps = len(edges) * feature_dim
    
    code = f'''
//==============================================================================
// Hardcoded Sparse Aggregation - {feature_dim} features
// 
// Graph: {N} nodes, {len(edges)} edges, sparsity={stats['sparsity']*100:.1f}%
// DSP: {sparse_dsps} multipliers (vs {dense_dsps} dense) → {(1-sparse_dsps/dense_dsps)*100:.1f}% savings
//==============================================================================

inline void aggregate_sparse_{feature_dim}{func_suffix}(
    const data_t x[{N}][{feature_dim}],
    data_t out[{N}][{feature_dim}]
) {{
    #pragma HLS INLINE
    #pragma HLS ARRAY_PARTITION variable=x complete dim=0
    #pragma HLS ARRAY_PARTITION variable=out complete dim=0
    
    AGG_FEAT: for (int f = 0; f < {feature_dim}; f++) {{
        #pragma HLS UNROLL
'''
    
    # Generate code for each node
    for i in range(N):
        neighbors = node_edges[i]
        degree = len(neighbors)
        
        code += f'''        
        // Node {i}: {degree} neighbor{"s" if degree != 1 else ""}\n'''
        
        if degree == 0:
            code += f'        out[{i}][f] = (data_t)0;\n'
        elif degree == 1:
            dst, weight = neighbors[0]
            code += f'        out[{i}][f] = (data_t)((acc_t){weight:.6f}f * (acc_t)x[{dst}][f]);\n'
        else:
            terms = []
            for dst, weight in neighbors:
                terms.append(f'(acc_t){weight:.6f}f * (acc_t)x[{dst}][f]')
            
            # Format with line breaks for readability
            if len(terms) <= 2:
                joined = " + ".join(terms)
                code += f'        out[{i}][f] = (data_t)({joined});\n'
            else:
                nl = '\n'
                code += f'        out[{i}][f] = (data_t)({nl}'
                for j, term in enumerate(terms):
                    if j < len(terms) - 1:
                        code += f'                             {term} +{nl}'
                    else:
                        code += f'                             {term});{nl}'
    
    code += '''    }
}
'''
    
    return code


def generate_header_file(adj: np.ndarray, feature_dims: list, 
                          hidden_dim: int, out_features: int) -> str:
    """Generate complete HLS header file with hardcoded sparse aggregation."""
    
    stats = analyze_graph(adj)
    N = stats['num_nodes']
    
    # Calculate total DSP comparison
    in_features = feature_dims[0]
    dense_agg1 = N * N * in_features
    dense_agg2 = N * N * hidden_dim
    sparse_agg1 = stats['num_edges'] * in_features
    sparse_agg2 = stats['num_edges'] * hidden_dim
    linear1 = N * in_features * hidden_dim
    linear2 = N * hidden_dim * out_features
    
    dense_total = dense_agg1 + linear1 + dense_agg2 + linear2
    sparse_total = sparse_agg1 + linear1 + sparse_agg2 + linear2
    
    header = f'''/*******************************************************************************
 * graphsage_hardcoded_sparse_generated.h - Auto-Generated Sparse GraphSAGE
 * 
 * Generated from adjacency matrix with compile-time sparsity embedding.
 * All connectivity patterns are baked into hardware - no wasted multipliers.
 * 
 * Graph Statistics:
 *   Nodes: {N}
 *   Edges: {stats['num_edges']}
 *   Sparsity: {stats['sparsity']*100:.1f}%
 *   Max Degree: {stats['max_degree']}
 * 
 * DSP Comparison:
 *   Dense Total:  {dense_total:,} DSPs
 *   Sparse Total: {sparse_total:,} DSPs
 *   Savings:      {dense_total - sparse_total:,} DSPs ({(1-sparse_total/dense_total)*100:.1f}%)
 * 
 * Latency: SAME as dense (all operations are fully parallel, II=1)
 ******************************************************************************/

#ifndef GRAPHSAGE_HARDCODED_SPARSE_GENERATED_H
#define GRAPHSAGE_HARDCODED_SPARSE_GENERATED_H

#include <ap_fixed.h>

//==============================================================================
// Configuration
//==============================================================================

#define NUM_NODES       {N}
#define IN_FEATURES     {in_features}
#define HIDDEN_DIM      {hidden_dim}
#define OUT_FEATURES    {out_features}

//==============================================================================
// Data Types
//==============================================================================

typedef ap_fixed<16, 8> data_t;
typedef ap_fixed<16, 8> weight_t;
typedef ap_fixed<32, 16> acc_t;

'''
    
    # Generate aggregation functions for each feature dimension
    for feat_dim in feature_dims:
        header += generate_aggregate_function(stats, feat_dim)
    
    # Add linear transform template
    header += '''
//==============================================================================
// Linear Transform (dense - weights are not sparse)
//==============================================================================

template<int N_NODES, int IN_FEAT, int OUT_FEAT>
void linear_transform(
    const data_t input[N_NODES][IN_FEAT],
    const weight_t weights[OUT_FEAT][IN_FEAT],
    const weight_t bias[OUT_FEAT],
    data_t output[N_NODES][OUT_FEAT]
) {
    #pragma HLS INLINE
    #pragma HLS ARRAY_PARTITION variable=input complete dim=0
    #pragma HLS ARRAY_PARTITION variable=weights complete dim=0
    #pragma HLS ARRAY_PARTITION variable=bias complete dim=0
    #pragma HLS ARRAY_PARTITION variable=output complete dim=0
    
    LIN_N: for (int n = 0; n < N_NODES; n++) {
        #pragma HLS UNROLL
        LIN_OUT: for (int o = 0; o < OUT_FEAT; o++) {
            #pragma HLS UNROLL
            acc_t acc = 0;
            LIN_IN: for (int i = 0; i < IN_FEAT; i++) {
                #pragma HLS UNROLL
                acc += (acc_t)input[n][i] * (acc_t)weights[o][i];
            }
            acc += (acc_t)bias[o];
            output[n][o] = (data_t)acc;
        }
    }
}

//==============================================================================
// ReLU Activation
//==============================================================================

template<int N_NODES, int N_FEAT>
void relu(
    const data_t input[N_NODES][N_FEAT],
    data_t output[N_NODES][N_FEAT]
) {
    #pragma HLS INLINE
    #pragma HLS ARRAY_PARTITION variable=input complete dim=0
    #pragma HLS ARRAY_PARTITION variable=output complete dim=0
    
    RELU_N: for (int n = 0; n < N_NODES; n++) {
        #pragma HLS UNROLL
        RELU_F: for (int f = 0; f < N_FEAT; f++) {
            #pragma HLS UNROLL
            output[n][f] = (input[n][f] > 0) ? input[n][f] : (data_t)0;
        }
    }
}

//==============================================================================
// Complete Forward Pass
//==============================================================================

'''
    
    header += f'''void graphsage_hardcoded_sparse(
    const data_t node_features[NUM_NODES][IN_FEATURES],
    const weight_t weights1[HIDDEN_DIM][IN_FEATURES],
    const weight_t bias1[HIDDEN_DIM],
    const weight_t weights2[OUT_FEATURES][HIDDEN_DIM],
    const weight_t bias2[OUT_FEATURES],
    data_t output[NUM_NODES][OUT_FEATURES]
) {{
    #pragma HLS INTERFACE ap_ctrl_hs port=return
    #pragma HLS PIPELINE II=1
    
    static data_t agg1[NUM_NODES][IN_FEATURES];
    static data_t lin1[NUM_NODES][HIDDEN_DIM];
    static data_t relu1[NUM_NODES][HIDDEN_DIM];
    static data_t agg2[NUM_NODES][HIDDEN_DIM];
    
    #pragma HLS ARRAY_PARTITION variable=agg1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=lin1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=relu1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=agg2 complete dim=0
    
    // Layer 1: Sparse Agg → Linear → ReLU
    aggregate_sparse_{in_features}(node_features, agg1);
    linear_transform<NUM_NODES, IN_FEATURES, HIDDEN_DIM>(agg1, weights1, bias1, lin1);
    relu<NUM_NODES, HIDDEN_DIM>(lin1, relu1);
    
    // Layer 2: Sparse Agg → Linear
    aggregate_sparse_{hidden_dim}(relu1, agg2);
    linear_transform<NUM_NODES, HIDDEN_DIM, OUT_FEATURES>(agg2, weights2, bias2, output);
}}

#endif // GRAPHSAGE_HARDCODED_SPARSE_GENERATED_H
'''
    
    return header


def main():
    parser = argparse.ArgumentParser(description='Generate hardcoded sparse HLS code')
    parser.add_argument('--adj-file', type=str, 
                        default='build/test_vectors_float/adj_norm.txt',
                        help='Path to adjacency matrix file')
    parser.add_argument('--in-features', type=int, default=16,
                        help='Input feature dimension')
    parser.add_argument('--hidden-dim', type=int, default=24,
                        help='Hidden layer dimension')
    parser.add_argument('--out-features', type=int, default=7,
                        help='Output feature dimension')
    parser.add_argument('--output', type=str, 
                        default='hls/graphsage_hardcoded_sparse_generated.h',
                        help='Output header file path')
    
    args = parser.parse_args()
    
    # Load adjacency matrix
    print(f"Loading adjacency matrix from {args.adj_file}...")
    adj = np.loadtxt(args.adj_file, dtype=np.float32)
    
    # Analyze graph
    stats = analyze_graph(adj)
    print(f"\n✓ Graph statistics:")
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Edges: {stats['num_edges']}")
    print(f"  Sparsity: {stats['sparsity']*100:.1f}%")
    print(f"  Max degree: {stats['max_degree']}")
    print(f"  Degrees: {stats['degrees']}")
    
    # Generate header
    feature_dims = [args.in_features, args.hidden_dim]
    header = generate_header_file(adj, feature_dims, args.hidden_dim, args.out_features)
    
    # Write output
    with open(args.output, 'w') as f:
        f.write(header)
    
    print(f"\n✓ Generated HLS header: {args.output}")
    
    # Print DSP comparison
    N = stats['num_nodes']
    E = stats['num_edges']
    dense_agg = N * N * (args.in_features + args.hidden_dim)
    sparse_agg = E * (args.in_features + args.hidden_dim)
    
    print(f"\n╔═══════════════════════════════════════════════════╗")
    print(f"║           DSP SAVINGS SUMMARY                     ║")
    print(f"╠═══════════════════════════════════════════════════╣")
    print(f"║  Dense Aggregation:  {dense_agg:>6} DSPs                 ║")
    print(f"║  Sparse Aggregation: {sparse_agg:>6} DSPs                 ║")
    print(f"║  Savings:            {dense_agg-sparse_agg:>6} DSPs ({(1-sparse_agg/dense_agg)*100:.1f}%)        ║")
    print(f"║                                                   ║")
    print(f"║  Latency: UNCHANGED (still II=1)                  ║")
    print(f"╚═══════════════════════════════════════════════════╝")


if __name__ == '__main__':
    main()
