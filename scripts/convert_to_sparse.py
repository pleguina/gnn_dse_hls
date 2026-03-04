#!/usr/bin/env python3
"""
convert_to_sparse.py - Convert dense adjacency to neighbor list format

This script converts a dense adjacency matrix to the sparse neighbor list
format used by the optimized FPGA implementation.

Input format (dense):  adj[N][N] - N×N matrix with many zeros
Output format (sparse): neighbors[N][MAX_DEG], weights[N][MAX_DEG]

Example:
    Dense adjacency (8×8 = 64 values, 20 non-zeros):
    [[0.00, 0.33, 0.00, 0.00, 0.00, 0.33, 0.00, 0.33],
     [0.33, 0.00, 0.00, 0.00, 0.33, 0.00, 0.33, 0.00],
     ...]
    
    Sparse neighbor list (8×4 = 32 values):
    neighbors = [[1, 5, 7, -1], [0, 4, 6, -1], ...]
    weights = [[0.33, 0.33, 0.33, 0.0], [0.33, 0.33, 0.33, 0.0], ...]

Usage:
    python convert_to_sparse.py [--adj-file path/to/adj.txt] [--output-dir path/to/output]
"""

import numpy as np
import argparse
import os
from pathlib import Path


def load_adjacency_matrix(filepath: str) -> np.ndarray:
    """Load adjacency matrix from file."""
    return np.loadtxt(filepath, dtype=np.float32)


def analyze_sparsity(adj: np.ndarray) -> dict:
    """Analyze sparsity statistics of adjacency matrix."""
    N = adj.shape[0]
    non_zeros = np.count_nonzero(adj)
    total = N * N
    sparsity = 1 - (non_zeros / total)
    
    # Compute degree distribution
    degrees = np.sum(adj != 0, axis=1)
    max_degree = int(np.max(degrees))
    avg_degree = np.mean(degrees)
    
    return {
        'num_nodes': N,
        'total_elements': total,
        'non_zeros': non_zeros,
        'sparsity': sparsity,
        'degrees': degrees.tolist(),
        'max_degree': max_degree,
        'avg_degree': avg_degree
    }


def to_neighbor_list(adj: np.ndarray, max_degree: int = None) -> tuple:
    """
    Convert dense adjacency matrix to neighbor list format.
    
    Args:
        adj: Dense adjacency matrix [N×N]
        max_degree: Maximum neighbors per node (auto-detected if None)
    
    Returns:
        neighbors: Neighbor indices [N×MAX_DEG], padded with -1
        weights: Edge weights [N×MAX_DEG], padded with 0.0
    """
    N = adj.shape[0]
    
    # Auto-detect max_degree if not provided
    if max_degree is None:
        degrees = np.sum(adj != 0, axis=1)
        max_degree = int(np.max(degrees))
    
    # Initialize output arrays
    neighbors = np.full((N, max_degree), -1, dtype=np.int16)
    weights = np.zeros((N, max_degree), dtype=np.float32)
    
    # Convert each row
    for i in range(N):
        # Find non-zero columns (neighbors)
        edge_cols = np.nonzero(adj[i])[0]
        
        # Fill neighbor list (up to max_degree)
        for k, j in enumerate(edge_cols[:max_degree]):
            neighbors[i, k] = j
            weights[i, k] = adj[i, j]
    
    return neighbors, weights


def to_csr(adj: np.ndarray) -> tuple:
    """
    Convert dense adjacency matrix to CSR (Compressed Sparse Row) format.
    
    Returns:
        row_ptr: Row pointers [N+1]
        col_idx: Column indices [nnz]
        values: Non-zero values [nnz]
    """
    N = adj.shape[0]
    
    row_ptr = [0]
    col_idx = []
    values = []
    
    for i in range(N):
        edges = np.nonzero(adj[i])[0]
        for j in edges:
            col_idx.append(int(j))
            values.append(float(adj[i, j]))
        row_ptr.append(len(col_idx))
    
    return np.array(row_ptr, dtype=np.int32), \
           np.array(col_idx, dtype=np.int32), \
           np.array(values, dtype=np.float32)


def export_c_header(neighbors: np.ndarray, weights: np.ndarray, 
                    output_path: str, var_prefix: str = "adj_sparse"):
    """Export sparse adjacency as C header file."""
    N, max_deg = neighbors.shape
    
    header = f"""/*******************************************************************************
 * {os.path.basename(output_path)} - Auto-generated sparse adjacency data
 * 
 * Generated from dense adjacency matrix
 * Nodes: {N}, Max Degree: {max_deg}
 ******************************************************************************/

#ifndef ADJ_SPARSE_DATA_H
#define ADJ_SPARSE_DATA_H

#include <cstdint>

#define SPARSE_NUM_NODES {N}
#define SPARSE_MAX_DEGREE {max_deg}

// Neighbor indices (-1 indicates no neighbor / padding)
static const int16_t {var_prefix}_neighbors[SPARSE_NUM_NODES][SPARSE_MAX_DEGREE] = {{
"""
    
    # Add neighbor data
    for i in range(N):
        row = ", ".join(f"{v:3d}" for v in neighbors[i])
        header += f"    {{{row}}}"
        header += ",\n" if i < N - 1 else "\n"
    
    header += """};

// Edge weights (0.0 for invalid/padding entries)
"""
    header += f"static const float {var_prefix}_weights[SPARSE_NUM_NODES][SPARSE_MAX_DEGREE] = {{\n"
    
    # Add weight data
    for i in range(N):
        row = ", ".join(f"{v:.6f}" for v in weights[i])
        header += f"    {{{row}}}"
        header += ",\n" if i < N - 1 else "\n"
    
    header += """}};

#endif // ADJ_SPARSE_DATA_H
"""
    
    with open(output_path, 'w') as f:
        f.write(header)
    
    print(f"✓ Exported C header to {output_path}")


def export_comparison_table(stats: dict, neighbors: np.ndarray) -> str:
    """Generate comparison table between dense and sparse formats."""
    N = stats['num_nodes']
    max_deg = neighbors.shape[1]
    
    dense_storage = N * N  # Dense: N×N values
    sparse_storage = N * max_deg * 2  # Sparse: neighbors + weights
    
    dense_ops = N * N  # Dense aggregation: N×N multiplies per feature
    sparse_ops = N * max_deg  # Sparse aggregation: N×max_deg multiplies per feature
    
    table = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    DENSE vs SPARSE FORMAT COMPARISON                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  Graph Statistics:                                                            ║
║    • Nodes:           {N:>6}                                                   ║
║    • Non-zero edges:  {stats['non_zeros']:>6}                                                   ║
║    • Sparsity:        {stats['sparsity']*100:>6.1f}%                                                 ║
║    • Max degree:      {max_deg:>6}                                                   ║
║    • Avg degree:      {stats['avg_degree']:>6.2f}                                                  ║
║                                                                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  Storage (per aggregation):                                                   ║
║    • Dense:  {dense_storage:>6} values  (N×N = {N}×{N})                                ║
║    • Sparse: {sparse_storage:>6} values  (N×MAX_DEG×2 = {N}×{max_deg}×2)                         ║
║    • Savings: {(1-sparse_storage/dense_storage)*100:>5.1f}%                                                     ║
║                                                                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  Computation (per feature, per aggregation):                                  ║
║    • Dense:  {dense_ops:>6} multiplications  (N×N = {N}×{N})                         ║
║    • Sparse: {sparse_ops:>6} multiplications  (N×MAX_DEG = {N}×{max_deg})                     ║
║    • Savings: {(1-sparse_ops/dense_ops)*100:>5.1f}%                                                     ║
║                                                                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  Estimated DSP Savings (for your model):                                      ║
║    F_in=16, F_h=24                                                            ║
║    Dense Agg1:  {N}×{N}×16 = {N*N*16:>5} DSPs                                              ║
║    Sparse Agg1: {N}×{max_deg}×16 = {N*max_deg*16:>5} DSPs                                              ║
║    Dense Agg2:  {N}×{N}×24 = {N*N*24:>5} DSPs                                              ║
║    Sparse Agg2: {N}×{max_deg}×24 = {N*max_deg*24:>5} DSPs                                              ║
║                                                                               ║
║    Total Dense:  {N*N*16 + N*N*24:>5} DSPs (aggregation only)                             ║
║    Total Sparse: {N*max_deg*16 + N*max_deg*24:>5} DSPs (aggregation only)                             ║
║    DSP Savings:  {N*N*16 + N*N*24 - N*max_deg*16 - N*max_deg*24:>5} DSPs ({(1-(N*max_deg*16+N*max_deg*24)/(N*N*16+N*N*24))*100:.1f}% of aggregation)                      ║
║                                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    return table


def main():
    parser = argparse.ArgumentParser(description='Convert dense adjacency to sparse format')
    parser.add_argument('--adj-file', type=str, 
                        default='build/test_vectors_float/adj_norm.txt',
                        help='Path to dense adjacency matrix file')
    parser.add_argument('--output-dir', type=str,
                        default='build/sparse_format',
                        help='Output directory for sparse files')
    parser.add_argument('--max-degree', type=int, default=None,
                        help='Maximum neighbors per node (auto-detect if not set)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load adjacency matrix
    print(f"Loading adjacency matrix from {args.adj_file}...")
    adj = load_adjacency_matrix(args.adj_file)
    
    # Analyze sparsity
    stats = analyze_sparsity(adj)
    print(f"\n✓ Loaded {stats['num_nodes']}×{stats['num_nodes']} matrix")
    print(f"  Non-zeros: {stats['non_zeros']}/{stats['total_elements']} ({stats['sparsity']*100:.1f}% sparse)")
    print(f"  Max degree: {stats['max_degree']}")
    print(f"  Avg degree: {stats['avg_degree']:.2f}")
    
    # Convert to neighbor list format
    max_deg = args.max_degree if args.max_degree else stats['max_degree']
    neighbors, weights = to_neighbor_list(adj, max_deg)
    
    # Export files
    np.savetxt(os.path.join(args.output_dir, 'neighbors.txt'), neighbors, fmt='%d')
    np.savetxt(os.path.join(args.output_dir, 'edge_weights.txt'), weights, fmt='%.6f')
    
    # Export C header
    export_c_header(neighbors, weights, 
                    os.path.join(args.output_dir, 'adj_sparse_data.h'))
    
    # Export as CSR for comparison
    row_ptr, col_idx, values = to_csr(adj)
    np.savetxt(os.path.join(args.output_dir, 'csr_row_ptr.txt'), row_ptr, fmt='%d')
    np.savetxt(os.path.join(args.output_dir, 'csr_col_idx.txt'), col_idx, fmt='%d')
    np.savetxt(os.path.join(args.output_dir, 'csr_values.txt'), values, fmt='%.6f')
    
    # Print comparison table
    print(export_comparison_table(stats, neighbors))
    
    # Save summary
    summary_path = os.path.join(args.output_dir, 'conversion_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(export_comparison_table(stats, neighbors))
    print(f"✓ Summary saved to {summary_path}")


if __name__ == '__main__':
    main()
