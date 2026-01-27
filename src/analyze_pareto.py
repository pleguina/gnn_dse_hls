#!/usr/bin/env python3
"""
Pareto Front Analysis for Design Space Exploration

Analyzes design space results and generates Pareto front visualizations:
- 2D Pareto plots (accuracy vs DSP, latency vs DSP, etc.)
- 3D Pareto plots
- Pareto-optimal design identification
- Trade-off analysis

Usage:
    python src/analyze_pareto.py --input build/experiments/design_space_results.json
    python src/analyze_pareto.py --input build/experiments/design_space_results.csv
    python src/analyze_pareto.py --input build/experiments/design_space_results.json --output build/plots
"""

import argparse
import json
import csv
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


@dataclass
class ParetoObjective:
    """Definition of a Pareto objective"""
    name: str
    direction: str  # "min" or "max"
    label: str
    unit: str = ""


# Standard objectives for HLS design space
STANDARD_OBJECTIVES = {
    'accuracy': ParetoObjective('test_accuracy', 'max', 'Accuracy', '%'),
    'latency_cycles': ParetoObjective('latency_max_cycles', 'min', 'Latency', 'cycles'),
    'latency_ns': ParetoObjective('latency_ns', 'min', 'Latency', 'ns'),
    'dsp_used': ParetoObjective('dsp_used', 'min', 'DSP Usage', ''),
    'dsp_pct': ParetoObjective('dsp_pct', 'min', 'DSP Usage', '%'),
    'lut_used': ParetoObjective('lut_used', 'min', 'LUT Usage', ''),
    'lut_pct': ParetoObjective('lut_pct', 'min', 'LUT Usage', '%'),
    'ff_used': ParetoObjective('ff_used', 'min', 'FF Usage', ''),
    'fmax_mhz': ParetoObjective('fmax_mhz', 'max', 'Fmax', 'MHz'),
    'throughput': ParetoObjective('throughput_inferences_per_us', 'max', 'Throughput', 'infer/µs'),
}


def load_results(input_path: Path) -> List[Dict]:
    """Load design space results from JSON or CSV"""
    if input_path.suffix == '.json':
        with open(input_path, 'r') as f:
            return json.load(f)
    elif input_path.suffix == '.csv':
        with open(input_path, 'r') as f:
            reader = csv.DictReader(f)
            results = []
            for row in reader:
                # Convert numeric fields
                for key, value in row.items():
                    try:
                        if '.' in value:
                            row[key] = float(value)
                        else:
                            row[key] = int(value)
                    except (ValueError, TypeError):
                        pass
                results.append(row)
            return results
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")


def filter_completed(results: List[Dict]) -> List[Dict]:
    """Filter to only completed designs"""
    return [r for r in results if r.get('status') == 'completed']


def is_dominated(point: np.ndarray, other: np.ndarray, directions: List[str]) -> bool:
    """
    Check if 'point' is dominated by 'other'.
    
    A point is dominated if another point is at least as good in all objectives
    and strictly better in at least one.
    
    Args:
        point: Values for each objective
        other: Values for the potentially dominating point
        directions: "min" or "max" for each objective
    """
    at_least_as_good = True
    strictly_better = False
    
    for i, direction in enumerate(directions):
        if direction == 'min':
            if other[i] > point[i]:
                at_least_as_good = False
            if other[i] < point[i]:
                strictly_better = True
        else:  # max
            if other[i] < point[i]:
                at_least_as_good = False
            if other[i] > point[i]:
                strictly_better = True
    
    return at_least_as_good and strictly_better


def compute_pareto_front(results: List[Dict], objectives: List[ParetoObjective]) -> List[int]:
    """
    Compute Pareto-optimal indices from results.
    
    Returns:
        List of indices into results that are Pareto-optimal
    """
    n = len(results)
    if n == 0:
        return []
    
    # Extract objective values
    obj_names = [obj.name for obj in objectives]
    directions = [obj.direction for obj in objectives]
    
    points = np.zeros((n, len(objectives)))
    for i, r in enumerate(results):
        for j, name in enumerate(obj_names):
            points[i, j] = r.get(name, 0)
    
    # Find non-dominated points
    pareto_indices = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i != j and is_dominated(points[i], points[j], directions):
                dominated = True
                break
        if not dominated:
            pareto_indices.append(i)
    
    return pareto_indices


def plot_pareto_2d(results: List[Dict], 
                   obj_x: ParetoObjective, 
                   obj_y: ParetoObjective,
                   output_path: Optional[Path] = None,
                   show: bool = True,
                   highlight_pareto: bool = True,
                   title: Optional[str] = None):
    """Create 2D Pareto plot"""
    
    # Extract values
    x_vals = [r.get(obj_x.name, 0) for r in results]
    y_vals = [r.get(obj_y.name, 0) for r in results]
    names = [r.get('design_id', f'design_{i}') for i, r in enumerate(results)]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Compute Pareto front
    if highlight_pareto:
        pareto_idx = compute_pareto_front(results, [obj_x, obj_y])
        pareto_set = set(pareto_idx)
        
        # Plot non-Pareto points
        non_pareto_x = [x_vals[i] for i in range(len(results)) if i not in pareto_set]
        non_pareto_y = [y_vals[i] for i in range(len(results)) if i not in pareto_set]
        ax.scatter(non_pareto_x, non_pareto_y, c='lightgray', s=60, alpha=0.6, 
                   label='Dominated', edgecolors='gray', linewidth=0.5)
        
        # Plot Pareto points
        pareto_x = [x_vals[i] for i in pareto_idx]
        pareto_y = [y_vals[i] for i in pareto_idx]
        ax.scatter(pareto_x, pareto_y, c='#e74c3c', s=120, alpha=0.9,
                   label='Pareto-optimal', edgecolors='black', linewidth=1, zorder=5)
        
        # Connect Pareto front with line (sorted)
        if len(pareto_idx) > 1:
            pareto_points = sorted(zip(pareto_x, pareto_y), key=lambda p: p[0])
            px, py = zip(*pareto_points)
            ax.plot(px, py, 'r--', alpha=0.5, linewidth=2, label='Pareto front')
        
        # Label Pareto points
        for i in pareto_idx:
            short_name = names[i].split('_')[0] if '_' in names[i] else names[i][:10]
            ax.annotate(short_name, (x_vals[i], y_vals[i]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8)
    else:
        ax.scatter(x_vals, y_vals, c='#3498db', s=80, alpha=0.7,
                   edgecolors='black', linewidth=0.5)
    
    # Labels and formatting
    x_label = f"{obj_x.label}"
    if obj_x.unit:
        x_label += f" ({obj_x.unit})"
    y_label = f"{obj_y.label}"
    if obj_y.unit:
        y_label += f" ({obj_y.unit})"
    
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'{obj_y.label} vs {obj_x.label}', fontsize=14, fontweight='bold')
    
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    
    # Add arrows to indicate optimization direction
    ax.annotate('', xy=(0.02, 0.98), xycoords='axes fraction',
                xytext=(0.02, 0.88), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='green' if obj_y.direction == 'max' else 'red'))
    ax.annotate('', xy=(0.12, 0.88), xycoords='axes fraction',
                xytext=(0.02, 0.88), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='green' if obj_x.direction == 'max' else 'red'))
    ax.text(0.07, 0.85, 'Better', transform=ax.transAxes, fontsize=8, color='gray')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_pareto_3d(results: List[Dict],
                   obj_x: ParetoObjective,
                   obj_y: ParetoObjective,
                   obj_z: ParetoObjective,
                   output_path: Optional[Path] = None,
                   show: bool = True):
    """Create 3D Pareto plot"""
    
    x_vals = [r.get(obj_x.name, 0) for r in results]
    y_vals = [r.get(obj_y.name, 0) for r in results]
    z_vals = [r.get(obj_z.name, 0) for r in results]
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Compute Pareto front
    pareto_idx = compute_pareto_front(results, [obj_x, obj_y, obj_z])
    pareto_set = set(pareto_idx)
    
    # Plot non-Pareto points
    non_pareto_x = [x_vals[i] for i in range(len(results)) if i not in pareto_set]
    non_pareto_y = [y_vals[i] for i in range(len(results)) if i not in pareto_set]
    non_pareto_z = [z_vals[i] for i in range(len(results)) if i not in pareto_set]
    ax.scatter(non_pareto_x, non_pareto_y, non_pareto_z, c='lightgray', s=40, 
               alpha=0.5, label='Dominated')
    
    # Plot Pareto points
    pareto_x = [x_vals[i] for i in pareto_idx]
    pareto_y = [y_vals[i] for i in pareto_idx]
    pareto_z = [z_vals[i] for i in pareto_idx]
    ax.scatter(pareto_x, pareto_y, pareto_z, c='#e74c3c', s=100, 
               alpha=0.9, label='Pareto-optimal', edgecolors='black')
    
    ax.set_xlabel(f"{obj_x.label} ({obj_x.unit})" if obj_x.unit else obj_x.label)
    ax.set_ylabel(f"{obj_y.label} ({obj_y.unit})" if obj_y.unit else obj_y.label)
    ax.set_zlabel(f"{obj_z.label} ({obj_z.unit})" if obj_z.unit else obj_z.label)
    ax.set_title(f'3D Pareto: {obj_x.label} vs {obj_y.label} vs {obj_z.label}', 
                 fontsize=12, fontweight='bold')
    ax.legend()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()


def generate_pareto_report(results: List[Dict], objectives: List[Tuple[str, str]]) -> Dict:
    """Generate summary report of Pareto analysis"""
    report = {
        'total_designs': len(results),
        'completed_designs': len([r for r in results if r.get('status') == 'completed']),
        'pareto_fronts': {}
    }
    
    for obj1_name, obj2_name in objectives:
        obj1 = STANDARD_OBJECTIVES.get(obj1_name)
        obj2 = STANDARD_OBJECTIVES.get(obj2_name)
        
        if obj1 and obj2:
            completed = filter_completed(results)
            pareto_idx = compute_pareto_front(completed, [obj1, obj2])
            
            key = f"{obj1_name}_vs_{obj2_name}"
            report['pareto_fronts'][key] = {
                'num_pareto_points': len(pareto_idx),
                'pareto_designs': [completed[i].get('design_id') for i in pareto_idx],
                'pareto_values': [
                    {
                        'design_id': completed[i].get('design_id'),
                        obj1_name: completed[i].get(obj1.name),
                        obj2_name: completed[i].get(obj2.name),
                    }
                    for i in pareto_idx
                ]
            }
    
    return report


def create_pareto_dashboard(results: List[Dict], output_dir: Path, show: bool = False):
    """Create comprehensive Pareto analysis dashboard"""
    
    completed = filter_completed(results)
    if not completed:
        print("No completed designs to analyze")
        return
    
    print(f"\nAnalyzing {len(completed)} completed designs...")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate 2D Pareto plots
    plots_2d = [
        ('accuracy', 'dsp_used', 'Accuracy vs DSP Usage'),
        ('accuracy', 'latency_cycles', 'Accuracy vs Latency'),
        ('latency_cycles', 'dsp_used', 'Latency vs DSP Usage'),
        ('accuracy', 'lut_used', 'Accuracy vs LUT Usage'),
        ('dsp_used', 'lut_used', 'DSP vs LUT Usage'),
    ]
    
    for obj1_name, obj2_name, title in plots_2d:
        obj1 = STANDARD_OBJECTIVES.get(obj1_name)
        obj2 = STANDARD_OBJECTIVES.get(obj2_name)
        
        if obj1 and obj2:
            output_path = output_dir / f"pareto_{obj1_name}_vs_{obj2_name}.png"
            plot_pareto_2d(completed, obj1, obj2, output_path, show=show, title=title)
    
    # Generate 3D Pareto plot
    obj_acc = STANDARD_OBJECTIVES['accuracy']
    obj_lat = STANDARD_OBJECTIVES['latency_cycles']
    obj_dsp = STANDARD_OBJECTIVES['dsp_used']
    
    output_path = output_dir / "pareto_3d_accuracy_latency_dsp.png"
    plot_pareto_3d(completed, obj_acc, obj_lat, obj_dsp, output_path, show=show)
    
    # Generate report
    report = generate_pareto_report(results, [
        ('accuracy', 'dsp_used'),
        ('accuracy', 'latency_cycles'),
        ('latency_cycles', 'dsp_used'),
    ])
    
    report_path = output_dir / "pareto_analysis.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {report_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("PARETO ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total designs: {report['total_designs']}")
    print(f"Completed designs: {report['completed_designs']}")
    
    for front_name, front_data in report['pareto_fronts'].items():
        print(f"\n{front_name}:")
        print(f"  Pareto-optimal points: {front_data['num_pareto_points']}")
        for pv in front_data['pareto_values'][:5]:  # Show first 5
            print(f"    - {pv}")


def main():
    parser = argparse.ArgumentParser(
        description='Pareto Front Analysis for Design Space Exploration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input results file (JSON or CSV)')
    parser.add_argument('--output', '-o', type=str, default='build/plots/pareto',
                        help='Output directory for plots')
    parser.add_argument('--show', action='store_true',
                        help='Display plots interactively')
    parser.add_argument('--plot', type=str, nargs=2, metavar=('OBJ1', 'OBJ2'),
                        help='Generate single 2D plot with specified objectives')
    
    args = parser.parse_args()
    
    # Load results
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    results = load_results(input_path)
    print(f"Loaded {len(results)} design points from {input_path}")
    
    # Generate plots
    output_dir = Path(args.output)
    
    if args.plot:
        # Single plot mode
        obj1_name, obj2_name = args.plot
        obj1 = STANDARD_OBJECTIVES.get(obj1_name)
        obj2 = STANDARD_OBJECTIVES.get(obj2_name)
        
        if not obj1:
            print(f"Unknown objective: {obj1_name}")
            print(f"Available: {list(STANDARD_OBJECTIVES.keys())}")
            return 1
        if not obj2:
            print(f"Unknown objective: {obj2_name}")
            return 1
        
        completed = filter_completed(results)
        output_path = output_dir / f"pareto_{obj1_name}_vs_{obj2_name}.png"
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_pareto_2d(completed, obj1, obj2, output_path, show=args.show)
    else:
        # Full dashboard mode
        create_pareto_dashboard(results, output_dir, show=args.show)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
