#!/usr/bin/env python3
"""
HLS Synthesis Report Visualization

Creates visual comparisons of HLS implementation metrics:
- Resource utilization bar charts
- Latency comparison
- Resource efficiency scatter plots
- Summary dashboard

Usage:
    python visualize_hls_report.py --all build/hls
    python visualize_hls_report.py --all build/hls -o build/plots/hls_comparison.png
    python visualize_hls_report.py --json build/hls_comparison.json
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Import the parser
sys.path.insert(0, str(Path(__file__).parent))
from parse_hls_report import (
    HLSReport, find_all_hls_projects, find_csynth_xml, parse_csynth_xml
)


def load_reports_from_json(json_path: Path) -> list:
    """Load reports from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert JSON back to simple dict structure for plotting
    return data


def load_reports_from_projects(base_dir: Path) -> list:
    """Load reports from HLS project directories"""
    projects = find_all_hls_projects(base_dir)
    reports = []
    
    for project in projects:
        xml_path = find_csynth_xml(project)
        if xml_path:
            report = parse_csynth_xml(xml_path)
            if report:
                reports.append(report.to_dict())
    
    return reports


def create_resource_comparison(reports: list, ax: plt.Axes):
    """Create grouped bar chart comparing resource utilization"""
    names = [r['name'].replace('graphsage_', '') for r in reports]
    x = np.arange(len(names))
    width = 0.2
    
    # Extract actual percentages (no capping - show real values)
    dsp_pct = [r['resources']['dsp_pct'] for r in reports]
    ff_pct = [r['resources']['ff_pct'] for r in reports]
    lut_pct = [r['resources']['lut_pct'] for r in reports]
    
    bars1 = ax.bar(x - width, dsp_pct, width, label='DSP', color='#2ecc71', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, ff_pct, width, label='FF', color='#3498db', edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, lut_pct, width, label='LUT', color='#e74c3c', edgecolor='black', linewidth=0.5)
    
    # Add 100% reference line
    ax.axhline(y=100, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='100% utilization')
    
    ax.set_xlabel('Implementation', fontsize=11)
    ax.set_ylabel('Resource Utilization (%)', fontsize=11)
    ax.set_title('FPGA Resource Utilization by Implementation', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, max(max(dsp_pct), max(ff_pct), max(lut_pct)) * 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 5:
                ax.annotate(f'{height:.0f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=7)


def create_latency_comparison(reports: list, ax: plt.Axes):
    """Create bar chart comparing latency"""
    names = [r['name'].replace('graphsage_', '') for r in reports]
    latencies = [r['latency']['worst_cycles'] for r in reports]
    
    colors = ['#3498db' if not r.get('has_timing_violation', False) else '#e74c3c' for r in reports]
    
    bars = ax.bar(names, latencies, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Implementation', fontsize=11)
    ax.set_ylabel('Latency (clock cycles)', fontsize=11)
    ax.set_title('Inference Latency Comparison', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, lat in zip(bars, latencies):
        ax.annotate(f'{lat}',
                   xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Legend for timing status
    ok_patch = mpatches.Patch(color='#3498db', label='Timing Met')
    fail_patch = mpatches.Patch(color='#e74c3c', label='Timing Violation')
    ax.legend(handles=[ok_patch, fail_patch], loc='upper right', fontsize=9)


def create_absolute_resources(reports: list, ax: plt.Axes):
    """Create bar chart with absolute resource counts"""
    names = [r['name'].replace('graphsage_', '') for r in reports]
    x = np.arange(len(names))
    width = 0.35
    
    dsps = [r['resources']['dsp'] for r in reports]
    luts = [r['resources']['lut'] / 1000 for r in reports]  # in thousands
    
    ax2 = ax.twinx()
    
    bars1 = ax.bar(x - width/2, dsps, width, label='DSP', color='#2ecc71', edgecolor='black', linewidth=0.5)
    bars2 = ax2.bar(x + width/2, luts, width, label='LUT (K)', color='#e74c3c', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Implementation', fontsize=11)
    ax.set_ylabel('DSP Count', fontsize=11, color='#2ecc71')
    ax2.set_ylabel('LUT Count (thousands)', fontsize=11, color='#e74c3c')
    ax.set_title('Absolute Resource Usage', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    
    ax.tick_params(axis='y', labelcolor='#2ecc71')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
    
    ax.grid(axis='y', alpha=0.3)


def create_efficiency_scatter(reports: list, ax: plt.Axes):
    """Create scatter plot of latency vs resources (efficiency)"""
    names = [r['name'].replace('graphsage_', '') for r in reports]
    latencies = [r['latency']['worst_cycles'] for r in reports]
    dsps = [r['resources']['dsp'] for r in reports]
    luts = [r['resources']['lut'] for r in reports]
    
    # Size based on LUT usage
    sizes = [max(50, l / 5000) for l in luts]
    
    # Color based on timing
    colors = ['#2ecc71' if not r.get('has_timing_violation', False) else '#e74c3c' for r in reports]
    
    scatter = ax.scatter(dsps, latencies, s=sizes, c=colors, alpha=0.7, edgecolors='black', linewidth=1)
    
    # Add labels
    for i, name in enumerate(names):
        ax.annotate(name, (dsps[i], latencies[i]), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold')
    
    ax.set_xlabel('DSP Usage', fontsize=11)
    ax.set_ylabel('Latency (cycles)', fontsize=11)
    ax.set_title('Efficiency: Latency vs DSP Usage\n(bubble size = LUT usage)', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Legend
    ok_patch = mpatches.Patch(color='#2ecc71', label='Timing Met')
    fail_patch = mpatches.Patch(color='#e74c3c', label='Timing Violation')
    ax.legend(handles=[ok_patch, fail_patch], loc='upper right', fontsize=9)


def create_summary_table(reports: list, ax: plt.Axes):
    """Create summary table"""
    ax.axis('off')
    
    # Prepare data
    headers = ['Implementation', 'Latency\n(cycles)', 'II', 'DSP', 'FF', 'LUT', 'Timing']
    
    rows = []
    for r in reports:
        name = r['name'].replace('graphsage_', '')
        latency = f"{r['latency']['worst_cycles']}"
        ii = f"{r['latency']['interval_min']}"
        dsp = f"{r['resources']['dsp']:,}\n({r['resources']['dsp_pct']:.0f}%)"
        ff = f"{r['resources']['ff']:,}\n({r['resources']['ff_pct']:.0f}%)"
        lut = f"{r['resources']['lut']:,}\n({r['resources']['lut_pct']:.0f}%)"
        timing = "✓" if not r.get('has_timing_violation', False) else "✗"
        rows.append([name, latency, ii, dsp, ff, lut, timing])
    
    # Create table
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colColours=['#3498db'] * len(headers),
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Color timing column
    for i, r in enumerate(reports):
        if r.get('has_timing_violation', False):
            table[(i + 1, 6)].set_facecolor('#ffcccc')
        else:
            table[(i + 1, 6)].set_facecolor('#ccffcc')
    
    ax.set_title('HLS Implementation Summary', fontsize=14, fontweight='bold', pad=20)


def create_timing_comparison(reports: list, ax: plt.Axes):
    """Create timing/frequency comparison"""
    names = [r['name'].replace('graphsage_', '') for r in reports]
    target_freq = [r['timing']['target_freq_mhz'] for r in reports]
    est_freq = [r['timing']['estimated_freq_mhz'] for r in reports]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, target_freq, width, label='Target', color='#95a5a6', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, est_freq, width, label='Achievable', color='#3498db', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Implementation', fontsize=11)
    ax.set_ylabel('Frequency (MHz)', fontsize=11)
    ax.set_title('Clock Frequency Analysis', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, freq in zip(bars2, est_freq):
        ax.annotate(f'{freq:.0f}',
                   xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)


def create_dashboard(reports: list, output_path: Path = None, show: bool = True):
    """Create full dashboard with all visualizations"""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('HLS Implementation Comparison Dashboard', fontsize=16, fontweight='bold', y=0.98)
    
    # Create grid
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25, 
                          left=0.06, right=0.94, top=0.92, bottom=0.05)
    
    # Summary table (top spanning both columns)
    ax_table = fig.add_subplot(gs[0, :])
    create_summary_table(reports, ax_table)
    
    # Resource comparison
    ax_res = fig.add_subplot(gs[1, 0])
    create_resource_comparison(reports, ax_res)
    
    # Latency comparison
    ax_lat = fig.add_subplot(gs[1, 1])
    create_latency_comparison(reports, ax_lat)
    
    # Absolute resources
    ax_abs = fig.add_subplot(gs[2, 0])
    create_absolute_resources(reports, ax_abs)
    
    # Efficiency scatter
    ax_eff = fig.add_subplot(gs[2, 1])
    create_efficiency_scatter(reports, ax_eff)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved dashboard to: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()


def create_single_chart(reports: list, chart_type: str, output_path: Path = None, show: bool = True):
    """Create a single chart type"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if chart_type == 'resources':
        create_resource_comparison(reports, ax)
    elif chart_type == 'latency':
        create_latency_comparison(reports, ax)
    elif chart_type == 'absolute':
        create_absolute_resources(reports, ax)
    elif chart_type == 'efficiency':
        create_efficiency_scatter(reports, ax)
    elif chart_type == 'timing':
        create_timing_comparison(reports, ax)
    else:
        print(f"Unknown chart type: {chart_type}")
        return
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved chart to: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize HLS synthesis reports',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--all', '-a', type=str, metavar='DIR',
                        help='Parse all HLS projects in directory')
    parser.add_argument('--json', '-j', type=str, metavar='FILE',
                        help='Load reports from JSON file')
    parser.add_argument('--output', '-o', type=str, metavar='FILE',
                        help='Output image file')
    parser.add_argument('--chart', '-c', type=str,
                        choices=['resources', 'latency', 'absolute', 'efficiency', 'timing', 'dashboard'],
                        default='dashboard',
                        help='Chart type to generate (default: dashboard)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display the plot (only save)')
    
    args = parser.parse_args()
    
    if not args.all and not args.json:
        parser.print_help()
        return 1
    
    # Load reports
    if args.json:
        reports = load_reports_from_json(Path(args.json))
    else:
        reports = load_reports_from_projects(Path(args.all))
    
    if not reports:
        print("No reports to visualize", file=sys.stderr)
        return 1
    
    print(f"Loaded {len(reports)} reports: {[r['name'] for r in reports]}")
    
    # Generate visualization
    output_path = Path(args.output) if args.output else None
    show = not args.no_show
    
    if args.chart == 'dashboard':
        create_dashboard(reports, output_path, show)
    else:
        create_single_chart(reports, args.chart, output_path, show)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
