#!/usr/bin/env python3
"""
HLS Synthesis Report Parser

Parses Vitis HLS csynth.xml reports and extracts key metrics:
- Latency (cycles, real-time)
- Initiation Interval (II)
- Resource utilization (DSP, FF, LUT, BRAM, URAM)
- Clock period and frequency

Usage:
    # Parse single report
    python parse_hls_report.py build/hls/graphsage_int8
    
    # Compare multiple implementations
    python parse_hls_report.py build/hls/graphsage_float build/hls/graphsage_int8
    
    # Parse all HLS projects in directory
    python parse_hls_report.py --all build/hls
    
    # Export to JSON
    python parse_hls_report.py --json build/hls/graphsage_int8
    
    # Export comparison to CSV
    python parse_hls_report.py --csv --all build/hls
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
import sys


@dataclass
class HLSResources:
    """FPGA resource utilization"""
    dsp: int = 0
    ff: int = 0
    lut: int = 0
    bram_18k: int = 0
    uram: int = 0
    
    # Available resources (for percentage calculation)
    dsp_avail: int = 0
    ff_avail: int = 0
    lut_avail: int = 0
    bram_avail: int = 0
    uram_avail: int = 0
    
    @property
    def dsp_pct(self) -> float:
        return 100 * self.dsp / self.dsp_avail if self.dsp_avail > 0 else 0
    
    @property
    def ff_pct(self) -> float:
        return 100 * self.ff / self.ff_avail if self.ff_avail > 0 else 0
    
    @property
    def lut_pct(self) -> float:
        return 100 * self.lut / self.lut_avail if self.lut_avail > 0 else 0
    
    @property
    def bram_pct(self) -> float:
        return 100 * self.bram_18k / self.bram_avail if self.bram_avail > 0 else 0
    
    @property
    def uram_pct(self) -> float:
        return 100 * self.uram / self.uram_avail if self.uram_avail > 0 else 0


@dataclass
class HLSLatency:
    """Latency and timing information"""
    best_cycles: int = 0
    avg_cycles: int = 0
    worst_cycles: int = 0
    best_realtime: str = ""
    avg_realtime: str = ""
    worst_realtime: str = ""
    interval_min: int = 0  # II
    interval_max: int = 0


@dataclass 
class HLSTiming:
    """Clock and timing analysis"""
    target_clock_ns: float = 0.0
    estimated_clock_ns: float = 0.0
    clock_uncertainty_ns: float = 0.0
    
    @property
    def target_freq_mhz(self) -> float:
        return 1000 / self.target_clock_ns if self.target_clock_ns > 0 else 0
    
    @property
    def estimated_freq_mhz(self) -> float:
        return 1000 / self.estimated_clock_ns if self.estimated_clock_ns > 0 else 0
    
    @property
    def slack_ns(self) -> float:
        return self.target_clock_ns - self.clock_uncertainty_ns - self.estimated_clock_ns


@dataclass
class HLSReport:
    """Complete HLS synthesis report"""
    name: str
    top_function: str
    part: str
    product_family: str
    hls_version: str
    
    timing: HLSTiming
    latency: HLSLatency
    resources: HLSResources
    
    # Pipeline info
    pipeline_type: str = "no"
    has_timing_violation: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export"""
        return {
            "name": self.name,
            "top_function": self.top_function,
            "part": self.part,
            "product_family": self.product_family,
            "hls_version": self.hls_version,
            "pipeline_type": self.pipeline_type,
            "has_timing_violation": self.has_timing_violation,
            "timing": {
                "target_clock_ns": self.timing.target_clock_ns,
                "estimated_clock_ns": self.timing.estimated_clock_ns,
                "target_freq_mhz": round(self.timing.target_freq_mhz, 1),
                "estimated_freq_mhz": round(self.timing.estimated_freq_mhz, 1),
                "slack_ns": round(self.timing.slack_ns, 3),
            },
            "latency": {
                "best_cycles": self.latency.best_cycles,
                "avg_cycles": self.latency.avg_cycles,
                "worst_cycles": self.latency.worst_cycles,
                "best_realtime": self.latency.best_realtime,
                "interval_min": self.latency.interval_min,
                "interval_max": self.latency.interval_max,
            },
            "resources": {
                "dsp": self.resources.dsp,
                "dsp_pct": round(self.resources.dsp_pct, 1),
                "ff": self.resources.ff,
                "ff_pct": round(self.resources.ff_pct, 1),
                "lut": self.resources.lut,
                "lut_pct": round(self.resources.lut_pct, 1),
                "bram_18k": self.resources.bram_18k,
                "bram_pct": round(self.resources.bram_pct, 1),
                "uram": self.resources.uram,
                "uram_pct": round(self.resources.uram_pct, 1),
            }
        }


def find_csynth_xml(project_path: Path) -> Optional[Path]:
    """Find csynth.xml in HLS project directory"""
    # Standard locations
    candidates = [
        project_path / "solution1" / "syn" / "report" / "csynth.xml",
        project_path / "solution1" / "syn" / "report" / "csynth.xml",
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    # Search recursively
    for xml_file in project_path.rglob("csynth.xml"):
        return xml_file
    
    return None


def parse_csynth_xml(xml_path: Path, name: str = None) -> Optional[HLSReport]:
    """Parse csynth.xml and extract metrics"""
    if not xml_path.exists():
        print(f"Error: File not found: {xml_path}", file=sys.stderr)
        return None
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing {xml_path}: {e}", file=sys.stderr)
        return None
    
    # Extract user assignments
    user = root.find("UserAssignments")
    product_family = user.findtext("ProductFamily", "") if user is not None else ""
    part = user.findtext("Part", "") if user is not None else ""
    top_function = user.findtext("TopModelName", "") if user is not None else ""
    target_clock = float(user.findtext("TargetClockPeriod", "0") or 0) if user is not None else 0
    clock_uncertainty = float(user.findtext("ClockUncertainty", "0") or 0) if user is not None else 0
    
    # Extract version
    version = root.find("ReportVersion")
    hls_version = version.findtext("Version", "") if version is not None else ""
    
    # Extract performance estimates
    perf = root.find("PerformanceEstimates")
    pipeline_type = perf.findtext("PipelineType", "no") if perf is not None else "no"
    
    # Timing
    timing_summary = perf.find("SummaryOfTimingAnalysis") if perf is not None else None
    estimated_clock = float(timing_summary.findtext("EstimatedClockPeriod", "0") or 0) if timing_summary is not None else 0
    
    # Latency
    latency_summary = perf.find("SummaryOfOverallLatency") if perf is not None else None
    latency = HLSLatency()
    if latency_summary is not None:
        latency.best_cycles = int(latency_summary.findtext("Best-caseLatency", "0") or 0)
        latency.avg_cycles = int(latency_summary.findtext("Average-caseLatency", "0") or 0)
        latency.worst_cycles = int(latency_summary.findtext("Worst-caseLatency", "0") or 0)
        latency.best_realtime = latency_summary.findtext("Best-caseRealTimeLatency", "")
        latency.avg_realtime = latency_summary.findtext("Average-caseRealTimeLatency", "")
        latency.worst_realtime = latency_summary.findtext("Worst-caseRealTimeLatency", "")
        latency.interval_min = int(latency_summary.findtext("Interval-min", "0") or 0)
        latency.interval_max = int(latency_summary.findtext("Interval-max", "0") or 0)
    
    # Violations
    violations = perf.find("SummaryOfViolations") if perf is not None else None
    has_violation = False
    if violations is not None:
        issue_type = violations.findtext("IssueType", "-")
        has_violation = issue_type != "-" and "Violation" in issue_type
    
    # Area estimates
    area = root.find("AreaEstimates")
    resources = HLSResources()
    if area is not None:
        res = area.find("Resources")
        if res is not None:
            resources.dsp = int(res.findtext("DSP", "0") or 0)
            resources.ff = int(res.findtext("FF", "0") or 0)
            resources.lut = int(res.findtext("LUT", "0") or 0)
            resources.bram_18k = int(res.findtext("BRAM_18K", "0") or 0)
            resources.uram = int(res.findtext("URAM", "0") or 0)
        
        avail = area.find("AvailableResources")
        if avail is not None:
            resources.dsp_avail = int(avail.findtext("DSP", "0") or 0)
            resources.ff_avail = int(avail.findtext("FF", "0") or 0)
            resources.lut_avail = int(avail.findtext("LUT", "0") or 0)
            resources.bram_avail = int(avail.findtext("BRAM_18K", "0") or 0)
            resources.uram_avail = int(avail.findtext("URAM", "0") or 0)
    
    timing = HLSTiming(
        target_clock_ns=target_clock,
        estimated_clock_ns=estimated_clock,
        clock_uncertainty_ns=clock_uncertainty
    )
    
    return HLSReport(
        name=name or xml_path.parent.parent.parent.parent.name,
        top_function=top_function,
        part=part,
        product_family=product_family,
        hls_version=hls_version,
        pipeline_type=pipeline_type,
        has_timing_violation=has_violation,
        timing=timing,
        latency=latency,
        resources=resources
    )


def print_report(report: HLSReport, verbose: bool = False):
    """Print HLS report in human-readable format"""
    print("=" * 70)
    print(f"  {report.name}")
    print("=" * 70)
    
    print(f"\nTop Function: {report.top_function}")
    print(f"Target Device: {report.part} ({report.product_family})")
    print(f"HLS Version: {report.hls_version}")
    
    print(f"\n--- Timing ---")
    print(f"  Target Clock:    {report.timing.target_clock_ns:.2f} ns ({report.timing.target_freq_mhz:.1f} MHz)")
    print(f"  Estimated Clock: {report.timing.estimated_clock_ns:.3f} ns ({report.timing.estimated_freq_mhz:.1f} MHz)")
    print(f"  Slack:           {report.timing.slack_ns:.3f} ns")
    if report.has_timing_violation:
        print(f"  ⚠️  TIMING VIOLATION DETECTED")
    
    print(f"\n--- Latency ---")
    print(f"  Best Case:   {report.latency.best_cycles:,} cycles ({report.latency.best_realtime})")
    print(f"  Worst Case:  {report.latency.worst_cycles:,} cycles ({report.latency.worst_realtime})")
    print(f"  II (min):    {report.latency.interval_min}")
    print(f"  II (max):    {report.latency.interval_max}")
    print(f"  Pipeline:    {report.pipeline_type}")
    
    print(f"\n--- Resources ---")
    print(f"  DSP:     {report.resources.dsp:>8,} / {report.resources.dsp_avail:>8,} ({report.resources.dsp_pct:5.1f}%)")
    print(f"  FF:      {report.resources.ff:>8,} / {report.resources.ff_avail:>8,} ({report.resources.ff_pct:5.1f}%)")
    print(f"  LUT:     {report.resources.lut:>8,} / {report.resources.lut_avail:>8,} ({report.resources.lut_pct:5.1f}%)")
    print(f"  BRAM:    {report.resources.bram_18k:>8,} / {report.resources.bram_avail:>8,} ({report.resources.bram_pct:5.1f}%)")
    print(f"  URAM:    {report.resources.uram:>8,} / {report.resources.uram_avail:>8,} ({report.resources.uram_pct:5.1f}%)")
    print()


def print_comparison_table(reports: List[HLSReport]):
    """Print comparison table for multiple reports"""
    if not reports:
        return
    
    print("\n" + "=" * 100)
    print("  HLS IMPLEMENTATION COMPARISON")
    print("=" * 100)
    
    # Header
    names = [r.name for r in reports]
    max_name_len = max(len(n) for n in names)
    
    print(f"\n{'Metric':<25}", end="")
    for name in names:
        print(f"{name:>20}", end="")
    print()
    print("-" * (25 + 20 * len(reports)))
    
    # Timing
    print(f"{'Target Clock (ns)':<25}", end="")
    for r in reports:
        print(f"{r.timing.target_clock_ns:>20.2f}", end="")
    print()
    
    print(f"{'Est. Clock (ns)':<25}", end="")
    for r in reports:
        print(f"{r.timing.estimated_clock_ns:>20.3f}", end="")
    print()
    
    print(f"{'Est. Freq (MHz)':<25}", end="")
    for r in reports:
        print(f"{r.timing.estimated_freq_mhz:>20.1f}", end="")
    print()
    
    # Latency
    print(f"{'Latency (cycles)':<25}", end="")
    for r in reports:
        print(f"{r.latency.worst_cycles:>20,}", end="")
    print()
    
    print(f"{'Latency (realtime)':<25}", end="")
    for r in reports:
        print(f"{r.latency.worst_realtime:>20}", end="")
    print()
    
    print(f"{'II (min)':<25}", end="")
    for r in reports:
        print(f"{r.latency.interval_min:>20}", end="")
    print()
    
    # Resources
    print("-" * (25 + 20 * len(reports)))
    
    print(f"{'DSP':<25}", end="")
    for r in reports:
        print(f"{r.resources.dsp:>20,}", end="")
    print()
    
    print(f"{'DSP (%)':<25}", end="")
    for r in reports:
        print(f"{r.resources.dsp_pct:>19.1f}%", end="")
    print()
    
    print(f"{'FF':<25}", end="")
    for r in reports:
        print(f"{r.resources.ff:>20,}", end="")
    print()
    
    print(f"{'FF (%)':<25}", end="")
    for r in reports:
        print(f"{r.resources.ff_pct:>19.1f}%", end="")
    print()
    
    print(f"{'LUT':<25}", end="")
    for r in reports:
        print(f"{r.resources.lut:>20,}", end="")
    print()
    
    print(f"{'LUT (%)':<25}", end="")
    for r in reports:
        print(f"{r.resources.lut_pct:>19.1f}%", end="")
    print()
    
    print(f"{'BRAM':<25}", end="")
    for r in reports:
        print(f"{r.resources.bram_18k:>20,}", end="")
    print()
    
    print(f"{'URAM':<25}", end="")
    for r in reports:
        print(f"{r.resources.uram:>20,}", end="")
    print()
    
    # Timing violations
    print("-" * (25 + 20 * len(reports)))
    print(f"{'Timing Met':<25}", end="")
    for r in reports:
        status = "❌ NO" if r.has_timing_violation else "✅ YES"
        print(f"{status:>20}", end="")
    print()
    
    print()


def export_to_csv(reports: List[HLSReport], output_path: Path):
    """Export comparison to CSV file"""
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Name', 'Top Function', 'Part',
            'Target Clock (ns)', 'Est. Clock (ns)', 'Est. Freq (MHz)',
            'Latency (cycles)', 'Latency (realtime)', 'II',
            'DSP', 'DSP (%)', 'FF', 'FF (%)', 'LUT', 'LUT (%)',
            'BRAM', 'URAM', 'Timing Met'
        ])
        
        for r in reports:
            writer.writerow([
                r.name, r.top_function, r.part,
                r.timing.target_clock_ns, r.timing.estimated_clock_ns,
                round(r.timing.estimated_freq_mhz, 1),
                r.latency.worst_cycles, r.latency.worst_realtime,
                r.latency.interval_min,
                r.resources.dsp, round(r.resources.dsp_pct, 1),
                r.resources.ff, round(r.resources.ff_pct, 1),
                r.resources.lut, round(r.resources.lut_pct, 1),
                r.resources.bram_18k, r.resources.uram,
                'Yes' if not r.has_timing_violation else 'No'
            ])
    
    print(f"Exported to: {output_path}")


def find_all_hls_projects(base_dir: Path) -> List[Path]:
    """Find all HLS project directories containing synthesis reports"""
    projects = []
    
    for item in base_dir.iterdir():
        if item.is_dir():
            xml_path = find_csynth_xml(item)
            if xml_path:
                projects.append(item)
    
    return sorted(projects)


def main():
    parser = argparse.ArgumentParser(
        description='Parse HLS synthesis reports',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('paths', nargs='*', help='HLS project directories or csynth.xml files')
    parser.add_argument('--all', '-a', action='store_true',
                        help='Parse all HLS projects in the given directory')
    parser.add_argument('--json', '-j', action='store_true',
                        help='Output in JSON format')
    parser.add_argument('--csv', '-c', type=str, metavar='FILE',
                        help='Export comparison to CSV file')
    parser.add_argument('--output', '-o', type=str, metavar='FILE',
                        help='Output file (for JSON)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    if not args.paths:
        parser.print_help()
        return 1
    
    reports = []
    
    if args.all:
        # Find all projects in directory
        base_dir = Path(args.paths[0])
        if not base_dir.is_dir():
            print(f"Error: {base_dir} is not a directory", file=sys.stderr)
            return 1
        
        projects = find_all_hls_projects(base_dir)
        if not projects:
            print(f"No HLS projects found in {base_dir}", file=sys.stderr)
            return 1
        
        for project in projects:
            xml_path = find_csynth_xml(project)
            if xml_path:
                report = parse_csynth_xml(xml_path)
                if report:
                    reports.append(report)
    else:
        # Parse specified paths
        for path_str in args.paths:
            path = Path(path_str)
            
            if path.suffix == '.xml':
                xml_path = path
            else:
                xml_path = find_csynth_xml(path)
            
            if xml_path is None:
                print(f"Warning: No csynth.xml found in {path}", file=sys.stderr)
                continue
            
            report = parse_csynth_xml(xml_path)
            if report:
                reports.append(report)
    
    if not reports:
        print("No reports parsed successfully", file=sys.stderr)
        return 1
    
    # Output
    if args.json:
        data = [r.to_dict() for r in reports]
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Exported to: {args.output}")
        else:
            print(json.dumps(data, indent=2))
    elif args.csv:
        export_to_csv(reports, Path(args.csv))
    else:
        if len(reports) == 1:
            print_report(reports[0], verbose=args.verbose)
        else:
            print_comparison_table(reports)
            if args.verbose:
                for report in reports:
                    print_report(report, verbose=True)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
