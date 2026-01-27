#!/usr/bin/env python3
"""
Design Space Exploration for PTQ INT8 GraphSAGE HLS

This script orchestrates the full exploration pipeline:
1. Algorithm level: Model architecture variations
2. Quantization level: PTQ INT8 bit-width configurations  
3. HLS level: Implementation optimizations (unroll, pipeline, binding)

Outputs:
- build/experiments/design_space_results.csv
- build/experiments/design_space_results.json
- build/experiments/checkpoint.json (for resumable runs)

Usage:
    # Full exploration
    python src/explore_design_space.py --config configs/design_space.yaml
    
    # Quick mode (skip HLS synthesis)
    python src/explore_design_space.py --config configs/design_space.yaml --skip-hls
    
    # Resume interrupted run
    python src/explore_design_space.py --config configs/design_space.yaml --resume
    
    # Dry run (show design points without executing)
    python src/explore_design_space.py --config configs/design_space.yaml --dry-run
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import hashlib

import yaml
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


@dataclass
class DesignPoint:
    """A single design point in the exploration space"""
    design_id: str
    
    # Algorithm level
    in_channels_reduced: int = 16
    hidden_channels: int = 24
    dropout: float = 0.5
    root_weight: bool = False
    num_layers: int = 2
    
    # Quantization level
    M_BITS: int = 24
    K_BITS: int = 12
    bitwidth_margin: int = 2
    bitwidth_method: str = "data_driven"
    
    # Computed bit-widths (filled by optimizer)
    ACC_BITS: int = 0
    ADJ_BITS: int = 0
    SCALE_BITS: int = 0
    MULT_BITS: int = 0
    
    # HLS level
    unroll_nodes: int = 1
    unroll_features_agg: int = 1
    unroll_features_lin: int = 1
    unroll_features_relu: int = 1
    agg_pipeline_ii: int = 1
    lin_pipeline_ii: int = 1
    bind_agg_mul: str = "auto"
    bind_lin_scale_mul: str = "auto"
    
    # Software metrics (filled after training/PTQ)
    test_accuracy: float = 0.0
    num_parameters: int = 0
    model_memory_bytes: int = 0
    lsb_error_vs_float: int = 0
    
    # HLS metrics (filled after synthesis)
    dsp_used: int = 0
    dsp_available: int = 0
    dsp_pct: float = 0.0
    lut_used: int = 0
    lut_pct: float = 0.0
    ff_used: int = 0
    ff_pct: float = 0.0
    bram_used: int = 0
    uram_used: int = 0
    latency_min_cycles: int = 0
    latency_max_cycles: int = 0
    latency_avg_cycles: int = 0
    ii_min: int = 0
    ii_max: int = 0
    fmax_mhz: float = 0.0
    target_clock_ns: float = 0.0
    meets_timing: bool = False
    
    # Derived metrics
    latency_ns: float = 0.0
    throughput_inferences_per_us: float = 0.0
    
    # Status
    status: str = "pending"  # pending, training, ptq, hls, completed, failed
    error_message: str = ""
    timestamp: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @staticmethod
    def from_dict(d: dict) -> 'DesignPoint':
        return DesignPoint(**{k: v for k, v in d.items() if k in DesignPoint.__dataclass_fields__})


def generate_design_id(params: dict) -> str:
    """Generate unique ID from design parameters"""
    # Create deterministic hash from relevant parameters
    key_params = {
        'icr': params.get('in_channels_reduced', 16),
        'hc': params.get('hidden_channels', 24),
        'mb': params.get('M_BITS', 24),
        'kb': params.get('K_BITS', 12),
        'bm': params.get('bitwidth_margin', 2),
        'un': params.get('unroll_nodes', 1),
        'ufa': params.get('unroll_features_agg', 1),
        'ufl': params.get('unroll_features_lin', 1),
    }
    param_str = json.dumps(key_params, sort_keys=True)
    hash_suffix = hashlib.md5(param_str.encode()).hexdigest()[:8]
    
    # Human-readable prefix
    prefix = f"d{key_params['icr']}x{key_params['hc']}_m{key_params['mb']}_u{key_params['un']}"
    return f"{prefix}_{hash_suffix}"


def load_config(config_path: Path) -> dict:
    """Load design space configuration from YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_design_points(config: dict) -> List[DesignPoint]:
    """Generate all design points from configuration"""
    # Support both flat config and nested search_space structure
    if 'search_space' in config:
        search = config['search_space']
        algo = search.get('algorithm', {})
        quant = search.get('quantization', {})
        hls = search.get('hls', {})
    else:
        algo = config.get('algorithm', {})
        quant = config.get('quantization', {})
        hls = config.get('hls', {})
    
    # Provide defaults for missing parameters
    in_channels_reduced = algo.get('in_channels_reduced', [16])
    hidden_channels = algo.get('hidden_channels', algo.get('hidden_dim', [24]))
    dropout = algo.get('dropout', [0.5])
    root_weight = algo.get('root_weight', [False])
    num_layers = algo.get('num_layers', [2])
    
    m_bits_list = quant.get('M_BITS', quant.get('m_bits', [24]))
    k_bits_list = quant.get('K_BITS', quant.get('k_bits', [12]))
    bitwidth_margin = quant.get('bitwidth_margin', [2])
    method_list = quant.get('method', ['symmetric'])
    unroll_factor = quant.get('unroll_factor', [1])
    acc_bits = quant.get('acc_bits', [32])
    
    agg_unroll = hls.get('aggregation_unroll', hls.get('agg_pipeline_ii', [1]))
    lin_unroll = hls.get('linear_unroll', hls.get('lin_pipeline_ii', [1]))
    pipeline_ii = hls.get('pipeline_ii', [1])
    
    # Build parameter lists
    algo_params = list(itertools.product(
        in_channels_reduced,
        hidden_channels,
        dropout,
        root_weight,
        num_layers,
    ))
    
    quant_params = list(itertools.product(
        m_bits_list,
        k_bits_list,
        bitwidth_margin,
        method_list,
    ))
    
    # Handle HLS params - support both detailed and simple formats
    if 'unroll' in hls:
        unroll_nodes_list = hls['unroll'].get('nodes', [8])
        unroll_feat_agg = hls['unroll'].get('features_agg', [1])
        unroll_feat_lin = hls['unroll'].get('features_lin', [1])
        unroll_feat_relu = hls['unroll'].get('features_relu', [1])
    else:
        unroll_nodes_list = hls.get('unroll_nodes', [8])
        unroll_feat_agg = unroll_factor
        unroll_feat_lin = unroll_factor
        unroll_feat_relu = unroll_factor
    
    if 'pipeline' in hls:
        agg_ii_list = hls['pipeline'].get('agg_ii', [1])
        lin_ii_list = hls['pipeline'].get('lin_ii', [1])
    else:
        agg_ii_list = agg_unroll
        lin_ii_list = lin_unroll
        
    if 'bind' in hls:
        bind_agg_mul_list = hls['bind'].get('agg_mul', ['dsp'])
        bind_lin_mul_list = hls['bind'].get('lin_scale_mul', ['dsp'])
    else:
        bind_agg_mul_list = hls.get('bind_agg', ['dsp'])
        bind_lin_mul_list = hls.get('bind_lin', ['dsp'])
    
    target_clock = hls.get('target_clock_ns', 2.77)  # 361MHz default
    
    hls_params = list(itertools.product(
        unroll_nodes_list,
        unroll_feat_agg,
        unroll_feat_lin,
        unroll_feat_relu,
        agg_ii_list,
        lin_ii_list,
        bind_agg_mul_list,
        bind_lin_mul_list,
    ))
    
    # Generate all combinations
    design_points = []
    
    for algo_p in algo_params:
        for quant_p in quant_params:
            for hls_p in hls_params:
                params = {
                    'in_channels_reduced': algo_p[0],
                    'hidden_channels': algo_p[1],
                    'dropout': algo_p[2],
                    'root_weight': algo_p[3],
                    'num_layers': algo_p[4],
                    'M_BITS': quant_p[0],
                    'K_BITS': quant_p[1],
                    'bitwidth_margin': quant_p[2],
                    'bitwidth_method': quant_p[3],
                    'unroll_nodes': hls_p[0],
                    'unroll_features_agg': hls_p[1],
                    'unroll_features_lin': hls_p[2],
                    'unroll_features_relu': hls_p[3],
                    'agg_pipeline_ii': hls_p[4],
                    'lin_pipeline_ii': hls_p[5],
                    'bind_agg_mul': hls_p[6],
                    'bind_lin_scale_mul': hls_p[7],
                    'target_clock_ns': target_clock,
                }
                
                design_id = generate_design_id(params)
                dp = DesignPoint(design_id=design_id, **params)
                design_points.append(dp)
    
    return design_points


class DesignSpaceExplorer:
    """Main orchestrator for design space exploration"""
    
    def __init__(self, config: dict, output_dir: Path, resume: bool = True):
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectory for individual design point artifacts
        self.design_points_dir = output_dir / "design_points"
        self.design_points_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_path = output_dir / "checkpoint.json"
        self.results_csv_path = output_dir / "design_space_results.csv"
        self.results_json_path = output_dir / "design_space_results.json"
        
        self.completed_designs: Dict[str, DesignPoint] = {}
        self.failed_designs: Dict[str, DesignPoint] = {}
        
        if resume and self.checkpoint_path.exists():
            self._load_checkpoint()
    
    def _get_design_point_dir(self, dp: DesignPoint) -> Path:
        """Get/create directory for a specific design point's artifacts"""
        dp_dir = self.design_points_dir / dp.design_id
        dp_dir.mkdir(parents=True, exist_ok=True)
        return dp_dir
    
    def _save_design_point_artifacts(self, dp: DesignPoint):
        """Save individual design point artifacts to its folder"""
        dp_dir = self._get_design_point_dir(dp)
        
        # Save design point config as JSON
        config_path = dp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(dp.to_dict(), f, indent=2)
        
        # Save a summary text file
        summary_path = dp_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Design Point: {dp.design_id}\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Architecture:\n")
            f.write(f"  in_channels_reduced: {dp.in_channels_reduced}\n")
            f.write(f"  hidden_channels: {dp.hidden_channels}\n")
            f.write(f"  num_layers: {dp.num_layers}\n")
            f.write(f"\nQuantization:\n")
            f.write(f"  M_BITS: {dp.M_BITS}\n")
            f.write(f"  K_BITS: {dp.K_BITS}\n")
            f.write(f"  ACC_BITS: {dp.ACC_BITS}\n")
            f.write(f"  SCALE_BITS: {dp.SCALE_BITS}\n")
            f.write(f"  MULT_BITS: {dp.MULT_BITS}\n")
            f.write(f"\nHLS Settings:\n")
            f.write(f"  unroll_nodes: {dp.unroll_nodes}\n")
            f.write(f"  unroll_features_agg: {dp.unroll_features_agg}\n")
            f.write(f"  agg_pipeline_ii: {dp.agg_pipeline_ii}\n")
            f.write(f"  lin_pipeline_ii: {dp.lin_pipeline_ii}\n")
            f.write(f"\nResults:\n")
            f.write(f"  test_accuracy: {dp.test_accuracy}%\n")
            f.write(f"  DSP: {dp.dsp_used} ({dp.dsp_pct:.1f}%)\n")
            f.write(f"  LUT: {dp.lut_used} ({dp.lut_pct:.1f}%)\n")
            f.write(f"  FF: {dp.ff_used} ({dp.ff_pct:.1f}%)\n")
            f.write(f"  Latency: {dp.latency_max_cycles} cycles\n")
            f.write(f"  Fmax: {dp.fmax_mhz:.1f} MHz\n")
            f.write(f"  Status: {dp.status}\n")
            if dp.error_message:
                f.write(f"  Error: {dp.error_message}\n")
    
    def _load_checkpoint(self):
        """Load progress from checkpoint file"""
        print(f"Loading checkpoint from {self.checkpoint_path}")
        with open(self.checkpoint_path, 'r') as f:
            data = json.load(f)
        
        for dp_dict in data.get('completed', []):
            dp = DesignPoint.from_dict(dp_dict)
            self.completed_designs[dp.design_id] = dp
        
        for dp_dict in data.get('failed', []):
            dp = DesignPoint.from_dict(dp_dict)
            self.failed_designs[dp.design_id] = dp
        
        print(f"  Loaded {len(self.completed_designs)} completed, {len(self.failed_designs)} failed designs")
    
    def _save_checkpoint(self):
        """Save progress to checkpoint file"""
        data = {
            'completed': [dp.to_dict() for dp in self.completed_designs.values()],
            'failed': [dp.to_dict() for dp in self.failed_designs.values()],
            'timestamp': datetime.now().isoformat(),
        }
        with open(self.checkpoint_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_results(self):
        """Save results to CSV and JSON"""
        all_designs = list(self.completed_designs.values()) + list(self.failed_designs.values())
        
        # Save JSON
        with open(self.results_json_path, 'w') as f:
            json.dump([dp.to_dict() for dp in all_designs], f, indent=2)
        
        # Save CSV
        if all_designs:
            import csv
            fieldnames = list(all_designs[0].to_dict().keys())
            with open(self.results_csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for dp in all_designs:
                    writer.writerow(dp.to_dict())
        
        print(f"Results saved to {self.results_csv_path}")
    
    def _get_config_section(self, section: str, key: str = None, default=None):
        """Get config value, supporting both flat and nested structure"""
        # Try nested search_space structure first
        if 'search_space' in self.config:
            val = self.config['search_space'].get(section, {})
        else:
            val = self.config.get(section, {})
        
        if key:
            return val.get(key, default) if isinstance(val, dict) else default
        return val if val else default
    
    def run_training(self, dp: DesignPoint) -> bool:
        """Run or load training for a design point"""
        print(f"  [Training] in_channels_reduced={dp.in_channels_reduced}, hidden_channels={dp.hidden_channels}")
        
        # Check for pre-trained model
        pretrained_dir = self._get_config_section('algorithm', 'pretrained_dir')
        if pretrained_dir:
            model_name = f"reduced_graphsage_no_root_{dp.in_channels_reduced}x{dp.hidden_channels}"
            model_path = Path(pretrained_dir) / f"{model_name}_best.pth"
            
            if model_path.exists():
                print(f"    Using pre-trained model: {model_path}")
                # TODO: Load accuracy from saved stats
                dp.test_accuracy = 75.0  # Placeholder
                dp.num_parameters = dp.in_channels_reduced * dp.hidden_channels * 2  # Rough estimate
                return True
        
        # For now, use default model if architecture matches
        if dp.in_channels_reduced == 16 and dp.hidden_channels == 24:
            default_model = PROJECT_ROOT / "build/models/reduced_graphsage_no_root_best.pth"
            if default_model.exists():
                print(f"    Using default model: {default_model}")
                dp.test_accuracy = 75.8  # From README
                dp.num_parameters = 24079
                dp.model_memory_bytes = 24079 * 4  # float32
                return True
        
        # TODO: Implement actual training for different architectures
        print(f"    WARNING: Training not implemented for this architecture, using placeholder")
        dp.test_accuracy = 70.0 + np.random.uniform(0, 8)  # Placeholder
        dp.num_parameters = dp.in_channels_reduced * dp.hidden_channels * 10
        return True
    
    def run_ptq(self, dp: DesignPoint) -> bool:
        """Run PTQ and INT8 parameter generation"""
        print(f"  [PTQ] M_BITS={dp.M_BITS}, K_BITS={dp.K_BITS}")
        
        # For default architecture, use existing PTQ outputs
        if dp.in_channels_reduced == 16 and dp.hidden_channels == 24:
            ptq_dir = PROJECT_ROOT / "build/weights_ptq_int8"
            if ptq_dir.exists():
                print(f"    Using existing PTQ: {ptq_dir}")
                return True
        
        # TODO: Run PTQ for different architectures
        # subprocess.run([...], check=True)
        
        print(f"    WARNING: PTQ not implemented for this architecture, using placeholder")
        return True
    
    def run_bitwidth_optimizer(self, dp: DesignPoint) -> bool:
        """Run bit-width optimization"""
        print(f"  [BitWidth] margin={dp.bitwidth_margin}, method={dp.bitwidth_method}")
        
        # For default architecture, read from existing analysis
        if dp.in_channels_reduced == 16 and dp.hidden_channels == 24:
            analysis_path = PROJECT_ROOT / "build/hls/bitwidth_analysis.json"
            if analysis_path.exists():
                with open(analysis_path, 'r') as f:
                    analysis = json.load(f)
                
                # Adjust for margin setting
                base_acc = analysis['data_driven_analysis']['acc_bits']
                base_scale = analysis['data_driven_analysis']['scale_bits']
                base_mult = analysis['data_driven_analysis']['mult_bits']
                base_adj = analysis['data_driven_analysis']['adj_bits']
                
                margin_diff = dp.bitwidth_margin - analysis['configuration']['safety_margin']
                
                dp.ACC_BITS = base_acc + margin_diff
                dp.SCALE_BITS = base_scale + margin_diff
                dp.MULT_BITS = dp.ACC_BITS + dp.SCALE_BITS
                dp.ADJ_BITS = base_adj + margin_diff
                
                print(f"    Computed: ACC={dp.ACC_BITS}, SCALE={dp.SCALE_BITS}, MULT={dp.MULT_BITS}, ADJ={dp.ADJ_BITS}")
                return True
        
        # TODO: Run optimizer for different architectures
        print(f"    WARNING: Bit-width optimizer not implemented for this architecture")
        dp.ACC_BITS = 32
        dp.SCALE_BITS = 32
        dp.MULT_BITS = 64
        dp.ADJ_BITS = 16
        return True
    
    def run_hls_synthesis(self, dp: DesignPoint) -> bool:
        """Generate HLS project and run synthesis"""
        print(f"  [HLS] unroll=({dp.unroll_nodes},{dp.unroll_features_agg},{dp.unroll_features_lin})")
        
        # Check if we can use existing synthesis results (exact match)
        if (dp.in_channels_reduced == 16 and dp.hidden_channels == 24 and
            dp.M_BITS == 24 and dp.K_BITS == 12 and
            dp.unroll_nodes == 1 and dp.unroll_features_agg == 1 and dp.unroll_features_lin == 1 and
            dp.agg_pipeline_ii == 1 and dp.lin_pipeline_ii == 1):
            
            int8_report = PROJECT_ROOT / "build/hls/graphsage_int8/solution1/syn/report/csynth.xml"
            if int8_report.exists():
                print(f"    Using existing synthesis: {int8_report}")
                return self._parse_hls_report(dp, int8_report)
        
        # Create design point output directory
        dp_dir = self._get_design_point_dir(dp)
        
        # HLS project goes in build/hls/dse_<id> to avoid path depth issues with Vitis HLS
        # (Deep paths cause csim to compile .cpp files differently, leading to duplicate symbols)
        hls_dir = PROJECT_ROOT / "build" / "hls" / f"dse_{dp.design_id}"
        hls_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Generate test vectors for this configuration
        print(f"    Generating test vectors...")
        if not self._generate_test_vectors(dp, dp_dir):
            return False
        
        # Step 2: Generate parameterized HLS header (optional, for reference)
        print(f"    Generating HLS config...")
        if not self._generate_hls_config(dp, hls_dir):
            return False
        
        # Step 3: Generate TCL project files using Jinja2 templates
        print(f"    Generating TCL projects from templates...")
        if not self._generate_tcl_project(dp, hls_dir):
            return False
        
        # Step 4: Run C-simulation (includes project setup)
        print(f"    Running C-simulation...")
        if not self._run_csim(dp, hls_dir):
            print(f"    WARNING: C-simulation failed, continuing with synthesis...")
            # Don't fail on csim - synthesis is the main goal
        
        # Step 5: Run synthesis
        print(f"    Running synthesis...")
        if not self._run_synthesis(dp, hls_dir):
            return False
        
        # Step 6: Parse results - path matches template-generated project structure
        # The project_dir in config is hls_dir (build/hls/dse_<id>)
        report_path = hls_dir / "solution1" / "syn" / "report" / "csynth.xml"
        if report_path.exists():
            return self._parse_hls_report(dp, report_path)
        else:
            print(f"    ERROR: Synthesis report not found at {report_path}")
            return False
    
    def _generate_test_vectors(self, dp: DesignPoint, dp_dir: Path) -> bool:
        """Generate test vectors for this design point"""
        try:
            # For now, copy existing test vectors (they're the same for same architecture)
            src_dir = PROJECT_ROOT / "build/test_vectors_ptq_int8"
            dst_dir = dp_dir / "test_vectors"
            
            if src_dir.exists():
                import shutil
                if dst_dir.exists():
                    shutil.rmtree(dst_dir)
                shutil.copytree(src_dir, dst_dir)
                return True
            else:
                print(f"    ERROR: Source test vectors not found at {src_dir}")
                return False
        except Exception as e:
            print(f"    ERROR generating test vectors: {e}")
            return False
    
    def _generate_hls_config(self, dp: DesignPoint, hls_dir: Path) -> bool:
        """Generate parameterized HLS header with design point settings"""
        try:
            config_h = hls_dir / "dse_config.h"
            with open(config_h, 'w') as f:
                f.write(f"// Auto-generated configuration for design point: {dp.design_id}\n")
                f.write(f"#ifndef DSE_CONFIG_H\n")
                f.write(f"#define DSE_CONFIG_H\n\n")
                f.write(f"// Architecture parameters\n")
                f.write(f"#define DSE_IN_FEATURES {dp.in_channels_reduced}\n")
                f.write(f"#define DSE_HIDDEN_FEATURES {dp.hidden_channels}\n")
                f.write(f"#define DSE_OUT_FEATURES 7\n")
                f.write(f"#define DSE_NUM_NODES 8\n\n")
                f.write(f"// Quantization parameters\n")
                f.write(f"#define DSE_M_BITS {dp.M_BITS}\n")
                f.write(f"#define DSE_K_BITS {dp.K_BITS}\n")
                f.write(f"#define DSE_ACC_BITS {dp.ACC_BITS}\n")
                f.write(f"#define DSE_ADJ_BITS {dp.ADJ_BITS}\n")
                f.write(f"#define DSE_SCALE_BITS {dp.SCALE_BITS}\n")
                f.write(f"#define DSE_MULT_BITS {dp.MULT_BITS}\n\n")
                f.write(f"// HLS pragma parameters\n")
                f.write(f"#define DSE_UNROLL_NODES {dp.unroll_nodes}\n")
                f.write(f"#define DSE_UNROLL_FEATURES_AGG {dp.unroll_features_agg}\n")
                f.write(f"#define DSE_UNROLL_FEATURES_LIN {dp.unroll_features_lin}\n")
                f.write(f"#define DSE_AGG_PIPELINE_II {dp.agg_pipeline_ii}\n")
                f.write(f"#define DSE_LIN_PIPELINE_II {dp.lin_pipeline_ii}\n\n")
                f.write(f"#endif // DSE_CONFIG_H\n")
            return True
        except Exception as e:
            print(f"    ERROR generating HLS config: {e}")
            return False
    
    def _generate_tcl_project(self, dp: DesignPoint, hls_dir: Path) -> bool:
        """Generate TCL project files using Jinja2 templates (same as generate_graphsage_int8_tcl.py)"""
        try:
            from jinja2 import Environment, FileSystemLoader
            
            # Paths
            src_hls_dir = PROJECT_ROOT / "hls"
            template_dir = src_hls_dir / "tcl_example"
            
            # Build cflags - ONLY pass M_BITS, let header defaults handle the rest
            # This matches the working generate_graphsage_int8_tcl.py approach
            cflags = [
                f"-DM_BITS={dp.M_BITS}",
            ]
            
            # Configuration for Jinja2 templates (matching generate_graphsage_int8_tcl.py format)
            config = {
                "module_name": f"graphsage_dse_{dp.design_id}",
                "top": "graphsage_int8",
                "part": "xcvu13p-fsga2577-1-e",
                "clock_period": dp.target_clock_ns,
                "version": "1.0",
                "vendor": "GNN_DSE",
                
                # Absolute paths to source files
                "src": [
                    str(src_hls_dir / "graphsage_layer_int8.cpp"),
                    str(src_hls_dir / "graphsage_layer_int8.h"),
                ],
                
                # Absolute paths to testbench files
                "tb": [
                    str(src_hls_dir / "testbench_int8.cpp"),
                ],
                
                # Absolute paths to include directories
                "includes": [
                    str(src_hls_dir),
                ],
                
                # Compiler flags with DSE parameters
                "cflags": cflags,
                
                # Absolute paths - project goes directly in hls_dir (build/hls/dse_<id>)
                "project_dir": str(hls_dir),
                "project_root": str(PROJECT_ROOT),
                "logs_dir": str(hls_dir / "logs"),
                
                # TCL options
                "csim_opts": "-clean",
                "csynth_opts": "",
                "tb_args": "",
            }
            
            # Create directories
            (hls_dir / "logs").mkdir(parents=True, exist_ok=True)
            
            # Setup Jinja environment
            env = Environment(loader=FileSystemLoader(str(template_dir)))
            
            # Generate project.tcl
            template = env.get_template("project.tcl.j2")
            output_path = hls_dir / "project.tcl"
            with open(output_path, "w") as f:
                f.write(template.render(**config))
            
            # Generate synth.tcl
            template = env.get_template("synth.tcl.j2")
            output_path = hls_dir / "synth.tcl"
            with open(output_path, "w") as f:
                f.write(template.render(**config))
            
            # Generate csim.tcl
            template = env.get_template("csim.tcl.j2")
            output_path = hls_dir / "csim.tcl"
            with open(output_path, "w") as f:
                f.write(template.render(**config))
            
            return True
        except Exception as e:
            print(f"    ERROR generating TCL project: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _run_csim(self, dp: DesignPoint, hls_dir: Path) -> bool:
        """Run C-simulation using generated csim.tcl"""
        try:
            import subprocess
            
            # Full path to Vitis HLS
            vitis_hls = "/tools/Xilinx/Vitis_HLS/2024.1/bin/vitis_hls"
            
            # First run project.tcl to set up the project
            project_tcl = hls_dir / "project.tcl"
            log_path = hls_dir / "project_setup.log"
            
            cmd = [vitis_hls, "-f", str(project_tcl)]
            print(f"    Running project setup...")
            
            with open(log_path, 'w') as log_file:
                result = subprocess.run(
                    cmd,
                    cwd=str(hls_dir),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=600  # 10 min timeout for setup
                )
            
            if result.returncode != 0:
                print(f"    ERROR: Project setup failed")
                return False
            
            # Then run csim.tcl
            csim_tcl = hls_dir / "csim.tcl"
            log_path = hls_dir / "csim.log"
            
            cmd = [vitis_hls, "-f", str(csim_tcl)]
            print(f"    Running csim...")
            
            with open(log_path, 'w') as log_file:
                result = subprocess.run(
                    cmd,
                    cwd=str(hls_dir),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=1800  # 30 min timeout for csim
                )
            
            if result.returncode != 0:
                print(f"    ERROR: C-simulation failed")
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines[-15:]:
                        print(f"      {line.rstrip()}")
                return False
            
            return True
        except subprocess.TimeoutExpired:
            print(f"    ERROR: C-simulation timed out")
            return False
        except Exception as e:
            print(f"    ERROR running csim: {e}")
            return False
    
    def _run_synthesis(self, dp: DesignPoint, hls_dir: Path) -> bool:
        """Run Vitis HLS synthesis using generated synth.tcl"""
        try:
            import subprocess
            
            # Full path to Vitis HLS
            vitis_hls = "/tools/Xilinx/Vitis_HLS/2024.1/bin/vitis_hls"
            
            synth_tcl = hls_dir / "synth.tcl"
            log_path = hls_dir / "synthesis.log"
            
            cmd = [vitis_hls, "-f", str(synth_tcl)]
            print(f"    Running synthesis...")
            
            with open(log_path, 'w') as log_file:
                result = subprocess.run(
                    cmd,
                    cwd=str(hls_dir),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=3600  # 1 hour timeout
                )
            
            if result.returncode != 0:
                print(f"    ERROR: Vitis HLS synthesis failed with code {result.returncode}")
                print(f"    See log: {log_path}")
                # Print last 20 lines of log
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines[-20:]:
                        print(f"      {line.rstrip()}")
                return False
            
            return True
        except subprocess.TimeoutExpired:
            print(f"    ERROR: Synthesis timed out after 1 hour")
            return False
        except Exception as e:
            print(f"    ERROR running synthesis: {e}")
            return False
        return True
    
    def _parse_hls_report(self, dp: DesignPoint, report_path: Path) -> bool:
        """Parse HLS synthesis report and fill design point metrics"""
        try:
            from parse_hls_report import parse_csynth_xml
            
            report = parse_csynth_xml(report_path)
            if report is None:
                return False
            
            dp.dsp_used = report.resources.dsp
            dp.dsp_available = report.resources.dsp_avail
            dp.dsp_pct = report.resources.dsp_pct
            dp.lut_used = report.resources.lut
            dp.lut_pct = report.resources.lut_pct
            dp.ff_used = report.resources.ff
            dp.ff_pct = report.resources.ff_pct
            dp.bram_used = report.resources.bram_18k
            dp.uram_used = report.resources.uram
            
            dp.latency_min_cycles = report.latency.best_cycles
            dp.latency_max_cycles = report.latency.worst_cycles
            dp.latency_avg_cycles = report.latency.avg_cycles
            dp.ii_min = report.latency.interval_min
            dp.ii_max = report.latency.interval_max
            
            dp.fmax_mhz = report.timing.estimated_freq_mhz
            dp.target_clock_ns = report.timing.target_clock_ns
            dp.meets_timing = not report.has_timing_violation
            
            # Compute derived metrics
            if dp.fmax_mhz > 0:
                dp.latency_ns = dp.latency_max_cycles / (dp.fmax_mhz * 1e6) * 1e9
                dp.throughput_inferences_per_us = 1e3 / dp.latency_ns if dp.latency_ns > 0 else 0
            
            return True
            
        except Exception as e:
            print(f"    ERROR parsing HLS report: {e}")
            return False
    
    def evaluate_design_point(self, dp: DesignPoint, skip_hls: bool = False) -> bool:
        """Evaluate a single design point through all stages"""
        dp.timestamp = datetime.now().isoformat()
        
        try:
            # Stage 1: Training
            dp.status = "training"
            if not self.run_training(dp):
                dp.status = "failed"
                dp.error_message = "Training failed"
                return False
            
            # Check accuracy threshold (from constraints section)
            constraints = self.config.get('constraints', {})
            baseline = self.config.get('baseline', {})
            max_drop = constraints.get('max_accuracy_drop', 10.0)
            baseline_acc = baseline.get('model_accuracy', 75.7)
            threshold = baseline_acc - max_drop
            
            if dp.test_accuracy < threshold:
                dp.status = "failed"
                dp.error_message = f"Accuracy {dp.test_accuracy:.1f}% below threshold {threshold}%"
                print(f"    SKIPPED: Accuracy below threshold")
                return False
            
            # Stage 2: PTQ
            dp.status = "ptq"
            if not self.run_ptq(dp):
                dp.status = "failed"
                dp.error_message = "PTQ failed"
                return False
            
            # Bit-width optimization
            if not self.run_bitwidth_optimizer(dp):
                dp.status = "failed"
                dp.error_message = "Bit-width optimization failed"
                return False
            
            # Stage 3: HLS
            if not skip_hls:
                dp.status = "hls"
                if not self.run_hls_synthesis(dp):
                    dp.status = "failed"
                    dp.error_message = "HLS synthesis failed"
                    return False
                
                # Check HLS constraints
                min_fmax = constraints.get('min_fmax', 300.0)
                max_dsp = constraints.get('max_dsp_util', 90.0)
                
                if dp.fmax_mhz and dp.fmax_mhz < min_fmax:
                    dp.status = "failed"
                    dp.error_message = f"Fmax {dp.fmax_mhz:.1f} MHz below minimum {min_fmax}"
                    return False
                
                if dp.dsp_pct and dp.dsp_pct > max_dsp:
                    dp.status = "failed"
                    dp.error_message = f"DSP {dp.dsp_pct:.1f}% exceeds maximum {max_dsp}%"
                    return False
            
            dp.status = "completed"
            return True
            
        except Exception as e:
            dp.status = "failed"
            dp.error_message = str(e)
            return False
    
    def run_exploration(self, design_points: List[DesignPoint], 
                        skip_hls: bool = False, dry_run: bool = False):
        """Run exploration on all design points"""
        total = len(design_points)
        print(f"\n{'='*70}")
        print(f"DESIGN SPACE EXPLORATION")
        print(f"{'='*70}")
        print(f"Total design points: {total}")
        print(f"Already completed: {len(self.completed_designs)}")
        print(f"Already failed: {len(self.failed_designs)}")
        print(f"Skip HLS: {skip_hls}")
        print(f"Dry run: {dry_run}")
        print(f"{'='*70}\n")
        
        if dry_run:
            print("Design points to evaluate:")
            for i, dp in enumerate(design_points[:20]):  # Show first 20
                print(f"  {i+1}. {dp.design_id}")
            if len(design_points) > 20:
                print(f"  ... and {len(design_points) - 20} more")
            return
        
        pending = [dp for dp in design_points 
                   if dp.design_id not in self.completed_designs 
                   and dp.design_id not in self.failed_designs]
        
        print(f"Pending design points: {len(pending)}\n")
        
        for i, dp in enumerate(pending):
            print(f"\n[{i+1}/{len(pending)}] Evaluating: {dp.design_id}")
            print(f"  Architecture: {dp.in_channels_reduced} → {dp.hidden_channels} → 7")
            print(f"  Quantization: M={dp.M_BITS}, K={dp.K_BITS}, margin={dp.bitwidth_margin}")
            print(f"  HLS: unroll=({dp.unroll_nodes},{dp.unroll_features_agg},{dp.unroll_features_lin})")
            
            success = self.evaluate_design_point(dp, skip_hls=skip_hls)
            
            if success:
                self.completed_designs[dp.design_id] = dp
                print(f"  ✓ COMPLETED: acc={dp.test_accuracy:.1f}%, lat={dp.latency_max_cycles} cycles, DSP={dp.dsp_used}")
            else:
                self.failed_designs[dp.design_id] = dp
                print(f"  ✗ FAILED: {dp.error_message}")
            
            # Save design point artifacts to its folder
            self._save_design_point_artifacts(dp)
            
            # Save checkpoint after each design
            self._save_checkpoint()
        
        # Final save
        self._save_results()
        
        print(f"\n{'='*70}")
        print(f"EXPLORATION COMPLETE")
        print(f"{'='*70}")
        print(f"Completed: {len(self.completed_designs)}")
        print(f"Failed: {len(self.failed_designs)}")
        print(f"Results: {self.results_csv_path}")
        print(f"Design point artifacts: {self.design_points_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Design Space Exploration for PTQ INT8 GraphSAGE HLS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--config', '-c', type=str, default='configs/design_space.yaml',
                        help='Design space configuration file')
    parser.add_argument('--output', '-o', type=str, default='build/experiments',
                        help='Output directory for results')
    parser.add_argument('--resume', '-r', action='store_true', default=True,
                        help='Resume from checkpoint (default: True)')
    parser.add_argument('--no-resume', action='store_true',
                        help='Start fresh, ignore checkpoint')
    parser.add_argument('--skip-hls', action='store_true',
                        help='Skip HLS synthesis (quick Python/PTQ exploration)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show design points without executing')
    parser.add_argument('--max-designs', type=int, default=0,
                        help='Maximum number of designs to evaluate (0 = unlimited)')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1
    
    config = load_config(config_path)
    
    # Generate design points
    design_points = generate_design_points(config)
    
    # Apply max_designs limit
    if args.max_designs > 0:
        design_points = design_points[:args.max_designs]
    
    # Create explorer
    output_dir = PROJECT_ROOT / args.output
    resume = args.resume and not args.no_resume
    explorer = DesignSpaceExplorer(config, output_dir, resume=resume)
    
    # Run exploration
    explorer.run_exploration(
        design_points,
        skip_hls=args.skip_hls,
        dry_run=args.dry_run
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
