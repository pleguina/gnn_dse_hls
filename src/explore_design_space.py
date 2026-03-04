#!/usr/bin/env python3
"""
Design Space Exploration for PTQ INT8 GraphSAGE HLS

This script orchestrates the full exploration pipeline:
1. Algorithm level: Model architecture variations
2. Quantization level: PTQ INT8 bit-width configurations  
3. HLS level: Implementation optimizations (unroll, pipeline, binding)
4. Verification: C-simulation to compare HLS vs Python golden outputs

Key Features:
- Per-design-point test vector generation (architecture-matched)
- C-simulation with automatic result parsing
- Verification metrics tracking (max_error, mean_error, pass/fail)
- Soft-fail approach: track verification issues but continue synthesis

Outputs:
- build/experiments/design_space_results.csv (includes csim metrics)
- build/experiments/design_space_results.json
- build/experiments/design_points/<id>/config.json (per-point metrics)
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
    
    # Fixed-point Q format (for ap_fixed implementation)
    data_w: int = 16  # Total bits for activations
    data_i: int = 8   # Integer bits for activations (fractional = data_w - data_i)
    weight_w: int = 16  # Total bits for weights
    weight_i: int = 4   # Integer bits for weights
    acc_w: int = 32  # Total bits for accumulators
    acc_i: int = 16  # Integer bits for accumulators
    
    # Computed bit-widths (filled by optimizer)
    ACC_BITS: int = 0
    ADJ_BITS: int = 0
    SCALE_BITS: int = 0
    MULT_BITS: int = 0
    
    # HLS level
    hls_implementation: str = "int8_po2"  # float, int8, int8_po2, fixed, ptq
    unroll_nodes: int = 1
    unroll_features_agg: int = 1
    unroll_features_lin: int = 1
    unroll_features_relu: int = 1
    unroll_outputs: int = 1  # 1 = pipeline (default), >1 = partial unroll, 0 = complete unroll
    agg_pipeline_ii: int = 1
    lin_pipeline_ii: int = 1
    bind_agg_mul: str = "auto"
    bind_lin_scale_mul: str = "auto"
    alloc_agg_mul_limit: str = "unlimited"  # "unlimited" or integer
    alloc_lin_mul_limit: str = "unlimited"  # "unlimited" or integer
    
    # Software metrics (filled after training/PTQ)
    test_accuracy: float = 0.0
    num_parameters: int = 0
    model_memory_bytes: int = 0
    lsb_error_vs_float: int = 0
    
    # C-Simulation verification (filled after csim)
    csim_run: bool = False
    csim_passed: bool = False
    csim_max_error_lsb: int = 0
    csim_mean_error_lsb: float = 0.0
    csim_num_mismatches: int = 0
    csim_total_elements: int = 0
    
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
        'impl': params.get('hls_implementation', 'int8_po2'),
        'icr': params.get('in_channels_reduced', 16),
        'hc': params.get('hidden_channels', 24),
        'mb': params.get('M_BITS', 24),
        'kb': params.get('K_BITS', 12),
        'bm': params.get('bitwidth_margin', 2),
        'un': params.get('unroll_nodes', 1),
        'ufa': params.get('unroll_features_agg', 1),
        'ufl': params.get('unroll_features_lin', 1),
        'uo': params.get('unroll_outputs', 1),
        'bam': params.get('bind_agg_mul', 'dsp'),  # Add bind option to hash
        'aal': params.get('alloc_agg_mul_limit', 'unl'),
        'all': params.get('alloc_lin_mul_limit', 'unl'),
    }
    param_str = json.dumps(key_params, sort_keys=True)
    hash_suffix = hashlib.md5(param_str.encode()).hexdigest()[:8]
    
    # Human-readable prefix with implementation type
    impl_short = key_params['impl'][:3]  # int, fix, flo
    bind_short = 'd' if key_params['bam'] == 'dsp' else 'f'  # d=dsp, f=fabric
    prefix = f"{impl_short}_d{key_params['icr']}x{key_params['hc']}_m{key_params['mb']}_u{key_params['un']}{bind_short}"
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
    
    # Fixed-point Q format parameters (for ap_fixed implementation)
    if 'fixed' in hls:
        fixed_config = hls['fixed']
        data_w_list = fixed_config.get('data_w', [16])
        data_i_list = fixed_config.get('data_i', [8])
        weight_w_list = fixed_config.get('weight_w', [16])
        weight_i_list = fixed_config.get('weight_i', [4])
        acc_w_list = fixed_config.get('acc_w', [32])
        acc_i_list = fixed_config.get('acc_i', [16])
    else:
        # Defaults for fixed-point
        data_w_list = [16]
        data_i_list = [8]
        weight_w_list = [16]
        weight_i_list = [4]
        acc_w_list = [32]
        acc_i_list = [16]
    
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
    implementations_list = hls.get('implementations', ['int8_po2'])  # Default to int8_po2
    
    if 'unroll' in hls:
        unroll_nodes_list = hls['unroll'].get('nodes', [8])
        unroll_feat_agg = hls['unroll'].get('features_agg', [1])
        unroll_feat_lin = hls['unroll'].get('features_lin', [1])
        unroll_feat_relu = hls['unroll'].get('features_relu', [1])
        unroll_outputs = hls['unroll'].get('outputs', [1])
    else:
        unroll_nodes_list = hls.get('unroll_nodes', [8])
        unroll_feat_agg = unroll_factor
        unroll_feat_lin = unroll_factor
        unroll_feat_relu = unroll_factor
        unroll_outputs = [1]
    
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
    
    # Allocation limits for DSP reuse
    if 'allocation' in hls:
        alloc_agg_mul_list = hls['allocation'].get('agg_mul_limit', ['unlimited'])
        alloc_lin_mul_list = hls['allocation'].get('lin_mul_limit', ['unlimited'])
    else:
        alloc_agg_mul_list = ['unlimited']
        alloc_lin_mul_list = ['unlimited']
    
    target_clock = hls.get('target_clock_ns', 2.77)  # 361MHz default
    
    # Generate Q format product (for fixed-point exploration)
    qformat_params = list(itertools.product(
        data_w_list,
        data_i_list,
        weight_w_list,
        weight_i_list,
        acc_w_list,
        acc_i_list,
    ))
    
    hls_params = list(itertools.product(
        implementations_list,
        unroll_nodes_list,
        unroll_feat_agg,
        unroll_feat_lin,
        unroll_feat_relu,
        unroll_outputs,
        agg_ii_list,
        lin_ii_list,
        bind_agg_mul_list,
        bind_lin_mul_list,
        alloc_agg_mul_list,
        alloc_lin_mul_list,
    ))
    
    # Generate all combinations
    design_points = []
    
    for algo_p in algo_params:
        for quant_p in quant_params:
            for hls_p in hls_params:
                for qfmt_p in qformat_params:
                    # Extract algorithm dimensions (these are what actually go into HLS)
                    in_channels = algo_p[0]      # Layer 1 input features
                    hidden_channels = algo_p[1]  # Layer 1 output / Layer 2 input
                    out_features = 7             # Fixed for Cora dataset
                    num_nodes = 8                # Fixed subgraph size
                    
                    # Extract Q format parameters
                    data_w, data_i, weight_w, weight_i, acc_w, acc_i = qfmt_p
                    
                    # Filter invalid Q formats: integer bits must be < total bits
                    if data_i >= data_w or weight_i >= weight_w or acc_i >= acc_w:
                        continue
                    
                    # Filter Q formats with too few fractional bits (< 4)
                    if (data_w - data_i) < 4 or (weight_w - weight_i) < 4 or (acc_w - acc_i) < 4:
                        continue
                    
                    # Extract HLS unroll factors
                    unroll_nodes = hls_p[1]
                    unroll_feat_agg = hls_p[2]
                    unroll_feat_lin = hls_p[3]
                    alloc_agg_limit = hls_p[10]
                    alloc_lin_limit = hls_p[11]
                    
                    # Convert allocation limits to int if not "unlimited"
                    alloc_agg_limit_int = None if alloc_agg_limit == "unlimited" else int(alloc_agg_limit)
                    alloc_lin_limit_int = None if alloc_lin_limit == "unlimited" else int(alloc_lin_limit)
                
                # Calculate actual parallel multiplier demand
                # AGG loops: for i in nodes, for f in features, for j in nodes (fully unrolled)
                #   - Demand = nodes × features (outer loops determine parallelism)
                #   - When unroll=0, it means complete unroll
                # LIN loops: for n in nodes, for o in out_feat, for i in in_feat (fully unrolled)
                #   - Demand = nodes × in_features (or out_features, depending on which loops unroll)
                
                # For AGG: worst case is max(in_channels, hidden_channels) since both layers run
                max_agg_features = max(in_channels, hidden_channels)
                demand_agg = (unroll_nodes if unroll_nodes > 0 else num_nodes) * \
                             (unroll_feat_agg if unroll_feat_agg > 0 else max_agg_features)
                
                # For LIN: worst case is max(in_channels, hidden_channels) for feature dimension
                max_lin_features = max(in_channels, hidden_channels)
                demand_lin = (unroll_nodes if unroll_nodes > 0 else num_nodes) * \
                             (unroll_feat_lin if unroll_feat_lin > 0 else max_lin_features)
                
                # Skip invalid allocation configurations
                # Rule: allocation limit must be <= demand (otherwise it's pointless)
                #       and demand should be divisible by limit (for clean reuse)
                if alloc_agg_limit_int is not None:
                    if alloc_agg_limit_int > demand_agg:
                        continue  # Limit higher than demand - no effect, skip
                    if demand_agg % alloc_agg_limit_int != 0:
                        continue  # Fractional reuse - invalid hardware config
                
                if alloc_lin_limit_int is not None:
                    if alloc_lin_limit_int > demand_lin:
                        continue  # Limit higher than demand - no effect, skip
                    if demand_lin % alloc_lin_limit_int != 0:
                        continue  # Fractional reuse - invalid hardware config
                
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
                    'hls_implementation': hls_p[0],
                    'unroll_nodes': hls_p[1],
                    'unroll_features_agg': hls_p[2],
                    'unroll_features_lin': hls_p[3],
                    'unroll_features_relu': hls_p[4],
                    'unroll_outputs': hls_p[5],
                    'agg_pipeline_ii': hls_p[6],
                    'lin_pipeline_ii': hls_p[7],
                    'bind_agg_mul': hls_p[8],
                    'bind_lin_scale_mul': hls_p[9],
                    'alloc_agg_mul_limit': hls_p[10],
                    'alloc_lin_mul_limit': hls_p[11],
                    'target_clock_ns': target_clock,
                    # Q format parameters (always included, used only for fixed-point)
                    'data_w': data_w,
                    'data_i': data_i,
                    'weight_w': weight_w,
                    'weight_i': weight_i,
                    'acc_w': acc_w,
                    'acc_i': acc_i,
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
            f.write(f"  Implementation: {dp.hls_implementation}\n")
            f.write(f"  M_BITS: {dp.M_BITS}\n")
            f.write(f"  K_BITS: {dp.K_BITS}\n")
            if dp.hls_implementation == "fixed":
                f.write(f"\nFixed-Point Q Format:\n")
                f.write(f"  Data (activations): Q({dp.data_w},{dp.data_i}) = {dp.data_w-dp.data_i} fractional bits\n")
                f.write(f"  Weights: Q({dp.weight_w},{dp.weight_i}) = {dp.weight_w-dp.weight_i} fractional bits\n")
                f.write(f"  Accumulators: Q({dp.acc_w},{dp.acc_i}) = {dp.acc_w-dp.acc_i} fractional bits\n")
            else:
                f.write(f"  ACC_BITS: {dp.ACC_BITS}\n")
                f.write(f"  SCALE_BITS: {dp.SCALE_BITS}\n")
                f.write(f"  MULT_BITS: {dp.MULT_BITS}\n")
            f.write(f"\nHLS Settings:\n")
            f.write(f"  unroll_nodes: {dp.unroll_nodes}\n")
            f.write(f"  unroll_features_agg: {dp.unroll_features_agg}\n")
            f.write(f"  agg_pipeline_ii: {dp.agg_pipeline_ii}\n")
            f.write(f"  lin_pipeline_ii: {dp.lin_pipeline_ii}\n")
            f.write(f"\nVerification (C-Simulation):\n")
            if dp.csim_run:
                f.write(f"  Status: {'PASSED' if dp.csim_passed else 'WARNING'}\n")
                f.write(f"  Max error: {dp.csim_max_error_lsb} LSB\n")
                f.write(f"  Mean error: {dp.csim_mean_error_lsb:.2f} LSB\n")
                f.write(f"  Mismatches: {dp.csim_num_mismatches}/{dp.csim_total_elements}\n")
            else:
                f.write(f"  Status: Not run\n")
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
        
        try:
            # Check for pre-trained model
            pretrained_dir = self._get_config_section('algorithm', 'pretrained_dir')
            if pretrained_dir:
                model_name = f"reduced_graphsage_no_root_{dp.in_channels_reduced}x{dp.hidden_channels}"
                model_path = Path(pretrained_dir) / f"{model_name}_best.pth"
                
                if model_path.exists():
                    print(f"    ✓ Using pre-trained model: {model_path}")
                    # Load accuracy from saved model if available
                    history_path = Path(pretrained_dir).parent / f"{model_name}_history.json"
                    if history_path.exists():
                        with open(history_path, 'r') as f:
                            history = json.load(f)
                            dp.test_accuracy = history.get('best_test_acc', 75.0)
                    else:
                        dp.test_accuracy = 75.0  # Fallback
                    dp.num_parameters = dp.in_channels_reduced * dp.hidden_channels * 2  # Rough estimate
                    print(f"    ✓ Training completed: accuracy={dp.test_accuracy:.1f}%")
                    return True
            
            # Check for default model in build/models
            root_str = "" if dp.root_weight else "_no_root"
            model_name = f"reduced_graphsage{root_str}_{dp.in_channels_reduced}x{dp.hidden_channels}"
            model_path = PROJECT_ROOT / "build/models" / f"{model_name}_best.pth"
            
            if model_path.exists():
                print(f"    ✓ Using existing model: {model_path}")
                history_path = PROJECT_ROOT / "build" / f"{model_name}_history.json"
                if history_path.exists():
                    with open(history_path, 'r') as f:
                        history = json.load(f)
                        # History contains 'test_acc' as a list - get max and convert to %
                        test_acc_list = history.get('test_acc', [])
                        if test_acc_list:
                            dp.test_accuracy = max(test_acc_list) * 100
                        else:
                            dp.test_accuracy = history.get('best_test_acc', 75.0)
                        dp.num_parameters = history.get('num_parameters', dp.in_channels_reduced * dp.hidden_channels * 10)
                else:
                    dp.test_accuracy = 75.0
                    dp.num_parameters = dp.in_channels_reduced * dp.hidden_channels * 10
                print(f"    ✓ Training completed: accuracy={dp.test_accuracy:.1f}%")
                return True
            
            # No pre-trained model found - train now
            print(f"    → No pre-trained model found, training new model...")
            print(f"    → Architecture: {dp.in_channels_reduced} → {dp.hidden_channels} → 7")
            
            # Import training function
            import sys
            sys.path.insert(0, str(PROJECT_ROOT / "src"))
            from train import train_reduced_model
            
            # Run training
            model, data, history = train_reduced_model(
                epochs=200,  # Use default from config
                lr=0.01,
                in_channels_reduced=dp.in_channels_reduced,
                hidden_channels=dp.hidden_channels,
                dropout=dp.dropout,
                root_weight=dp.root_weight,
                use_config=False  # Use provided parameters
            )
            
            # Note: train_reduced_model already saves the model with proper checkpoint format
            # to ../build/models/reduced_graphsage{suffix}_best.pth
            # We also save with architecture-specific name for DSE tracking
            model_dir = PROJECT_ROOT / "build/models"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            import torch
            # Save with proper checkpoint format (matching train.py)
            torch.save({
                'model_state_dict': model.state_dict(),
                'root_weight': dp.root_weight,
            }, model_path)
            
            # Save history with architecture-specific name
            history_path = PROJECT_ROOT / "build" / f"{model_name}_history.json"
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
            
            # Extract metrics
            # History contains 'test_acc' as a list of all epochs - get the best (max)
            test_acc_list = history.get('test_acc', [])
            if test_acc_list:
                dp.test_accuracy = max(test_acc_list) * 100  # Convert to percentage
            else:
                dp.test_accuracy = history.get('best_test_acc', 0.0)
            dp.num_parameters = sum(p.numel() for p in model.parameters())
            dp.model_memory_bytes = dp.num_parameters * 4  # float32
            
            print(f"    ✓ Training completed: accuracy={dp.test_accuracy:.1f}%")
            print(f"    ✓ Model saved: {model_path}")
            return True
            
        except Exception as e:
            print(f"    ✗ Training FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_ptq(self, dp: DesignPoint) -> bool:
        """Run PTQ and INT8 parameter generation"""
        print(f"  [PTQ] M_BITS={dp.M_BITS}, K_BITS={dp.K_BITS}")
        
        try:
            # Check if PTQ already exists for this architecture
            arch_key = f"{dp.in_channels_reduced}x{dp.hidden_channels}"
            # Store PTQ weights per-architecture for clean separation
            ptq_dir = PROJECT_ROOT / "build" / "weights_ptq_per_arch" / arch_key
            
            # Check if PTQ outputs exist
            weights_layer1 = ptq_dir / "weights_layer1_int8.txt"
            if weights_layer1.exists():
                # Verify it's for the right architecture by checking dimensions
                with open(weights_layer1, 'r') as f:
                    lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
                    if len(lines) > 0:
                        # weights_layer1 should be hidden × in_channels
                        expected_count = dp.hidden_channels * dp.in_channels_reduced
                        if len(lines) == expected_count:
                            print(f"    ✓ Using existing PTQ for {arch_key}: {ptq_dir}")
                            print(f"    ✓ PTQ completed successfully")
                            return True
            
            # Need to run PTQ for this architecture
            print(f"    → Running PTQ for architecture {arch_key}...")
            
            # Check if model exists
            root_str = "" if dp.root_weight else "_no_root"
            model_name = f"reduced_graphsage{root_str}_{arch_key}"
            model_path = PROJECT_ROOT / "build/models" / f"{model_name}_best.pth"
            
            if not model_path.exists():
                print(f"    ✗ Model not found: {model_path}")
                return False
            
            # Step 1: Run quantization_ptq.py with architecture parameters
            print(f"    → Step 1: Running PTQ quantization...")
            
            # Create architecture-specific PTQ output directory
            ptq_dir.mkdir(parents=True, exist_ok=True)
            
            cmd = [
                "python", "src/quantization_ptq.py",
                "--in-channels", str(dp.in_channels_reduced),
                "--hidden-channels", str(dp.hidden_channels),
                "--output-dir", str(ptq_dir)
            ]
            # Flag logic: script uses --use-root-weight to enable root_weight.
            # dp.root_weight is typically False for HLS, so the flag is omitted.
            if dp.root_weight:
                cmd.append("--use-root-weight")
            
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                print(f"    ✗ PTQ quantization failed")
                if result.stdout:
                    print(f"      stdout: {result.stdout[-500:]}")
                if result.stderr:
                    print(f"      stderr: {result.stderr[-500:]}")
                return False
            
            # Step 1.5: Generate PTQ test vectors (needed by prepare_ptq_int8_parameters)
            print(f"    → Step 1.5: Generating PTQ test vectors...")
            cmd = [
                "python", "tests/generate_test_vectors_ptq_float.py",
                "--in-channels", str(dp.in_channels_reduced),
                "--hidden-channels", str(dp.hidden_channels),
                "--hw-round"  # Use HW-style rounding for better HLS match
            ]
            
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                print(f"    ✗ PTQ test vector generation failed")
                if result.stdout:
                    print(f"      stdout: {result.stdout[-500:]}")
                if result.stderr:
                    print(f"      stderr: {result.stderr[-500:]}")
                return False
            
            # Step 2: Run prepare_ptq_int8_parameters.py with architecture
            print(f"    → Step 2: Preparing INT8 parameters (M={dp.M_BITS})...")
            cmd = [
                "python", "src/prepare_ptq_int8_parameters.py",
                "--in-channels", str(dp.in_channels_reduced),
                "--hidden-channels", str(dp.hidden_channels),
                "--output-dir", str(ptq_dir),
                "--m-bits", str(dp.M_BITS)
            ]
            
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                print(f"    ✗ INT8 parameter preparation failed")
                if result.stdout:
                    print(f"      stdout: {result.stdout[-500:]}")
                if result.stderr:
                    print(f"      stderr: {result.stderr[-500:]}")
                return False
            
            print(f"    ✓ PTQ completed successfully for {arch_key}")
            return True
            
        except Exception as e:
            print(f"    ✗ PTQ FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_bitwidth_optimizer(self, dp: DesignPoint) -> bool:
        """Run bit-width optimization"""
        print(f"  [BitWidth] margin={dp.bitwidth_margin}, method={dp.bitwidth_method}")
        
        try:
            analysis_path = PROJECT_ROOT / "build/hls/bitwidth_analysis.json"
            
            # Check if we need to run the optimizer
            if not analysis_path.exists():
                print(f"    → Running bitwidth optimizer...")
                
                # Run the optimizer script
                cmd = [
                    sys.executable, str(PROJECT_ROOT / "src/optimize_bitwidths_int8.py"),
                    "--safety-margin", str(dp.bitwidth_margin),
                    "--method", "both",  # Use 'both' for theoretical + data-driven analysis
                    "--m-bits", str(dp.M_BITS),
                    "--hidden-channels", str(dp.hidden_channels),
                    "--in-channels", str(dp.in_channels_reduced)
                ]
                
                result = subprocess.run(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                if result.returncode != 0:
                    print(f"    ⚠ Bitwidth optimizer failed, using defaults")
                    if result.stderr:
                        print(f"      Error: {result.stderr[-300:]}")
                    # Fall through to use defaults
                else:
                    print(f"    ✓ Bitwidth optimizer completed")
            
            # Now read the analysis file if it exists
            if analysis_path.exists():
                with open(analysis_path, 'r') as f:
                    analysis = json.load(f)
                
                # Read optimized values (key is 'final_bitwidths' in the JSON)
                bitwidths = analysis.get('final_bitwidths') or analysis.get('optimized_bitwidths', {})
                dp.ACC_BITS = bitwidths.get('acc_bits', 32)
                dp.SCALE_BITS = bitwidths.get('scale_bits', 32)
                dp.MULT_BITS = bitwidths.get('mult_bits', 64)
                dp.ADJ_BITS = bitwidths.get('adj_bits', 16)
                
                print(f"    ✓ Loaded: ACC={dp.ACC_BITS}, SCALE={dp.SCALE_BITS}, MULT={dp.MULT_BITS}, ADJ={dp.ADJ_BITS}")
                return True
            
            # If file still doesn't exist, use conservative defaults
            print(f"    ⚠ Using conservative defaults (no analysis file)")
            dp.ACC_BITS = 32
            dp.SCALE_BITS = 32
            dp.MULT_BITS = 64
            dp.ADJ_BITS = 16
            print(f"    ✓ BitWidth defaults: ACC={dp.ACC_BITS}, SCALE={dp.SCALE_BITS}")
            return True
            
        except Exception as e:
            print(f"    ✗ BitWidth optimization FAILED: {e}")
            # Use safe defaults on failure
            dp.ACC_BITS = 32
            dp.SCALE_BITS = 32
            dp.MULT_BITS = 64
            dp.ADJ_BITS = 16
            return True
    
    def run_hls_synthesis(self, dp: DesignPoint) -> bool:
        """Generate HLS project and run synthesis"""
        print(f"  [HLS] unroll=({dp.unroll_nodes},{dp.unroll_features_agg},{dp.unroll_features_lin})")
        
        try:
            # Check if we can use existing synthesis results (exact match)
            if (dp.in_channels_reduced == 16 and dp.hidden_channels == 24 and
                dp.M_BITS == 24 and dp.K_BITS == 12 and
                dp.unroll_nodes == 1 and dp.unroll_features_agg == 1 and dp.unroll_features_lin == 1 and
                dp.agg_pipeline_ii == 1 and dp.lin_pipeline_ii == 1):
                
                int8_report = PROJECT_ROOT / "build/hls/graphsage_int8/solution1/syn/report/csynth.xml"
                if int8_report.exists():
                    print(f"    ✓ Using existing synthesis: {int8_report}")
                    result = self._parse_hls_report(dp, int8_report)
                    if result:
                        print(f"    ✓ HLS synthesis completed: DSP={dp.dsp_used}, latency={dp.latency_max_cycles}cy")
                    return result
            
            # Create design point output directory
            dp_dir = self._get_design_point_dir(dp)
            
            # HLS project goes in build/hls/dse_<id> to avoid path depth issues with Vitis HLS
            # (Deep paths cause csim to compile .cpp files differently, leading to duplicate symbols)
            hls_dir = PROJECT_ROOT / "build" / "hls" / f"dse_{dp.design_id}"
            hls_dir.mkdir(parents=True, exist_ok=True)
            
            # Step 1: Generate test vectors for this configuration
            print(f"    Generating test vectors...")
            if not self._generate_test_vectors(dp, dp_dir):
                print(f"    ✗ Test vector generation failed")
                return False
            print(f"    ✓ Test vectors generated")
            
            # Step 2: Generate parameterized HLS header (optional, for reference)
            print(f"    Generating HLS config...")
            if not self._generate_hls_config(dp, hls_dir):
                print(f"    ✗ HLS config generation failed")
                return False
            print(f"    ✓ HLS config generated")
            
            # Step 3: Generate TCL project files using Jinja2 templates
            print(f"    Generating TCL projects from templates...")
            
            # Verify auto_generated_bitwidths.h exists
            bitwidths_h = PROJECT_ROOT / "hls" / "auto_generated_bitwidths.h"
            if not bitwidths_h.exists():
                print(f"    ⚠ WARNING: auto_generated_bitwidths.h not found at {bitwidths_h}")
                print(f"    → Run: python src/optimize_bitwidths_int8.py --safety-margin 2")
                print(f"    → Continuing without optimized bitwidths (will use defaults)")
            else:
                print(f"    ✓ Using optimized bitwidths from {bitwidths_h}")
            
            if not self._generate_tcl_project(dp, hls_dir):
                print(f"    ✗ TCL project generation failed")
                return False
            print(f"    ✓ TCL projects generated")
            
            # Step 4: Run C-simulation (includes project setup)
            print(f"    Running C-simulation...")
            csim_ok = self._run_csim(dp, hls_dir)
            if not csim_ok or not dp.csim_passed:
                if not dp.csim_run:
                    print(f"    ✗ C-simulation failed to run")
                elif not dp.csim_passed:
                    print(f"    ⚠ C-simulation verification failed (continuing with synthesis)")
                # Don't fail on csim - synthesis is the main goal
            else:
                print(f"    ✓ C-simulation passed: max_error={dp.csim_max_error_lsb} LSB")
            
            # Step 5: Run synthesis
            print(f"    Running synthesis (this may take 5-10 minutes)...")
            if not self._run_synthesis(dp, hls_dir):
                print(f"    ✗ Synthesis failed")
                return False
            print(f"    ✓ Synthesis completed")
            
            # Step 6: Parse results - path matches template-generated project structure
            # The project_dir in config is hls_dir (build/hls/dse_<id>)
            report_path = hls_dir / "solution1" / "syn" / "report" / "csynth.xml"
            if report_path.exists():
                result = self._parse_hls_report(dp, report_path)
                if result:
                    print(f"    ✓ HLS synthesis completed: DSP={dp.dsp_used}, latency={dp.latency_max_cycles}cy, Fmax={dp.fmax_mhz:.1f}MHz")
                else:
                    print(f"    ✗ Report parsing failed")
                return result
            else:
                print(f"    ✗ Synthesis report not found at {report_path}")
                return False
        except Exception as e:
            print(f"    ✗ HLS synthesis FAILED: {e}")
            return False
    
    def _generate_test_vectors(self, dp: DesignPoint, dp_dir: Path) -> bool:
        """Generate test vectors for this design point's architecture and implementation"""
        try:
            arch_key = f"{dp.in_channels_reduced}x{dp.hidden_channels}"
            
            # Check if architecture+implementation-specific test vectors already exist
            test_vec_dir = PROJECT_ROOT / "build" / "test_vectors_arch" / dp.hls_implementation / arch_key
            
            # Check for actual test vector files, not just directory existence
            required_files = ["network_input.txt", "weights_layer1.txt", "weights_layer2.txt"]
            has_test_vectors = test_vec_dir.exists() and all((test_vec_dir / f).exists() for f in required_files)
            
            if not has_test_vectors:
                print(f"    ⚠ No test vectors for {dp.hls_implementation}/{arch_key}")
                print(f"    → Generating test vectors...")
                
                # Generate test vectors for this architecture and implementation
                import subprocess
                gen_script = PROJECT_ROOT / "tests" / "generate_test_vectors_per_arch.py"
                
                cmd = [
                    sys.executable, str(gen_script),
                    "--in-channels", str(dp.in_channels_reduced),
                    "--hidden-channels", str(dp.hidden_channels),
                    "--implementation", dp.hls_implementation,
                    "--output-dir", str(PROJECT_ROOT / "build" / "test_vectors_arch")
                ]
                
                # Add Q format parameters if using fixed implementation
                if dp.hls_implementation == "fixed":
                    cmd.extend([
                        "--data-w", str(dp.data_w),
                        "--data-i", str(dp.data_i),
                        "--weight-w", str(dp.weight_w),
                        "--weight-i", str(dp.weight_i),
                        "--acc-w", str(dp.acc_w),
                        "--acc-i", str(dp.acc_i)
                    ])
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode != 0:
                    print(f"    ✗ Test vector generation failed")
                    if result.stdout:
                        print(f"      stdout: {result.stdout[-500:]}")
                    if result.stderr:
                        print(f"      stderr: {result.stderr[-500:]}")
                    return False
                
                print(f"    ✓ Test vectors generated for {dp.hls_implementation}/{arch_key}")
            else:
                print(f"    ✓ Using existing test vectors for {dp.hls_implementation}/{arch_key}")
            
            # Copy test vectors to design point directory
            dst_dir = dp_dir / "test_vectors"
            import shutil
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            shutil.copytree(test_vec_dir, dst_dir)
            
            # ALSO populate legacy directories that testbenches expect
            # The testbenches have hardcoded paths like build/test_vectors_ptq_int8_po2/
            if dp.hls_implementation == "int8_po2":
                legacy_dir = PROJECT_ROOT / "build" / "test_vectors_ptq_int8_po2"
                legacy_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy reference output file with expected name
                src_ref = test_vec_dir / "network_output_int8_po2_reference.txt"
                dst_ref = legacy_dir / "network_output_int8_po2_reference.txt"
                if src_ref.exists():
                    shutil.copy2(src_ref, dst_ref)
                    print(f"    ✓ Copied reference output to legacy location")
                
                # Copy po2_config.json if exists
                src_config = test_vec_dir / "po2_config.json"
                dst_config = legacy_dir / "po2_config.json"
                if src_config.exists():
                    shutil.copy2(src_config, dst_config)
            
            return True
            
        except subprocess.TimeoutExpired:
            print(f"    ✗ Test vector generation timed out")
            return False
        except Exception as e:
            print(f"    ✗ Test vector generation failed: {e}")
            import traceback
            traceback.print_exc()
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
                f.write(f"#define DSE_UNROLL_FEATURES_RELU {dp.unroll_features_relu}\n")
                f.write(f"#define DSE_UNROLL_OUTPUTS {dp.unroll_outputs}\n")
                f.write(f"#define DSE_AGG_PIPELINE_II {dp.agg_pipeline_ii}\n")
                f.write(f"#define DSE_LIN_PIPELINE_II {dp.lin_pipeline_ii}\n\n")
                f.write(f"// Resource binding/allocation parameters\n")
                # bind values must be unquoted identifiers for HLS pragmas
                # HLS expects: #pragma HLS BIND_OP ... impl=dsp (not impl="dsp")
                f.write(f"#define DSE_BIND_AGG_MUL {dp.bind_agg_mul}\n")
                f.write(f"#define DSE_BIND_LIN_MUL {dp.bind_lin_scale_mul}\n")
                # Allocation limits: "unlimited" -> -1, otherwise the numeric value
                alloc_agg = "-1" if dp.alloc_agg_mul_limit == "unlimited" else str(dp.alloc_agg_mul_limit)
                alloc_lin = "-1" if dp.alloc_lin_mul_limit == "unlimited" else str(dp.alloc_lin_mul_limit)
                f.write(f"#define DSE_ALLOC_AGG_MUL_LIMIT {alloc_agg}\n")
                f.write(f"#define DSE_ALLOC_LIN_MUL_LIMIT {alloc_lin}\n\n")
                f.write(f"#endif // DSE_CONFIG_H\n")
            return True
        except Exception as e:
            print(f"    ERROR generating HLS config: {e}")
            return False
    
    def _generate_tcl_project(self, dp: DesignPoint, hls_dir: Path) -> bool:
        """Generate TCL project files using Jinja2 templates"""
        try:
            from jinja2 import Environment, FileSystemLoader
            
            # Paths
            src_hls_dir = PROJECT_ROOT / "hls"
            template_dir = src_hls_dir / "tcl_example"
            
            # Load PO2 shift values from config (for int8_po2 implementation)
            # Check both new location (test_vectors_arch) and legacy location
            po2_shifts = {}
            arch_key = f"{dp.in_channels_reduced}x{dp.hidden_channels}"
            po2_config_paths = [
                PROJECT_ROOT / "build" / "test_vectors_arch" / "int8_po2" / arch_key / "po2_config.json",
                PROJECT_ROOT / "build" / "test_vectors_ptq_int8_po2" / "po2_config.json",  # Legacy
            ]
            for po2_config_path in po2_config_paths:
                if po2_config_path.exists():
                    with open(po2_config_path, 'r') as f:
                        po2_config = json.load(f)
                        po2_shifts = po2_config.get('shifts', {})
                        print(f"    ✓ Loaded PO2 shifts from {po2_config_path.name}: {po2_shifts}")
                    break
            
            # Determine source files based on implementation
            impl = dp.hls_implementation
            if impl == "int8_po2":
                top_func = "graphsage_int8_po2"
                src_files = [
                    str(src_hls_dir / "graphsage_layer_int8_po2.cpp"),
                    str(src_hls_dir / "graphsage_layer_int8_po2.h"),
                ]
                tb_files = [str(src_hls_dir / "testbench_int8_po2.cpp")]
                cflags = [
                    f"-DM_BITS={dp.M_BITS}",
                    f"-DDSE_CONFIG",  # Enable DSE config header
                    f"-DUSE_OPTIMIZED_BITWIDTHS",  # Enable auto_generated_bitwidths.h
                    f"-I{hls_dir}",   # Include path for dse_config.h
                ]
                # Add PO2 shift values from config
                if po2_shifts:
                    cflags.extend([
                        f"-DBETA1_SHIFT={po2_shifts.get('BETA1_SHIFT', 17)}",
                        f"-DBETA2_SHIFT={po2_shifts.get('BETA2_SHIFT', 12)}",
                        f"-DEFF_SCALE1_SHIFT={po2_shifts.get('EFF_SCALE1_SHIFT', 7)}",
                        f"-DEFF_SCALE2_SHIFT={po2_shifts.get('EFF_SCALE2_SHIFT', 8)}",
                    ])
                # Add test vector paths - point ALL to architecture-specific directory
                # Each design point uses its own consistent test vector set
                # No trailing slash - C++ testbench will add separator
                test_vec_arch_dir = PROJECT_ROOT / "build" / "test_vectors_arch" / "int8_po2" / arch_key
                cflags.extend([
                    f'-DINT8_PARAMS_DIR={test_vec_arch_dir}',
                    f'-DTEST_VECTORS_DIR={test_vec_arch_dir}',
                    f'-DPO2_REFERENCE_DIR={test_vec_arch_dir}',
                ])
            elif impl == "int8":
                top_func = "graphsage_int8"
                src_files = [
                    str(src_hls_dir / "graphsage_layer_int8.cpp"),
                    str(src_hls_dir / "graphsage_layer_int8.h"),
                ]
                tb_files = [str(src_hls_dir / "testbench_int8.cpp")]
                cflags = [
                    f"-DM_BITS={dp.M_BITS}",
                    f"-DDSE_CONFIG",
                    f"-DUSE_OPTIMIZED_BITWIDTHS",  # Enable auto_generated_bitwidths.h
                    f"-I{hls_dir}",
                ]
            elif impl == "fixed":
                top_func = "graphsage_fixed"
                src_files = [
                    str(src_hls_dir / "graphsage_layer_fixed.cpp"),
                    str(src_hls_dir / "graphsage_layer_fixed.h"),
                ]
                tb_files = [str(src_hls_dir / "testbench_fixed.cpp")]
                # Fixed-point needs bitwidth configs - use design point Q format
                cflags = [
                    f"-DDATA_W={dp.data_w}",
                    f"-DDATA_I={dp.data_i}",
                    f"-DWEIGHT_W={dp.weight_w}",
                    f"-DWEIGHT_I={dp.weight_i}",
                    f"-DACC_W={dp.acc_w}",
                    f"-DACC_I={dp.acc_i}",
                    f"-DDSE_CONFIG",
                    f"-DUSE_OPTIMIZED_BITWIDTHS",  # Enable auto_generated_bitwidths.h
                    f"-I{hls_dir}",
                ]
            elif impl == "float":
                # Skip float - too expensive for hardware
                print(f"    WARNING: Skipping float implementation (not hardware-friendly)")
                return False
            elif impl == "ptq":
                # Skip PTQ - broken
                print(f"    WARNING: Skipping PTQ implementation (known bugs)")
                return False
            else:
                print(f"    ERROR: Unknown implementation '{impl}'")
                return False
            
            # Configuration for Jinja2 templates
            config = {
                "module_name": f"graphsage_dse_{dp.design_id}",
                "top": top_func,
                "part": "xcvu13p-fsga2577-1-e",
                "clock_period": dp.target_clock_ns,
                "version": "1.0",
                "vendor": "GNN_DSE",
                
                # Source files (implementation-specific)
                "src": src_files,
                "tb": tb_files,
                
                # Absolute paths to include directories
                # Note: auto_generated_bitwidths.h should be in src_hls_dir
                "includes": [
                    str(src_hls_dir),
                ],
                
                # Compiler flags (implementation-specific)
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
        """Run C-simulation and parse verification results"""
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
                print(f"    ✗ Project setup failed")
                dp.csim_run = False
                return False
            
            # Then run csim.tcl
            csim_tcl = hls_dir / "csim.tcl"
            log_path = hls_dir / "csim.log"
            
            cmd = [vitis_hls, "-f", str(csim_tcl)]
            print(f"    Running C-simulation...")
            
            with open(log_path, 'w') as log_file:
                result = subprocess.run(
                    cmd,
                    cwd=str(hls_dir),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=1800  # 30 min timeout for csim
                )
            
            dp.csim_run = True
            
            # Parse csim results even if return code != 0
            # (testbench may return 1 for warnings but still produce valid comparison)
            self._parse_csim_results(dp, log_path)
            
            if result.returncode != 0:
                print(f"    ⚠ C-simulation returned non-zero exit code")
                # Don't fail - we got verification metrics
                return True  # Continue with synthesis
            
            # Check if verification passed
            if dp.csim_passed:
                print(f"    ✓ Verification PASSED: max_error={dp.csim_max_error_lsb} LSB")
            else:
                print(f"    ⚠ Verification WARNING: max_error={dp.csim_max_error_lsb} LSB")
            
            return True
            
            return True
        except subprocess.TimeoutExpired:
            print(f"    ERROR: C-simulation timed out")
            dp.csim_run = False
            return False
        except Exception as e:
            print(f"    ERROR running csim: {e}")
            dp.csim_run = False
            return False
    
    def _parse_csim_results(self, dp: DesignPoint, log_path: Path) -> None:
        """Parse csim.log to extract verification metrics"""
        try:
            if not log_path.exists():
                return
            
            with open(log_path, 'r') as f:
                log_content = f.read()
            
            import re
            
            # Look for verification summary in testbench output
            # Pattern: "Max error: 3 LSB"
            max_error_match = re.search(r'Max error:\s+(\d+)\s+LSB', log_content)
            if max_error_match:
                dp.csim_max_error_lsb = int(max_error_match.group(1))
            
            # Pattern: "Mean absolute error: 0.82 LSB"
            mean_error_match = re.search(r'Mean absolute error:\s+([\d.]+)\s+LSB', log_content)
            if mean_error_match:
                dp.csim_mean_error_lsb = float(mean_error_match.group(1))
            
            # Pattern: "Total differences: 12/56"
            diff_match = re.search(r'Total differences:\s+(\d+)/(\d+)', log_content)
            if diff_match:
                dp.csim_num_mismatches = int(diff_match.group(1))
                dp.csim_total_elements = int(diff_match.group(2))
            
            # Determine pass/fail based on thresholds
            # Testbench output patterns:
            # "*** PERFECT MATCH!" or "*** GOOD:" -> pass
            # "*** WARNING:" -> soft pass (continue synthesis)
            # "*** ERROR:" -> fail
            if '*** PERFECT MATCH' in log_content or '*** GOOD:' in log_content:
                dp.csim_passed = True
            elif '*** WARNING:' in log_content:
                # Soft fail - track but don't block synthesis
                dp.csim_passed = False  # Warning level
            elif '*** ERROR:' in log_content:
                dp.csim_passed = False
            else:
                # If no explicit verdict, check for compilation failure
                if 'CSIM FAILED' in log_content or 'compilation error' in log_content.lower():
                    dp.csim_passed = False
                    print(f"    ✗ C-simulation compilation failed")
                elif dp.csim_max_error_lsb <= 10 and dp.csim_max_error_lsb > 0:
                    # Only pass if we actually got a result (max_error > 0 means test ran)
                    dp.csim_passed = True
                elif 'Max error:' in log_content:
                    # Got result but high error
                    dp.csim_passed = False
                else:
                    # No results found - probably compilation failed
                    dp.csim_passed = False
                    print(f"    ⚠ No verification results found in csim log")
            
        except Exception as e:
            print(f"    ⚠ Could not parse csim results: {e}")
            dp.csim_passed = False  # Fail on parse errors
    
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
        print(f"✓ Completed: {len(self.completed_designs)}")
        print(f"✗ Failed: {len(self.failed_designs)}")
        
        if self.completed_designs:
            print(f"\n✓ Successful designs:")
            for design_id, dp in list(self.completed_designs.items())[:10]:
                print(f"  • {design_id}: acc={dp.test_accuracy:.1f}%, DSP={dp.dsp_used}, lat={dp.latency_max_cycles}cy")
            if len(self.completed_designs) > 10:
                print(f"  ... and {len(self.completed_designs) - 10} more")
        
        if self.failed_designs:
            print(f"\n✗ Failed designs:")
            for design_id, dp in list(self.failed_designs.items())[:10]:
                error_msg = dp.error_message[:60] if dp.error_message else "Unknown error"
                print(f"  • {design_id}: {error_msg}")
            if len(self.failed_designs) > 10:
                print(f"  ... and {len(self.failed_designs) - 10} more")
        
        print(f"\nResults saved:")
        print(f"  • CSV:  {self.results_csv_path}")
        print(f"  • JSON: {self.results_json_path}")
        print(f"  • Artifacts: {self.design_points_dir}")


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
