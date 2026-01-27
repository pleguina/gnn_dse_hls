# Simple GNN - GraphSAGE for FPGA Implementation

A complete pipeline for training, optimizing, and implementing a GraphSAGE Graph Neural Network model from CPU to FPGA prototype with multiple HLS implementations.

## Overview

This project implements a GraphSAGE model for node classification on the Cora dataset, with a complete pipeline from CPU training to FPGA-ready HLS code. The workflow includes:

1. **Base Model Training** - Train GraphSAGE on full Cora dataset (80.3% accuracy)
2. **Model Reduction** - Create FPGA-friendly reduced model (75.8% accuracy)
3. **Subgraph Extraction** - Extract fixed 8-node subgraph for hardware validation
4. **Post-Training Quantization (PTQ)** - INT8 quantization (75.7% accuracy)
5. **Multiple HLS Implementations** - Float, PTQ-Float, PTQ-INT8
6. **Validation** - Bit-exact testbench verification (0 LSB error achieved)

## Quick Start

```bash
# 1. Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Run full pipeline
python run_pipeline.py

# 3. Run HLS C-simulation (requires Vitis HLS)
cd hls
python generate_graphsage_tcl.py      # Generate Float TCL
vitis_hls -f project.tcl              # Run Float csim

python generate_graphsage_int8_tcl.py # Generate INT8 TCL
cd ../build/hls/graphsage_int8
vitis_hls -f project.tcl              # Run INT8 csim
```

## Project Structure

```
simple-gnn/
├── src/                              # Python source code
│   ├── model_base.py                 # GraphSAGE model definitions
│   ├── train.py                      # Training script (base + reduced models)
│   ├── subgraph_extraction.py        # 8-node subgraph extraction
│   ├── quantization_ptq.py           # Post-Training Quantization
│   ├── prepare_ptq_int8_parameters.py # Integer-only PTQ conversion
│   ├── analyze_models.py             # Model analysis and visualization
│   ├── visualization.py              # Plotting utilities
│   ├── config.py                     # Configuration management
│   ├── optimize_bitwidths_int8.py    # Bit-width optimizer for INT8
│   ├── explore_design_space.py       # Design Space Exploration (DSE)
│   └── analyze_pareto.py             # Pareto front analysis
│
├── hls/                              # HLS C++ implementations
│   ├── graphsage_layer_float.h/cpp   # Float implementation
│   ├── graphsage_layer_ptq.h/cpp     # PTQ with float quant/dequant
│   ├── graphsage_layer_int8.h/cpp    # Pure INT8 (integer-only)
│   ├── graphsage_layer_fixed.h/cpp   # Fixed-point (experimental)
│   ├── testbench_float.cpp           # Float testbench
│   ├── testbench_ptq.cpp             # PTQ testbench
│   ├── testbench_int8.cpp            # INT8 testbench
│   ├── generate_graphsage_tcl.py     # Float TCL generator
│   ├── generate_graphsage_ptq_tcl.py # PTQ TCL generator
│   ├── generate_graphsage_int8_tcl.py # INT8 TCL generator
│   └── tcl_example/                  # Jinja2 TCL templates
│
├── scripts/                          # Utility scripts
│   ├── parse_hls_report.py           # Parse HLS synthesis reports
│   ├── visualize_hls_report.py       # Visualize HLS metrics
│   ├── verify_setup.py               # Environment verification
│   └── clean_build.py                # Clean build artifacts
│
├── tests/                            # Test vector generators
│   ├── generate_test_vectors_float.py      # Float test vectors
│   ├── generate_test_vectors_ptq_float.py  # PTQ-Float test vectors
│   ├── generate_test_vectors_ptq_int8.py   # PTQ-INT8 test vectors
│   └── compare_ptq_float_vs_int8_detailed.py # LSB error analysis
│
├── build/                            # Build outputs (generated)
│   ├── models/                       # Trained model checkpoints
│   │   ├── base_graphsage_best.pth
│   │   ├── reduced_graphsage_best.pth
│   │   ├── reduced_graphsage_no_root_best.pth  # HLS-compatible
│   │   └── reduced_graphsage_qat_no_root_best.pth
│   ├── weights_float/                # Float weights
│   ├── weights_ptq_float/            # PTQ weights (INT8 + float scales)
│   ├── weights_ptq_int8/             # Integer-only PTQ parameters
│   ├── test_vectors_float/           # Float HLS test vectors
│   ├── test_vectors_ptq_float/       # PTQ-Float test vectors
│   ├── test_vectors_ptq_int8/        # PTQ-INT8 test vectors
│   ├── plots/                        # Visualization outputs
│   │   └── model_stats.json          # Model comparison statistics
│   ├── experiments/                  # DSE results (generated)
│   │   ├── design_space_results.json # Full results database
│   │   └── design_points/            # Individual design artifacts
│   └── hls/                          # HLS project outputs
│       ├── graphsage_float/          # Float HLS project
│       ├── graphsage_ptq/            # PTQ HLS project
│       └── graphsage_int8/           # INT8 HLS project
│
├── configs/
│   ├── model_config.yaml             # Model hyperparameters
│   └── design_space.yaml             # DSE search space configuration
├── docs/                             # Additional documentation
├── run_pipeline.py                   # Main pipeline script
└── requirements.txt                  # Python dependencies
```

## Model Variants and Accuracy

All models evaluated on **Cora test set (1000 nodes)**:

| Model | Test Accuracy | Parameters | Memory | Description |
|-------|---------------|------------|--------|-------------|
| **Base** | 80.3% | 184,391 | 0.70 MB | Full GraphSAGE (64 hidden) |
| **Reduced** | 75.8% | 24,079 | 0.09 MB | FPGA-friendly (16→24→7) |
| **PTQ-Float** | 75.7% | 24,079 | 0.02 MB | INT8 weights, float ops |
| **PTQ-INT8** | 75.7% | 24,079 | 0.02 MB | Pure integer arithmetic |

**Note**: PTQ-Float and PTQ-INT8 have identical classification accuracy because the numerical precision difference (≤13 LSB) doesn't affect argmax predictions.

## HLS Implementations

### Implementation Comparison

| Implementation | Arithmetic | Weights | Activations | Scaling | Test Vectors |
|----------------|-----------|---------|-------------|---------|--------------|
| **Float** | float32 | float | float | N/A | `test_vectors_float/` |
| **PTQ-Float** | float32 | INT8 | INT8 | float dequant | `test_vectors_ptq_float/` |
| **PTQ-INT8** | int8/int32 | INT8 | INT8 | fixed-point (M=24) | `test_vectors_ptq_int8/` |

### C-Simulation Results

| Implementation | Status | Max Error | Notes |
|----------------|--------|-----------|-------|
| Float | ✅ PASS | Reference | Baseline implementation |
| PTQ-Float | ✅ PASS | 0 LSB | Matches Python exactly |
| PTQ-INT8 | ✅ PASS | **0 LSB** | Perfect match with M=24 + HW rounding |

### Synthesis Results (Xilinx xcvu9p)

| Implementation | DSP | FF | LUT | Latency | Clock |
|----------------|-----|-----|-----|---------|-------|
| **Float** | 5,896 (48%) | 454,788 (13%) | 284,231 (16%) | 42 cycles | 362 MHz |
| **PTQ-INT8** | 6,816 (55%) | 569,609 (16%) | 278,289 (16%) | 56 cycles | 357 MHz |

**Note**: INT8 uses more DSPs due to full loop unrolling and 64-bit scaling multiplications.

## Pipeline Steps

### Step 1: Train Models

```bash
cd src
python train.py
```

**Outputs:**
- `build/models/base_graphsage_best.pth` - Base model
- `build/models/reduced_graphsage_best.pth` - Reduced (with root_weight)
- `build/models/reduced_graphsage_no_root_best.pth` - HLS-compatible
- `build/plots/base_model_training.png`
- `build/plots/reduced_model_training.png`

### Step 2: Extract Subgraph

```bash
cd src
python subgraph_extraction.py
```

**Outputs:**
- `build/subgraph/` - 8-node subgraph for HLS validation

### Step 3: Quantize Model (PTQ)

```bash
cd src
python quantization_ptq.py                    # PTQ-Float weights
python prepare_ptq_int8_parameters.py         # Integer-only parameters
```

**Outputs:**
- `build/weights_ptq_float/` - INT8 weights + float scales
- `build/weights_ptq_int8/` - INT32 biases + fixed-point scales (M=24)

### Step 4: Generate Test Vectors

```bash
cd tests
python generate_test_vectors_float.py         # Float vectors
python generate_test_vectors_ptq_float.py     # PTQ-Float vectors
python generate_test_vectors_ptq_int8.py      # PTQ-INT8 vectors
```

**Outputs:**
- `build/test_vectors_float/`
- `build/test_vectors_ptq_float/`
- `build/test_vectors_ptq_int8/`

### Step 5: Analyze Models

```bash
cd src
python analyze_models.py
```

**Outputs:**
- `build/plots/model_comparison.png`
- `build/plots/accuracy_degradation.png`
- `build/plots/efficiency_analysis.png`
- `build/plots/model_stats.json`

### Step 6: Run HLS C-Simulation

#### Float Implementation
```bash
cd hls
python generate_graphsage_tcl.py
vitis_hls -f project.tcl
```

#### PTQ-Float Implementation
```bash
cd hls
python generate_graphsage_ptq_tcl.py
cd ../build/hls/graphsage_ptq
vitis_hls -f project.tcl
```

#### PTQ-INT8 Implementation
```bash
cd hls
python generate_graphsage_int8_tcl.py
cd ../build/hls/graphsage_int8
vitis_hls -f project.tcl
```

### Step 7: Run HLS Synthesis (Optional)

```bash
cd build/hls/graphsage_int8
vitis_hls -f synth.tcl
# Results in solution1/syn/report/csynth.rpt
```

### Step 8: Analyze HLS Reports

Parse and compare synthesis reports from all implementations:

```bash
# Compare all implementations
python scripts/parse_hls_report.py --all build/hls

# Parse single project
python scripts/parse_hls_report.py build/hls/graphsage_int8

# Export to JSON
python scripts/parse_hls_report.py --json --all build/hls -o build/hls_comparison.json

# Export to CSV
python scripts/parse_hls_report.py --csv build/hls_comparison.csv --all build/hls
```

**Extracted metrics:**
- Timing: Target/estimated clock, frequency, slack
- Latency: Cycles, real-time, initiation interval (II)
- Resources: DSP, FF, LUT, BRAM, URAM (absolute and %)
- Timing violations detection

### Step 9: Visualize HLS Reports

Generate visual comparisons of HLS implementations:

```bash
# Generate full dashboard
python scripts/visualize_hls_report.py --all build/hls -o build/plots/hls_comparison.png

# From JSON file
python scripts/visualize_hls_report.py --json build/hls_comparison.json -o build/plots/hls_comparison.png

# Individual charts
python scripts/visualize_hls_report.py --all build/hls --chart resources -o build/plots/hls_resources.png
python scripts/visualize_hls_report.py --all build/hls --chart latency -o build/plots/hls_latency.png
python scripts/visualize_hls_report.py --all build/hls --chart efficiency -o build/plots/hls_efficiency.png
```

**Available charts:**
- `dashboard` (default): Full comparison with all metrics
- `resources`: Resource utilization bar chart (DSP, FF, LUT %)
- `latency`: Latency comparison with timing status
- `absolute`: Absolute resource counts
- `efficiency`: Latency vs DSP scatter plot
- `timing`: Clock frequency analysis

## Model Architecture

### Base Model (Full Cora)
```
GraphSAGE(
  SAGEConv(1433 → 64)    # Full input features
  ReLU + Dropout(0.5)
  SAGEConv(64 → 7)       # 7 classes
)
```

### Reduced Model (HLS-Compatible)
```
ReducedGraphSAGE(
  Linear(1433 → 16)      # Feature projection
  ReLU
  SAGEConv(16 → 24)      # Layer 1 (root_weight=False)
  ReLU + Dropout(0.5)
  SAGEConv(24 → 7)       # Layer 2 (7 classes)
)
```

### HLS Parameters
- **Nodes**: 8 (subgraph for validation)
- **Input Features**: 16 (after projection)
- **Hidden Features**: 24
- **Output Features**: 7 (classes)
- **Adjacency**: Row-normalized, scaled by K=4096 for INT8

## Quantization Details

### PTQ-Float
- Weights: INT8 symmetric quantization
- Activations: INT8 per-tensor quantization
- Operations: Dequantize → float multiply → quantize

### PTQ-INT8 (Integer-Only)
- Weights: INT8
- Biases: INT32 (pre-scaled)
- Activations: INT8
- Scaling: Fixed-point with M=24 fractional bits
- Adjacency: INT16 (scaled by K=4096)
- Accumulators: INT32/INT64

**Fixed-Point Formula:**
```
output = clamp((acc * scale_fp + (1 << (M-1))) >> M, -128, 127)
```

### Bit-Width Optimization (INT8)

The PTQ-INT8 implementation supports automatic bit-width optimization to reduce resource usage while maintaining numerical correctness.

**Problem**: Default conservative bit-widths waste FPGA resources:
- ACC_BITS = 32 (accumulator)
- SCALE_BITS = 32 (fixed-point scales)
- MULT_BITS = 64 (scaling products)

**Solution**: Analyze actual numerical ranges and compute minimal safe bit-widths.

#### Running the Optimizer

```bash
cd src
python optimize_bitwidths_int8.py              # Default: data-driven + 2-bit margin
python optimize_bitwidths_int8.py --method theoretical  # Worst-case bounds
python optimize_bitwidths_int8.py --safety-margin 4     # More conservative
```

**Outputs:**
- `build/hls/auto_generated_bitwidths.h` - HLS header with optimized types
- `hls/auto_generated_bitwidths.h` - Copy for direct inclusion
- `build/hls/bitwidth_analysis.json` - Detailed analysis report

#### Optimization Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| **Theoretical** | Worst-case bounds from model dimensions | Safety-critical designs |
| **Data-driven** | Actual maxima from integer simulation | Resource-optimized designs |
| **Both** (default) | Data-driven with theoretical validation | Recommended |

#### Example Results

For the 8-node subgraph with M=24:

| Type | Default | Optimized | Savings |
|------|---------|-----------|---------|
| ADJ_BITS | 16 | 16 | 0 bits |
| ACC_BITS | 32 | 22 | **10 bits** |
| SCALE_BITS | 32 | 21 | **11 bits** |
| MULT_BITS | 64 | 43 | **21 bits** |

**Total**: ~42 bits narrower types → significant DSP/LUT savings.

#### Using Optimized Bit-Widths in HLS

**Option 1: Include header with flag**
```cpp
#define USE_OPTIMIZED_BITWIDTHS
#include "graphsage_layer_int8.h"
```

**Option 2: Pass compiler flags**
```bash
vitis_hls -D USE_OPTIMIZED_BITWIDTHS -f project.tcl
```

**Option 3: Manual override**
```bash
vitis_hls -D ACC_BITS=22 -D SCALE_BITS=21 -D MULT_BITS=43 -f project.tcl
```

#### Theoretical Formulas

The optimizer uses these formulas for worst-case bounds:

```
# Aggregation accumulator
|tmp| ≤ deg_max × A_int_max × feat_max
AGG_BITS = ceil(log2(|tmp|)) + 1 + margin

# Linear accumulator
|acc| ≤ max(|bias|, F_in × feat_max × w_max)
LIN_BITS = ceil(log2(|acc|)) + 1 + margin

# Combined accumulator
ACC_BITS = max(AGG_BITS, LIN_BITS)

# Scaling product
MULT_BITS = ACC_BITS + SCALE_BITS
```

See `docs/notes/bitsize_optimization_int8ptq.txt` for full methodology.

### Design Space Exploration (DSE)

The project includes a comprehensive pipeline for automated design space exploration across model architecture, quantization parameters, and HLS optimization settings.

**Components:**
- `configs/design_space.yaml` - DSE configuration file
- `src/explore_design_space.py` - Main exploration orchestrator
- `src/analyze_pareto.py` - Pareto front analysis and visualization

#### Design Space Configuration

The search space is organized in three hierarchical levels:

```yaml
# configs/design_space.yaml
search_space:
  algorithm:            # Level 1: Model architecture
    hidden_dim: [16, 24, 32]
    m_bits: [20, 24, 28]      # Fixed-point precision
  quantization:         # Level 2: Quantization parameters
    unroll_factor: [1, 8]     # Loop unrolling
    acc_bits: [24, 32]
  hls:                  # Level 3: HLS pragmas
    agg_pipeline_ii: [1, 2]
    lin_pipeline_ii: [1, 2]
```

#### Running Design Space Exploration

```bash
# Dry run - preview design points without execution
python src/explore_design_space.py --config configs/design_space.yaml --dry-run

# Full exploration (requires Vitis HLS)
python src/explore_design_space.py --config configs/design_space.yaml

# Skip HLS synthesis (PTQ accuracy only)
python src/explore_design_space.py --config configs/design_space.yaml --skip-hls

# Resume interrupted exploration
python src/explore_design_space.py --config configs/design_space.yaml --resume

# Parallel execution (4 workers)
python src/explore_design_space.py --config configs/design_space.yaml --parallel 4
```

**Outputs:**
- `build/experiments/design_space_results.json` - Full results database
- `build/experiments/design_space_results.csv` - CSV export
- `build/experiments/design_points/` - Individual design point artifacts

#### Pareto Analysis

After exploration, analyze trade-offs with Pareto front visualization:

```bash
# Generate all Pareto plots
python src/analyze_pareto.py --input build/experiments/design_space_results.json

# Specific objective pairs
python src/analyze_pareto.py -i build/experiments/design_space_results.json \
    --plot accuracy_drop dsp_util

# Interactive display
python src/analyze_pareto.py -i build/experiments/design_space_results.json --show
```

**Pareto Plots Generated:**
| Plot | Objectives | Description |
|------|------------|-------------|
| `pareto_accuracy_vs_dsp.png` | Accuracy ↑, DSP ↓ | Accuracy/resource trade-off |
| `pareto_accuracy_vs_latency.png` | Accuracy ↑, Latency ↓ | Accuracy/speed trade-off |
| `pareto_latency_vs_dsp.png` | Latency ↓, DSP ↓ | Speed/resource trade-off |
| `pareto_3d.png` | All three | 3D Pareto surface |

#### DSE Pipeline Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                   DESIGN SPACE CONFIGURATION                    │
│  design_space.yaml → Algorithm × Quantization × HLS settings   │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│              FOR EACH DESIGN POINT (d16x24_m24_...)            │
│                                                                 │
│  1. Generate INT8 parameters with config                       │
│     └─ prepare_ptq_int8_parameters.py                          │
│  2. Generate test vectors                                       │
│     └─ generate_test_vectors_ptq_int8.py                       │
│  3. Generate HLS project with pragmas                           │
│     └─ generate_graphsage_int8_tcl.py                          │
│  4. Run HLS synthesis                                           │
│     └─ vitis_hls -f project.tcl                                │
│  5. Parse synthesis report                                      │
│     └─ parse_hls_report.py                                     │
│  6. Record results                                              │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                    PARETO ANALYSIS                              │
│  analyze_pareto.py → Pareto-optimal designs + visualizations   │
└────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING (Full Cora Dataset)                  │
│  train.py → Base Model (80.3%) → Reduced Model (75.8%)          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    QUANTIZATION (PTQ)                            │
│  quantization_ptq.py → INT8 weights + float scales (75.7%)      │
│  prepare_ptq_int8_parameters.py → Integer-only params           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TEST VECTORS (8-node Subgraph)                │
│  Float: test_vectors_float/                                      │
│  PTQ-Float: test_vectors_ptq_float/                              │
│  PTQ-INT8: test_vectors_ptq_int8/                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    HLS VALIDATION                                │
│  Float csim: ✅ PASS (reference)                                 │
│  PTQ-Float csim: ✅ PASS (0 LSB error)                           │
│  PTQ-INT8 csim: ✅ PASS (0 LSB error with M=24)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    HLS SYNTHESIS                                 │
│  Float: 5,896 DSP, 42 cycles                                     │
│  INT8: 6,816 DSP, 56 cycles                                      │
└─────────────────────────────────────────────────────────────────┘
```

## Requirements

### Python
- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- NumPy, Matplotlib, Jinja2, PyYAML

### FPGA Development
- Xilinx Vitis HLS 2022.1+ (for HLS synthesis)
- Vivado Design Suite (for FPGA implementation)

## Configuration

Model hyperparameters are defined in `configs/model_config.yaml`:

```yaml
reduced_model:
  in_channels_reduced: 16
  hidden_channels: 24
  out_channels: 7
  dropout: 0.5
  root_weight: false  # HLS-compatible
```

## Visualization

After running `analyze_models.py`, plots are saved to `build/plots/`:

| Plot | Description |
|------|-------------|
| `model_comparison.png` | Accuracy/Parameters/Memory comparison |
| `accuracy_degradation.png` | Accuracy drop from optimizations |
| `efficiency_analysis.png` | Accuracy vs Size trade-off |
| `quantization_error.png` | Weight quantization error distribution |
| `model_stats.json` | Raw statistics for all models |

## Troubleshooting

### Issue: Test vectors mismatch
**Solution**: Ensure you've run `train.py` first to generate trained models.

### Issue: HLS csim fails
**Solution**: Check that test vectors match the HLS implementation (float vs ptq vs int8).

### Issue: High DSP usage in INT8
**Cause**: Full loop unrolling + 64-bit scaling multiplications.
**Solution**: Use `#pragma HLS PIPELINE` instead of `#pragma HLS UNROLL` for area optimization.

### Issue: Different accuracy numbers
**Note**: Python accuracy (75.7%) is on 1000 test nodes. HLS validation uses 8-node subgraph.

## References

- **GraphSAGE**: [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/
- **Vitis HLS**: https://www.xilinx.com/products/design-tools/vitis/vitis-hls.html

## License

MIT License - Free for research and educational purposes.
