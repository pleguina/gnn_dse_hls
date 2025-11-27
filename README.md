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
│   └── config.py                     # Configuration management
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
│   └── hls/                          # HLS project outputs
│       ├── graphsage_float/          # Float HLS project
│       ├── graphsage_ptq/            # PTQ HLS project
│       └── graphsage_int8/           # INT8 HLS project
│
├── configs/
│   └── model_config.yaml             # Model hyperparameters
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
