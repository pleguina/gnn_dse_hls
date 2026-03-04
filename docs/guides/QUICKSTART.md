# Quick Start Guide

## Prerequisites

- Python 3.8+
- ~500 MB free disk space (Cora dataset + models)
- Internet connection (first run downloads the Cora dataset automatically)

## Installation

```bash
cd simple-gnn
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/verify_setup.py
```

## Run the Full Pipeline

```bash
python run_pipeline.py
```

This runs the following steps in sequence:

| Step | Script | Output |
|------|--------|--------|
| Train base + reduced models | `src/train.py` | `build/models/*.pth` |
| Train QAT model | `src/train_qat.py` | `build/models/4_qat_no_root.pth` |
| Extract 8-node subgraph | `src/subgraph_extraction.py` | `build/subgraph/` |
| PTQ quantization | `src/quantization_ptq.py` | `build/weights_ptq_float/` |
| QAT weight export | `src/quantization_qat.py` | `build/weights_qat/` |
| INT8 parameter conversion | `src/prepare_ptq_int8_parameters.py` | `build/weights_ptq_int8/` |
| Brevitas quantization | `src/brevitas_quantization.py` | `build/brevitas/` |
| Float test vectors | `tests/generate_test_vectors_float.py` | `build/test_vectors_float/` |
| PTQ test vectors | `tests/generate_test_vectors_ptq_float.py` | `build/test_vectors_ptq_float/` |
| INT8 test vectors | `tests/generate_test_vectors_ptq_int8.py` | `build/test_vectors_ptq_int8/` |
| Model analysis & plots | `src/analyze_models.py` | `build/plots/` |

Total run time: ~5–10 minutes on CPU.

## Run Individual Steps

```bash
# Train only
python run_pipeline.py --steps train

# Skip training, run quantization and vectors
python run_pipeline.py --skip-training --steps quant,int8_ptq,vectors

# Skip training and analysis
python run_pipeline.py --skip-training --skip-analysis
```

Available step names: `train`, `train_qat`, `subgraph`, `quant`, `quant_qat`, `int8_ptq`, `brevitas`, `vectors`, `analyze`, `all`.

## Run Steps Directly

```bash
source venv/bin/activate

# Train models
cd src && python train.py && cd ..

# Extract subgraph
cd src && python subgraph_extraction.py && cd ..

# PTQ quantization
cd src && python quantization_ptq.py && cd ..

# Generate INT8 test vectors
cd tests && python generate_test_vectors_ptq_int8.py && cd ..

# Analyze all model variants
cd src && python analyze_models.py && cd ..
```

## HLS C-Simulation

Requires Xilinx Vitis HLS 2022.1+.

```bash
# Generate HLS project for INT8-PO2 variant
cd hls && python generate_graphsage_int8_po2_tcl.py
cd ../build/hls/graphsage_int8_po2
vitis_hls -f project.tcl
```

For the float variant:
```bash
cd hls
vitis_hls -f run_csim_float.tcl
```

## Design Space Exploration

```bash
# Full exploration (training + PTQ + HLS synthesis)
python src/explore_design_space.py --config configs/design_space.yaml

# Software-only (no HLS)
python src/explore_design_space.py --config configs/design_space.yaml --skip-hls

# Preview design points without running
python src/explore_design_space.py --config configs/design_space.yaml --dry-run
```

After exploration, generate Pareto analysis:
```bash
python src/analyze_pareto.py --input build/experiments/design_space_results.json --output build/plots/pareto
```

## Expected Results

| Model | Test Accuracy (Cora) |
|-------|---------------------|
| Base (64 hidden) | ~80% |
| Reduced (16→24→7) | ~76% |
| PTQ-Float | ~76% |
| PTQ-INT8 | ~76% |

## Output Locations

| Artifact | Path |
|----------|------|
| Trained models | `build/models/` |
| Training plots | `build/plots/` |
| PTQ weights | `build/weights_ptq_float/` |
| INT8 weights | `build/weights_ptq_int8/` |
| QAT weights | `build/weights_qat/` |
| Float test vectors | `build/test_vectors_float/` |
| PTQ test vectors | `build/test_vectors_ptq_float/` |
| INT8 test vectors | `build/test_vectors_ptq_int8/` |
| Subgraph data | `build/subgraph/` |
| HLS synthesis | `build/hls/` |
| DSE results | `build/experiments/` |

## Troubleshooting

**Import errors**: Activate the virtual environment (`source venv/bin/activate`) and run `python scripts/verify_setup.py`.

**Dataset download fails**: The Cora dataset is fetched by PyTorch Geometric on first run. Check your internet connection and retry.

**HLS compilation**: The C++ testbenches in `hls/` require Vitis HLS. Without it, use `make` with a standard C++ compiler for basic validation.

## Architecture

```
Input (8 nodes × 16 features)
    ↓
[GraphSAGE Layer 1] (16 → 24, mean aggregation)
    ↓ ReLU
[GraphSAGE Layer 2] (24 → 7, mean aggregation)
    ↓
Output (8 nodes × 7 classes)
```

The full model includes a projection layer (1433 → 16) that maps raw Cora features down to the reduced dimension. The 8-node subgraph is extracted for HLS validation.
