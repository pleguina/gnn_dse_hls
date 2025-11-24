# Quick Start Guide

Get started with Simple-GNN in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- ~2GB free disk space
- Internet connection (for downloading Cora dataset)

## Installation

```bash
# 1. Clone or navigate to the project directory
cd simple-gnn

# 2. Activate the virtual environment (already created)
source venv/bin/activate

# 3. Verify setup
python verify_setup.py
```

## Quick Run

### Option 1: Run the Complete Pipeline

Run everything automatically:

```bash
python run_pipeline.py
```

This will:
- Train base and reduced GraphSAGE models (~5-10 minutes)
- Extract a fixed subgraph for FPGA
- Apply pruning (optional)
- Quantize to INT8
- Generate test vectors for HLS
- **Generate comprehensive visualization plots**

### Option 2: Run Steps Individually

#### Step 1: Train Models

```bash
cd src
python train.py
```

Expected output:
- `models/base_graphsage_best.pth` - Base model (~70-80% test accuracy)
- `models/reduced_graphsage_best.pth` - Reduced model (~65-75% test accuracy)

#### Step 2: Extract Subgraph

```bash
cd src
python subgraph_extraction.py
```

Output: `outputs/subgraph/` directory with adjacency matrix and features

#### Step 3: Quantize Model

```bash
cd src
python quantization.py
```

Output:
- `outputs/quantized/` - Quantized weights in text format
- `hls/weights.h` - C header file for FPGA

#### Step 4: Generate Test Vectors

```bash
cd tests
python generate_test_vectors.py
```

Output: `tests/test_vectors/` with HLS testbench data

#### Step 5: Analyze and Visualize

```bash
cd src
python analyze_models.py
```

Output: Multiple visualization plots in `outputs/plots/`:
- Training curves (loss and accuracy)
- Model comparison (accuracy, parameters, memory)
- Efficiency analysis (accuracy vs size)
- Quantization error analysis
- Comprehensive summary report

## Testing HLS Implementation

### C++ Testbench (No Vivado Required)

```bash
cd hls
make
make test
```

This compiles and runs a C++ testbench that validates the HLS implementation.

### Vivado HLS (If Installed)

```bash
cd hls
make hls-csim   # C simulation
make hls-synth  # Synthesis
make hls-cosim  # C/RTL cosimulation
```

## Expected Results

### Model Accuracy
- **Base Model**: ~75-80% test accuracy on Cora
- **Reduced Model**: ~65-75% test accuracy
- **Quantized Model**: Within 2-3% of floating-point model

### HLS Validation
- Testbench should pass with <1% error rate
- Max quantization error: ±2-3 INT8 levels

## Troubleshooting

### Import Errors
```bash
# Make sure you activated the virtual environment
source venv/bin/activate

# Verify installations
python verify_setup.py
```

### CUDA/GPU Issues
Models default to CPU. To use GPU, edit training scripts and add `.to('cuda')`.

### Dataset Download Fails
The Cora dataset downloads automatically. If it fails, check your internet connection and try again.

### HLS Compilation Errors
The C++ testbench doesn't require Vivado HLS. Use `make test` for basic validation.
For actual HLS synthesis, you need Xilinx Vivado HLS 2019.1 or newer.

## What's Next?

1. **Review Results**: Check `models/` for trained weights
2. **Customize Model**: Edit `src/model_base.py` to change architecture
3. **Tune Hyperparameters**: Modify `src/train.py` for different settings
4. **FPGA Implementation**: Use generated HLS code for FPGA deployment

## File Locations

| Output | Location |
|--------|----------|
| Trained models | `models/*.pth` |
| Subgraph data | `outputs/subgraph/` |
| Quantized weights | `outputs/quantized/` |
| **Visualization plots** | **`outputs/plots/*.png`** |
| Model statistics | `outputs/plots/model_stats.json` |
| HLS weights header | `hls/weights.h` |
| Test vectors | `tests/test_vectors/` |

## Command Reference

```bash
# Full pipeline
python run_pipeline.py

# Individual steps
python run_pipeline.py --steps train
python run_pipeline.py --steps subgraph
python run_pipeline.py --steps quant
python run_pipeline.py --steps vectors

# Skip training (use existing models)
python run_pipeline.py --skip-training

# HLS testing
cd hls && make test
```

## Performance

On a typical CPU:
- Training base model: ~3-5 minutes
- Training reduced model: ~2-3 minutes
- Quantization: <1 minute
- Test vector generation: <1 minute
- **Total pipeline: ~5-10 minutes**

## Getting Help

- See `README.md` for detailed documentation
- Check `build_instructions.txt` for the original plan
- Run `python verify_setup.py` to diagnose setup issues

## Architecture Overview

```
Input (32 nodes × 16 features)
    ↓
[Projection Layer] (1433 → 16)
    ↓
[GraphSAGE Layer 1] (16 → 24)
    ↓ ReLU
[GraphSAGE Layer 2] (24 → 7)
    ↓
Output (32 nodes × 7 classes)
```

All operations use INT8 quantization for FPGA efficiency.
