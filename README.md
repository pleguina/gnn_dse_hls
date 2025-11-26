# Simple GNN - GraphSAGE for FPGA Implementation

A complete pipeline for training, optimizing, and implementing a GraphSAGE Graph Neural Network model from CPU to FPGA prototype.

## Overview

This project implements a GraphSAGE model for node classification on the Cora dataset, with a complete pipeline from CPU training to FPGA-ready HLS code. The workflow includes:

1. **Base Model Training** - Train GraphSAGE on Cora dataset
2. **Model Reduction** - Create FPGA-friendly reduced model
3. **Subgraph Extraction** - Extract fixed subgraph for hardware
4. **Structured Pruning** - Apply channel pruning to reduce model size
5. **Quantization** - INT8 quantization for FPGA deployment
6. **HLS Implementation** - C++ implementation with Vivado HLS pragmas
7. **Validation** - Testbench for bit-accurate verification

## Project Structure

```
simple-gnn/
├── src/                          # Python source code
│   ├── model_base.py            # Base GraphSAGE model (no root connection)
│   ├── model_base_QAT.py        # QAT-compatible GraphSAGE model
│   ├── model_qat.py             # QAT model with fake quantization
│   ├── train.py                 # Training script for base & reduced models
│   ├── train_qat.py             # QAT training script
│   ├── subgraph_extraction.py   # Subgraph extraction utilities
│   ├── quantization_ptq.py      # PTQ (Post-Training Quantization) utilities
│   ├── quantization_qat.py      # QAT (Quantization-Aware Training) utilities
│   ├── prepare_ptq_int8_parameters.py  # Convert PTQ to integer-only format
│   ├── export_biases.py         # Export float biases for INT32 conversion
│   ├── analyze_models.py        # Model analysis and comparison
│   ├── visualization.py         # Visualization utilities
│   ├── config.py                # Configuration management
│   └── deprecated/              # Old/unused source files
├── hls/                         # HLS C++ implementation
│   ├── graphsage_layer_float.h  # Float HLS header
│   ├── graphsage_layer_float.cpp # Float HLS implementation
│   ├── graphsage_layer.h        # PTQ float quant/dequant HLS header
│   ├── graphsage_layer.cpp      # PTQ float quant/dequant HLS implementation
│   ├── testbench_float.cpp      # Float HLS testbench
│   ├── testbench.cpp            # PTQ HLS testbench
│   ├── Makefile                 # Build system for PTQ HLS
│   ├── Makefile.float           # Build system for float HLS
│   └── *.tcl                    # Vivado HLS TCL scripts
├── tests/                       # Test scripts and utilities
│   ├── generate_test_vectors_float.py       # Float model test vectors
│   ├── generate_test_vectors_ptq_float.py   # PTQ float quant/dequant vectors
│   ├── generate_test_vectors_ptq_int8.py    # PTQ integer-only vectors
│   ├── generate_all_test_vectors.py         # Unified generator wrapper
│   ├── compare_hls_vs_python.py             # Compare HLS vs Python outputs
│   ├── compare_python_vs_hls_testbench.py   # Python testbench for HLS validation
│   ├── evaluate_ptq_int8_accuracy.py        # Evaluate integer PTQ accuracy
│   ├── debug_ptq_node6.py                   # Debug PTQ for specific node
│   ├── verify_float_reference.py            # Verify float model outputs
│   ├── old_generators/          # Old test vector generators
│   └── deprecated/              # Old/unused test scripts
├── build/                       # Build outputs
│   ├── models/                  # Trained model checkpoints
│   │   ├── 1_base_full.pth     # Base model with root connection
│   │   ├── 2_reduced_with_root.pth  # Reduced model with root
│   │   ├── 3_reduced_no_root.pth    # Reduced model (HLS-compatible)
│   │   └── 4_qat_no_root.pth        # QAT model (HLS-compatible)
│   ├── training_history/        # Training history JSON files
│   ├── weights_float/           # Float model weights
│   ├── weights_ptq_float/       # PTQ weights (INT8 + float scales)
│   ├── weights_ptq_int8/        # PTQ integer-only parameters
│   ├── weights_qat/             # QAT weights
│   ├── test_vectors_float/      # Float HLS test vectors
│   ├── test_vectors_ptq_float/  # PTQ float quant/dequant test vectors
│   ├── test_vectors_ptq_int8/   # PTQ integer-only test vectors
│   ├── test_vectors_qat/        # QAT test vectors
│   ├── hls/                     # HLS build outputs
│   ├── plots/                   # Training plots and visualizations
│   └── subgraph/                # Extracted subgraph data
├── docs/                        # Documentation
│   ├── guides/                  # User guides
│   │   ├── CLEAN_BUILD_GUIDE.md
│   │   ├── QUICKSTART.md
│   │   └── VISUALIZATION_GUIDE.md
│   ├── specs/                   # Technical specifications
│   │   ├── QUANTIZATION_SPEC.md
│   │   ├── MODELS_AND_IMPLEMENTATIONS.md
│   │   ├── HLS_VARIANTS_AND_TEST_VECTORS.md
│   │   └── INTEGER_PTQ_PIPELINE.md
│   ├── notes/                   # Development notes
│   │   ├── CURRENT_STATUS.md
│   │   ├── DIRECTORY_STRUCTURE.md
│   │   ├── PTQ_ACCURACY_SUMMARY.md
│   │   ├── PTQ_ROUNDING_ANALYSIS.md
│   │   ├── QAT_TRAINING_NOTES.md
│   │   ├── SAGECONV_ADJACENCY_FIX.md
│   │   └── *.txt (various notes)
│   └── deprecated/              # Old/obsolete documentation
├── data/                        # Cora dataset (auto-downloaded)
├── configs/                     # Configuration files
│   └── model_config.yaml
├── scripts/                     # Utility scripts
│   ├── clean_build.py          # Clean build artifacts
│   ├── verify_setup.py          # Setup verification
│   ├── verify_both_versions.py  # Model version comparison
│   └── deprecated/              # Old one-time scripts
├── requirements.txt             # Python dependencies
├── run_pipeline.py              # Main pipeline orchestration script
└── README.md                    # This file
```

## Requirements

### Python Environment
- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- NumPy, matplotlib, scikit-learn

### FPGA Development (Optional)
- Xilinx Vivado HLS (for HLS synthesis)
- Vivado Design Suite (for FPGA implementation)

## Installation

### 1. Set up Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
cd src
python -c "import torch; import torch_geometric; print('Installation successful!')"
```

## Usage

### Step 1: Train Base Model

Train the base GraphSAGE model on the full Cora dataset:

```bash
cd src
python train.py
```

This will:
- Download Cora dataset automatically
- Train base GraphSAGE model (64 hidden channels)
- Train reduced GraphSAGE model (16 input, 24 hidden channels)
- Save models to `models/` directory

Expected output:
- `models/base_graphsage_best.pth` - Base model checkpoint
- `models/reduced_graphsage_best.pth` - Reduced model checkpoint

### Step 2: Extract Subgraph

Extract a fixed subgraph for FPGA implementation:

```bash
cd src
python subgraph_extraction.py
```

This creates:
- Fixed subgraph with 32 nodes
- Adjacency matrix in dense format
- Node features and labels
- Saved to `outputs/subgraph/`

### Step 3: Apply Pruning (Optional)

Apply structured pruning to further reduce model size:

```bash
cd src
python pruning.py
```

This will:
- Compute channel importance scores
- Prune less important channels
- Fine-tune the pruned model
- Save to `models/pruned_graphsage.pth`

### Step 4: Quantize Model

Quantize the model to INT8 for FPGA:

```bash
cd src
python quantization.py
```

Output:
- Quantized weights in `outputs/quantized/`
- C header file in `hls/weights.h`
- Quantization parameters in JSON format

### Step 5: Generate Test Vectors

Generate test vectors for HLS validation:

```bash
cd tests
python generate_test_vectors.py
```

This creates test vectors in `tests/test_vectors/`:
- `adj_matrix.txt` - Adjacency matrix
- `network_input.txt` - Quantized input features
- `weights_layer1.txt`, `bias_layer1.txt` - Layer 1 parameters
- `weights_layer2.txt`, `bias_layer2.txt` - Layer 2 parameters
- `network_output_reference.txt` - Expected output
- `scales.txt` - Quantization scale factors

### Step 6: Analyze Models and Generate Plots

Generate comprehensive visualization and analysis plots:

```bash
cd src
python analyze_models.py
```

This creates multiple visualization plots in `outputs/plots/`:
- **Training Curves**: Loss and accuracy over epochs for each model
- **Model Comparison**: Side-by-side accuracy, parameters, and memory comparison
- **Efficiency Analysis**: Accuracy vs model size and memory scatter plots
- **Accuracy Degradation**: Impact of optimizations on accuracy
- **Resource Utilization**: Parameter breakdown by layer type
- **Quantization Error**: Distribution analysis of quantization errors
- **Summary Report**: Comprehensive multi-panel summary figure

All statistics are also saved to `model_stats.json` for further analysis.

### Step 7: HLS Synthesis and Validation

#### C Simulation

```bash
cd hls
# Compile testbench
g++ -std=c++11 testbench.cpp graphsage_layer.cpp -o testbench

# Run testbench
./testbench
```

#### Vivado HLS Synthesis (if you have Vivado HLS installed)

```tcl
# Create HLS project
open_project graphsage_hls
set_top graphsage_network
add_files graphsage_layer.cpp
add_files graphsage_layer.h
add_files -tb testbench.cpp

# Run C simulation
csim_design

# Run synthesis
csynth_design

# Run C/RTL cosimulation
cosim_design
```

## Visualization and Analysis

The project includes comprehensive visualization tools that generate plots automatically during training and analysis:

### Training Visualizations
- **Loss Curves**: Track training loss convergence over epochs
- **Accuracy Curves**: Monitor train/val/test accuracy throughout training
- **Per-Model Plots**: Separate visualizations for base and reduced models

### Model Comparison Plots
1. **Model Comparison Bar Charts**
   - Accuracy comparison across all model variants
   - Parameter count comparison (in thousands)
   - Memory footprint comparison (MB)

2. **Efficiency Analysis**
   - Accuracy vs Parameters scatter plot
   - Accuracy vs Memory scatter plot
   - Identify the sweet spot between accuracy and efficiency

3. **Accuracy Degradation Analysis**
   - Shows accuracy impact of each optimization step
   - Color-coded bars (green = good, orange = acceptable, red = significant degradation)
   - Percentage degradation displayed on each bar

4. **Resource Utilization**
   - Stacked bar charts showing parameter breakdown by layer type
   - Conv layers, Linear layers, and Other components
   - Helps identify optimization opportunities

5. **Quantization Error Analysis**
   - Original vs quantized weight distributions
   - Quantization error histogram
   - Statistical summary (mean, std, max error, RMSE)

6. **Summary Report**
   - Comprehensive multi-panel figure with all key metrics
   - Includes summary table and compression ratios
   - Perfect for presentations and reports

### Generated Plot Files

After running the pipeline, you'll find these plots in `outputs/plots/`:
```
outputs/plots/
├── base_model_training.png        # Base model training curves
├── reduced_model_training.png     # Reduced model training curves
├── model_comparison.png           # Side-by-side comparison
├── efficiency_analysis.png        # Accuracy vs size/memory
├── accuracy_degradation.png       # Impact of optimizations
├── resource_utilization.png       # Parameter breakdown
├── quantization_error.png         # Quantization analysis
├── summary_report.png             # Comprehensive summary
└── model_stats.json               # Raw statistics (JSON)
```

### Using the Visualization Tools

**Automatic**: Plots are generated automatically when you run:
```bash
python run_pipeline.py
```

**Manual**: Generate plots separately:
```bash
# After training, analyze and plot
cd src
python analyze_models.py
```

**Custom Plotting**: Use the visualization module in your own scripts:
```python
from visualization import plot_training_curves, plot_model_comparison

# Plot your training history
history = {'train_loss': [...], 'train_acc': [...], ...}
plot_training_curves(history, save_path='my_plot.png')
```

## Model Architecture

### Base Model
```
GraphSAGE(
  SAGEConv(1433, 64)
  ReLU
  Dropout(0.5)
  SAGEConv(64, 7)
)
```

### Reduced Model (FPGA-friendly)
```
ReducedGraphSAGE(
  Linear(1433, 16)        # Feature projection
  ReLU
  SAGEConv(16, 24)        # Layer 1
  ReLU
  Dropout(0.5)
  SAGEConv(24, 7)         # Layer 2
)
```

### HLS Implementation
- **Input**: 32 nodes × 16 features (INT8)
- **Hidden**: 32 nodes × 24 features (INT8)
- **Output**: 32 nodes × 16 features (INT8)
- **Adjacency**: 32 × 32 (FLOAT, normalized)
- **Accumulators**: INT32

## Hardware Specifications

### Resource Estimates
- **Target Device**: Xilinx Zynq-7000 or similar
- **Data Type**: INT8 for weights/activations, INT32 for accumulators
- **Memory**:
  - Adjacency matrix: ~4KB (32×32 float)
  - Weights Layer 1: ~400 bytes (16×24 int8)
  - Weights Layer 2: ~400 bytes (24×16 int8)
  - Feature buffers: ~2KB

### Performance Targets
- **Latency**: ~1000 clock cycles for full inference (depends on clock frequency and parallelism)
- **Throughput**: Configurable via HLS pragmas
- **Accuracy**: Within 2-3 quantization levels of floating-point reference

## Dataset

**Cora Citation Network**
- 2,708 scientific publications (nodes)
- 5,429 citation links (edges)
- 1,433 binary features per node
- 7 classes (publication categories)
- Task: Node classification

## Customization

### Change Model Dimensions

Edit `src/train.py`:
```python
model = ReducedGraphSAGE(
    in_channels=1433,
    in_channels_reduced=16,    # Change this
    hidden_channels=24,         # Change this
    out_channels=7,
    dropout=0.5
)
```

Update corresponding constants in `hls/graphsage_layer.h`:
```cpp
#define MAX_FEATURES_IN 16      // Match in_channels_reduced
#define MAX_FEATURES_HIDDEN 24  // Match hidden_channels
```

### Change Subgraph Size

Edit `src/subgraph_extraction.py`:
```python
subgraph_data = extract_fixed_subgraph(data, num_nodes=32)  # Change this
```

Update `hls/graphsage_layer.h`:
```cpp
#define MAX_NODES 32  // Match num_nodes
```

## Validation

The testbench compares HLS output with PyTorch reference:
- **Tolerance**: ±2 quantization levels (INT8)
- **Metrics**: Element-wise comparison, error rate, max error

## Next Steps

1. **Add More Layers**: Extend to 3+ layer networks
2. **Optimize Parallelism**: Tune HLS pragmas for better performance
3. **Real Hardware**: Deploy to actual FPGA board
4. **Advanced Quantization**: Experiment with INT4 or mixed precision
5. **Different Datasets**: Try larger graphs or different domains

## References

- **GraphSAGE Paper**: [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/
- **Vivado HLS**: https://www.xilinx.com/products/design-tools/vivado/integration/hls.html

## License

MIT License - Feel free to use for research and educational purposes.

## Troubleshooting

### Issue: CUDA out of memory
**Solution**: Models default to CPU. If you want GPU, edit `src/train.py` to add `.to('cuda')`.

### Issue: Dataset download fails
**Solution**: Check internet connection or download Cora manually from PyG.

### Issue: HLS compilation errors
**Solution**: Ensure you're using Vivado HLS 2019.1 or newer. The `ap_int.h` header requires Xilinx tools.

### Issue: Test vectors mismatch
**Solution**: Ensure you've run training first to generate `models/reduced_graphsage_best.pth`.

## Contact

For questions or issues, please refer to the build instructions (build_instructions.txt) or open an issue in the repository.
