# Simple GNN — GraphSAGE for FPGA Implementation

A complete pipeline for training, quantizing, and deploying a GraphSAGE Graph Neural Network on FPGA via Vitis HLS.

## Overview

This project implements node classification on the [Cora dataset](https://arxiv.org/abs/1603.08861) with a GraphSAGE model, targeting FPGA inference through multiple HLS implementations:

| Model | Test Acc. | Arithmetic | Weights | HLS C-Sim |
|-------|-----------|-----------|---------|-----------|
| Base (64 hidden) | 80.3 % | float32 | float | ✅ reference |
| Reduced (16→24→7) | 75.8 % | float32 | float | ✅ 0 LSB |
| PTQ-Float | 75.7 % | float32 | INT8 | ✅ 0 LSB |
| **PTQ-INT8** | **75.7 %** | **int8/int32** | **INT8** | **✅ 0 LSB** |
| PTQ-INT8-PO2 | — | int8/shifts | INT8 | ✅ 0 LSB |
| **QAT-v2** | **76.8 %** | **int8/int32** | **INT8** | **✅ 0 LSB** |

## Model Variants

| Model | Definition | Training | Architecture | Purpose |
|-------|-----------|----------|-------------|---------|
| Base GraphSAGE | `model_base.py` → `GraphSAGE` | `train.py` | 1433→64→7, root_weight | Full-size baseline |
| Reduced GraphSAGE | `model_base.py` → `ReducedGraphSAGE` | `train.py` | 1433→16→24→7, no root | FPGA-targeted (small) |
| QAT v1 | `model_qat.py` / `model_base_QAT.py` | `train_qat.py` | Same as reduced, fake-quant | Quantization-aware training |
| QAT v2 | `model_qat_v2.py` | `train_qat_v2.py` | Same as reduced, INT32 accum | Corrected QAT matching FPGA |
| Brevitas | `brevitas_models.py` | `brevitas_quantization.py` | Same as reduced, PO2 scales | Brevitas QAT path |

All reduced variants use the same architecture: projection (1433→16) → SAGEConv(16→24) → ReLU → SAGEConv(24→7), with `root_weight=False`.

## HLS Implementations

6 self-contained C++ implementations in `hls/`, each with header, source, testbench, and TCL generator:

| Variant | Files | Data Types | Scale Handling | Top Function |
|---------|-------|-----------|---------------|-------------|
| **Float** | `graphsage_layer_float.{h,cpp}` | float32 everywhere | None | `graphsage_network` |
| **PTQ** | `graphsage_layer_ptq.{h,cpp}` | INT8 data, float ops | Float scale multiply | `graphsage_network_ptq` |
| **INT8** | `graphsage_layer_int8.{h,cpp}` | INT8/INT32, no float | Fixed-point M-bit shift | `graphsage_int8` |
| **INT8-PO2** | `graphsage_layer_int8_po2.{h,cpp}` | INT8/INT32, no float | Power-of-2 bit shifts | `graphsage_int8_po2` |
| **Fixed** | `graphsage_layer_fixed.{h,cpp}` | `ap_fixed<W,I>` | Built into type | `graphsage_network_fixed` |
| **QAT-v2** | `graphsage_layer_qat_v2.{h,cpp}` | INT8/INT32, no float | Fixed-point M-bit shift (QAT scales) | `graphsage_qat_v2` | ✅ 0 LSB |

Shared files: `auto_generated_bitwidths.h` (INT8/INT8-PO2), `dse_config.h` (generated at synthesis by DSE for INT8-PO2/Fixed).

**QAT v2 vs PTQ-INT8**: same kernel math, different scale parameters. QAT v2 has separate
`scale_agg` (aggregation output) and `scale_hidden` (layer output) derived from trained
fake-quantizers, so `beta_fp` generally differs from `K` (unlike PTQ-INT8 where
`scale_in == scale_agg`).

## How Each Implementation Is Generated

The 6 HLS variants differ in **where quantization scales come from**, **what arithmetic the hardware performs**, and **which Python scripts produce their weights**. The model architecture (projection 1433→16 → SAGEConv 16→24 → ReLU → SAGEConv 24→7) is identical for all non-base variants.

### Float

- **Weights**: full-precision float32 from `src/train.py` (no quantization)
- **HLS arithmetic**: float32 multiply-accumulate throughout
- **Scale handling**: none — weights loaded directly as `float` arrays
- **Purpose**: functional reference; highest accuracy, largest DSP footprint

### PTQ-Float

- **Weights**: INT8 (calibrated post-training via `src/quantization_ptq.py`)
  - Calibration runs the reduced model over the training set and records per-tensor min/max → scale
- **HLS arithmetic**: INT8 weights dequantized to float at runtime with `w_float = w_int8 * scale`; all MACs in float32
- **Scale handling**: one float multiplier per layer, no fixed-point tricks
- **Key difference from PTQ-INT8**: the HLS kernel still does float arithmetic — INT8 is storage-only

### PTQ-INT8

- **Weights**: INT8 via `src/quantization_ptq.py` → `src/prepare_ptq_int8_parameters.py`
  - Calibration gives `scale_in`, `scale_w`, `scale_out` per layer
  - Because PTQ calibrates activation ranges jointly, `scale_in ≡ scale_agg`, so `beta_fp = K = 4096` (a fixed constant)
- **HLS arithmetic**: fully integer — INT8 × INT8 → INT32 accumulator, then fixed-point requantization:
  ```
  q_out = clamp( (acc * eff_scale_fp + ROUND) >> M )
  ```
- **Scale handling**: `beta_fp` and `eff_scale_fp` are integers (M=24 fractional bits); no float in the kernel
- **Scripts**: `src/prepare_ptq_int8_parameters.py`, `tests/generate_test_vectors_ptq_int8.py`

### PTQ-INT8-PO2

- **Weights**: same INT8 PTQ calibration as PTQ-INT8
- **HLS arithmetic**: same INT8/INT32 structure, but `eff_scale_fp` is rounded to the nearest power-of-2
  - Multiply `acc * eff_scale_fp` becomes an arithmetic bit-shift (`acc >> shift_bits`)
  - Eliminates DSP multipliers for the requantization step; trades a small accuracy loss for lower hardware cost
- **Scale handling**: power-of-2 shifts replace fixed-point multiplications
- **Scripts**: `src/optimize_bitwidths_int8.py`; `hls/auto_generated_bitwidths.h` holds the shift amounts

### Fixed (ap_fixed)

- **Weights**: no explicit quantization pipeline — Vitis HLS `ap_fixed<W,I>` types handle precision implicitly
- **HLS arithmetic**: arbitrary fixed-point via `ap_fixed<W,I>`; bit-width and integer part are DSE parameters
- **Scale handling**: built into the type system; no separate quantization export step
- **Scripts**: `src/explore_design_space.py` sweeps `(W, I)` combinations; `hls/dse_config.h` generated per run
- **Purpose**: design-space exploration over precision vs. area trade-offs

### QAT-v2

- **Weights**: INT8 from a model trained with fake-quantizers (`src/train_qat_v2.py`)
  - Fake-quantizers are inserted at every quantization boundary (proj output, agg input/output, layer output) and trained jointly with the weights
  - Each boundary gets its own learned scale, so `scale_agg` and `scale_hidden` are independent
- **HLS arithmetic**: identical kernel math to PTQ-INT8 (`acc * eff_scale_fp >> M`), but the four scale parameters differ:
  - `beta_fp = round(scale_in / (K * scale_agg) * 2^M)` — generally **≠ K** (vs. PTQ-INT8 where `beta_fp = K`)
  - `eff_scale_fp = round(scale_agg * scale_w / scale_out * 2^M)`
- **Scale handling**: fixed-point M-bit shift (same as PTQ-INT8), but scale values come from trained observers
- **Scripts**: `src/train_qat_v2.py` → `src/export_qat_v2_parameters.py` → `tests/generate_test_vectors_qat_v2.py`
- **Key difference**: the trained fake-quantizers expose the true per-boundary quantization noise during training, so the model learns weights that tolerate integer arithmetic — giving +1.1 pp accuracy over PTQ at the same hardware cost

### Summary of differences

| | Precision | Scales source | beta_fp | Rescale op | Accuracy vs PTQ |
|---|---|---|---|---|---|
| Float | float32 weights + MACs | none | n/a | float multiply | +4.5 pp |
| PTQ-Float | INT8 weights, float32 MACs | calibration | n/a | float multiply | baseline |
| PTQ-INT8 | INT8 weights + acts, INT32 acc | calibration | = K (4096) | fixed-point shift | 0 pp |
| PTQ-INT8-PO2 | INT8 weights + acts, INT32 acc | calibration + PO2 rounding | power-of-2 | arithmetic bit-shift (no DSP) | −0.x pp |
| Fixed | ap_fixed<W,I> throughout | type system | n/a | ap_fixed MAC | varies with W,I |
| QAT-v2 | INT8 weights + acts, INT32 acc | trained fake-quantizers | ≠ K | fixed-point shift | +1.1 pp |

## Quick Start

```bash
# 1. Create virtual environment & install dependencies
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Run full pipeline (train → quantize → test vectors)
python run_pipeline.py

# 3. HLS C-simulation (requires Vitis HLS 2022.1+)
cd hls && python generate_graphsage_int8_po2_tcl.py
cd ../build/hls/graphsage_int8_po2 && vitis_hls -f project.tcl
```

## Project Structure

```
simple-gnn/
├── src/                              # Python source
│   ├── train.py                      #   Model training (base + reduced)
│   ├── model_base.py                 #   GraphSAGE model definitions
│   ├── subgraph_extraction.py        #   8-node subgraph for HLS
│   ├── quantization_ptq.py           #   Post-training quantization
│   ├── prepare_ptq_int8_parameters.py #  Integer-only PTQ conversion
│   ├── optimize_bitwidths_int8.py    #   Bit-width optimizer
│   ├── explore_design_space.py       #   Design space exploration
│   ├── analyze_pareto.py             #   Pareto front analysis
│   ├── config.py                     #   Configuration management
│   ├── visualization.py              #   Plotting utilities
│   └── ...                           #   QAT, Brevitas, analysis tools
│
├── hls/                              # HLS C++ implementations
│   ├── graphsage_layer_float.{h,cpp} #   Float reference
│   ├── graphsage_layer_ptq.{h,cpp}   #   PTQ float dequant
│   ├── graphsage_layer_int8.{h,cpp}  #   Pure integer
│   ├── graphsage_layer_int8_po2.{h,cpp} # INT8 with PO2 scales
│   ├── testbench_*.cpp               #   Per-variant testbenches
│   ├── generate_graphsage_*_tcl.py   #   TCL project generators
│   └── tcl_example/                  #   Jinja2 TCL templates
│
├── tests/                            # Test vector generators & verification
├── scripts/                          # HLS report parsing, visualization
├── configs/                          # YAML configurations (model, DSE)
├── docs/                             # Architecture, specs, guides
│   ├── guides/QUICKSTART.md
│   ├── specs/QUANTIZATION_SPEC.md
│   └── ...
├── run_pipeline.py                   # One-command full pipeline
├── requirements.txt                  # Python dependencies
└── build/                            # Generated outputs (git-ignored)
```

## Pipeline Steps

| Step | Command | Output |
|------|---------|--------|
| Train models | `python run_pipeline.py` or `python src/train.py` | `build/models/*.pth` |
| Extract subgraph | `python src/subgraph_extraction.py` | `build/subgraph/` |
| PTQ quantization | `python src/quantization_ptq.py` | `build/weights_ptq_float/` |
| INT8 conversion | `python src/prepare_ptq_int8_parameters.py` | `build/weights_ptq_int8/` |
| Test vectors | `python tests/generate_test_vectors_ptq_int8.py` | `build/test_vectors_ptq_int8/` |
| HLS C-sim | `cd hls && python generate_graphsage_int8_po2_tcl.py` | `build/hls/` |
| **QAT v2 train** | `cd src && python train_qat_v2.py` | `build/models/qat_v2_best.pth` |
| **QAT v2 export** | `cd src && python export_qat_v2_parameters.py` | `build/weights_qat_v2/` |
| **QAT v2 vectors** | `cd tests && python generate_test_vectors_qat_v2.py` | `build/test_vectors_qat_v2/` |
| **QAT v2 HLS** | `cd hls && python generate_graphsage_qat_v2_tcl.py` | `build/hls/graphsage_qat_v2/` |
| HLS synthesis | `vitis_hls -f synth.tcl` | `solution1/syn/report/` |
| Report parsing | `python scripts/parse_hls_report.py --all build/hls` | Console / CSV / JSON |
| Design space exploration | `python src/explore_design_space.py --config configs/design_space.yaml` | `build/experiments/` |

## Synthesis Results (Xilinx xcvu13p, 2.77 ns clock)

| Implementation | Test Acc. | Precision | DSP | FF | LUT | Latency | II | Clock |
|----------------|-----------|-----------|-----|-----|-----|---------|-----|-------|
| Float | 78 % | float32 MACs | 5,896 (48%) | 454,788 (13%) | 284,231 (16%) | 42 cyc | 1 | 362 MHz |
| INT8 | 75.7 % | INT8×INT8→INT32 | 6,816 (55%) | 569,609 (16%) | 278,289 (16%) | 56 cyc | — | 357 MHz |
| **INT8-PO2** | **~75 %** | **INT8×INT8→INT32, shift rescale** | **2,560 (21%)** | **232,658 (7%)** | **477,496 (28%)** | **19 cyc** | **1** | **~502 MHz** |
| **QAT-v2** | **76.8 %** | **INT8×INT8→INT32** | **6,760 (55%)** | **561,016 (16%)** | **282,192 (16%)** | **55 cyc** | **1** | **~493 MHz** |

INT8-PO2 note: DSP savings come from replacing fixed-point multipliers with bit-shifts; the LUT increase is because the shift-tree logic lands in LUTs rather than DSPs. Latency drops to 19 cycles because the shorter critical path allows more aggressive pipelining.

## Design Space Exploration

The DSE pipeline (`src/explore_design_space.py`) systematically evaluates combinations of model architecture, quantization parameters, and HLS directives. It automates training, PTQ, test vector generation, and Vitis HLS synthesis for each design point.

```bash
# Preview design points (no execution)
python src/explore_design_space.py --config configs/design_space.yaml --dry-run

# Software-only evaluation (skip HLS synthesis)
python src/explore_design_space.py --config configs/design_space.yaml --skip-hls

# Full run (training + PTQ + HLS synthesis per design point)
python src/explore_design_space.py --config configs/design_space.yaml
```

After exploration, generate Pareto analysis and plots:

```bash
python src/analyze_pareto.py \
    --input build/experiments/design_space_results.json \
    --output build/plots/pareto
```

Configuration files in `configs/`:

| Config | Purpose |
|--------|---------|
| `design_space.yaml` | Default exploration (architecture + unrolling) |
| `design_space_minimal.yaml` | Quick test (few design points) |
| `design_space_full_unroll.yaml` | Maximum unrolling (fastest, largest) |
| `design_space_qformat.yaml` | Fixed-point Q(W,I) format sweep |
| `design_space_pareto.yaml` | Pareto-focused configurations |

See [Design Space Exploration](docs/DESIGN_SPACE_EXPLORATION.md) for the full pipeline architecture, per-architecture isolation, and configuration reference.

## Requirements

- **Python** 3.8+ with PyTorch 2.0+, PyTorch Geometric 2.3+
- **FPGA** (optional): Xilinx Vitis HLS 2022.1+, Vivado Design Suite

## Documentation

| Document | Content |
|----------|---------|
| [Quickstart Guide](docs/guides/QUICKSTART.md) | Step-by-step setup |
| [Visualization Guide](docs/guides/VISUALIZATION_GUIDE.md) | Plot generation and analysis |
| [Quantization Spec](docs/specs/QUANTIZATION_SPEC.md) | INT8 quantization details |
| [HLS Variants](docs/specs/HLS_VARIANTS_AND_TEST_VECTORS.md) | Implementation comparison |
| [Design Space Exploration](docs/DESIGN_SPACE_EXPLORATION.md) | DSE configuration & usage |
| [Models & Implementations](docs/specs/MODELS_AND_IMPLEMENTATIONS.md) | Architecture details |
| [Clean Build Guide](docs/guides/CLEAN_BUILD_GUIDE.md) | Build artifact management |
| [Integer PTQ Pipeline](docs/specs/INTEGER_PTQ_PIPELINE.md) | INT8 conversion pipeline |

## License

<!-- TODO: Choose a license (MIT, Apache-2.0, etc.) and add a LICENSE file -->
No license file yet — add one before public release.
