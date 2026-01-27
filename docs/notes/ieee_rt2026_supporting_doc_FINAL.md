# Design-Space Exploration and Integer Quantization of Graph Neural Networks for Real-Time FPGA Track Finding

**Pelayo Leguina**
*IEEE Real-Time 2026 (25-29 May 2026), La Biodola - Isola d'Elba, Italy*

---

## Motivation and Context

Fixed-latency trigger/DAQ systems in the CMS Level-1 upgrade (12.5 μs budget) demand inference architectures that balance accuracy with hardware feasibility under strict throughput and resource constraints. Displaced-muon signatures from long-lived particles pose unique challenges for traditional pattern-based track finding because they deviate from prompt-track assumptions. Graph neural networks (GNNs) naturally represent sparse, irregular detector geometries: nodes encode stub-level features and edges capture local geometric compatibility, making them well-suited for the overlap muon track finder region where barrel and endcap detectors meet.

## Technical Contribution

This work establishes a complete, reproducible pipeline from PyTorch Geometric model training to validated FPGA-ready HLS implementations. Unlike automated ML toolflows that primarily target dense or convolutional layers, message-passing GNN operations require explicit handling of irregular aggregation patterns, dynamic data movement, and careful numeric design. Our contributions include:

1. **Three HLS implementation variants**: floating-point, PTQ-Float hybrid, and pure integer PTQ-INT8
2. **Automated design-space exploration (DSE)**: across architectural, quantization, and microarchitectural parameters with Pareto-optimal trade-off analysis
3. **Data-driven bit-width optimization**: reducing accumulator and scaling product widths by up to 42 bits while maintaining bit-exact validation

## Model Architecture and Training Results

We implement a GraphSAGE-based architecture with feature projection (1433→16), two message-passing layers (16→24→7), ReLU activations, and dropout. The base model achieves 80.3% accuracy on Cora's 1000-node test set; the reduced FPGA-compatible variant (24,079 parameters, 0.09 MB) achieves 75.8% accuracy. Post-training quantization to INT8 maintains 75.7% accuracy (0.1% degradation), while the same architecture with quantization-aware training yields 28.2% accuracy, indicating insufficient model capacity for learning under quantization constraints during training.

**Table 1: Model Accuracy Comparison on Cora Test Set (1000 nodes)**

| Model Variant | Test Accuracy | Parameters | Memory | Degradation |
|---------------|---------------|------------|--------|-------------|
| Base (float) | 80.3% | 184,391 | 0.70 MB | --- |
| Reduced (float) | 75.8% | 24,079 | 0.09 MB | -4.5% |
| PTQ-INT8 | 75.7% | 24,079 | 0.02 MB | -0.1% |
| QAT-INT8 | 28.2% | 23,527 | 0.02 MB | -47.6% |

*PTQ preserves accuracy effectively; QAT underperforms due to insufficient model capacity.*

## Integer-Only Quantization Scheme

The PTQ-INT8 implementation eliminates all floating-point operations from the datapath:
- **Weights and activations**: Symmetric INT8 quantization
- **Biases**: Pre-scaled to INT32 accumulator domain
- **Adjacency matrices**: Scaled by K=2^12 and stored as INT16
- **Fixed-point scaling**: Uses M fractional bits with round-to-nearest

The quantization formula:

```
y = clamp(⌊(a · s_fp + 2^(M-1)) / 2^M⌋, -128, 127)
```

where `a` is the INT32 accumulator and `s_fp` is the fixed-point scale factor with M fractional bits.

**Validation Results:**
- **M=24**: HLS C-simulation achieves **0 LSB error** versus Python integer emulator on 8-node subgraphs
- **M=20**: ≤13 LSB maximum error with 21-bit narrower multipliers (significant resource savings)

**Table 2: HLS C-Simulation Validation Results**

| Implementation | Arithmetic | Validation Target | Max Error |
|----------------|-----------|-------------------|-----------|
| Float | FP32 | Python float reference | 0 (reference) |
| PTQ-Float | FP32 + INT8 params | Python PTQ-Float | 0 LSB |
| PTQ-INT8 (M=24) | INT8/INT32/INT16 | Python int emulator | **0 LSB** |
| PTQ-INT8 (M=20) | INT8/INT32/INT16 | Python int emulator | ≤13 LSB |

*M=24 achieves bit-exact match; M=20 trades precision for resource savings.*

## HLS Synthesis Results and Optimization

Synthesis targets the Xilinx VU13P (xcvu9p-flga2577):

**Floating-point baseline:**
- 5,896 DSP slices (48% utilization)
- 454K flip-flops (13%)
- 284K LUTs (16%)
- 42-cycle latency at 362 MHz (116 ns per inference)

**PTQ-INT8 variant (full loop unrolling):**
- 6,816 DSPs (55%)
- 570K FFs (16%)
- 278K LUTs (16%)
- 56-cycle latency at 357 MHz (157 ns per inference)

**Counter-intuitive observation**: INT8 uses *more* DSPs due to wider 64-bit scaling multiplications (INT32 × scale_fp) combined with aggressive unrolling.

**Table 3: Xilinx VU13P Synthesis Results**

| Implementation | DSP (%) | FF (%) | LUT (%) | Latency (cycles) | Clock (MHz) | Latency (ns) |
|----------------|---------|--------|---------|------------------|-------------|--------------|
| Float | 5,896 (48%) | 454K (13%) | 284K (16%) | 42 | 362 | 116 |
| PTQ-INT8 | 6,816 (55%) | 570K (16%) | 278K (16%) | 56 | 357 | 157 |

*INT8 uses more DSPs due to 64-bit scaling multiplications and full unrolling.*

### Data-Driven Bit-Width Optimization

Bit-width optimization analyzes actual numerical ranges during integer simulation and computes minimal safe widths with configurable margins. For the 8-node subgraph with M=24:

| Type | Conservative | Optimized | Savings |
|------|--------------|-----------|---------|
| Adjacency | 16 bits | 16 bits | 0 bits |
| Accumulators | 32 bits | 22 bits | **10 bits** |
| Scaling factors | 32 bits | 21 bits | **11 bits** |
| Scaling products | 64 bits | 43 bits | **21 bits** |

**Total**: 42 fewer bits in critical multiplier paths, directly reducing DSP and LUT consumption.

## Design Space Exploration

The DSE framework systematically explores three hierarchical levels:

1. **Algorithm level**: Hidden dimensions (16, 24, 32), fixed-point precision M (20, 24, 28)
2. **Quantization level**: Unroll factors (1, 8), accumulator bit-widths
3. **HLS level**: Pipeline initiation intervals, resource binding

Each design point executes the full flow:
- PTQ parameter generation
- Test vector creation
- HLS synthesis
- Report parsing

**Pareto analysis** identifies non-dominated solutions across accuracy, latency, and DSP usage, enabling rapid navigation of multi-objective trade-offs. Current results show a single Pareto point (16×24 architecture, M=24, unroll=1) at:
- **Accuracy**: 75.8%
- **Latency**: 56 cycles (157 ns)
- **DSP usage**: 6,816 (55%)

Ongoing parameter sweeps will populate the full Pareto frontier.

## Path to Physics Application

While this work uses the **Cora citation network** (2,708 nodes, 7 classes) as a publicly accessible proxy to develop and stress-test the toolchain, the target application is **displaced-muon track finding in the CMS overlap region**.

### Planned Transfer Steps:

1. **Graph topology definition**:
   - Nodes = detector stubs from DT/CSC/RPC chambers
   - Edges = local Δη-Δφ geometric compatibility

2. **Retraining**: On simulated displaced-muon events

3. **DSE re-execution**: To meet the 12.5 μs latency budget and resource allocation on Phase-2 FPGAs (e.g., Xilinx VU13P with ~12K DSPs)

The established pipeline ensures that once physics data is integrated, hardware feasibility can be rapidly assessed.

## Reproducibility and Code Availability

The complete workflow is orchestrated by:
- **Main pipeline**: `run_pipeline.py`
- **Design space exploration**: `src/explore_design_space.py`
- **HLS kernels**: `hls/` directory
- **Configuration files**:
  - `configs/model_config.yaml`
  - `configs/design_space.yaml`
- **Verification**: `tests/` directory

All scripts, test vector generators, and synthesis report parsers are version-controlled. The repository includes automated verification scripts that validate bit-exact agreement between Python and HLS at each pipeline stage, ensuring reproducibility across development cycles.

### Key Pipeline Stages:

1. **Training**: GraphSAGE model on Cora dataset
2. **Model reduction**: FPGA-compatible architecture (16→24→7)
3. **Subgraph extraction**: Fixed 8-node subgraph for validation
4. **Quantization**: PTQ-INT8 with fixed-point scaling
5. **HLS implementation**: Three variants (Float, PTQ-Float, PTQ-INT8)
6. **Validation**: Bit-exact C-simulation (0 LSB error)
7. **Synthesis**: Resource/latency characterization
8. **DSE**: Automated parameter sweeps with Pareto analysis

---

## Summary of Key Results

- ✅ **End-to-end pipeline**: Training → Quantization → HLS → Validation
- ✅ **Bit-exact validation**: 0 LSB error (M=24) between Python and HLS
- ✅ **Minimal accuracy loss**: 75.7% vs 75.8% (0.1% degradation from PTQ)
- ✅ **Hardware-validated**: Synthesis on Xilinx VU13P, 157 ns latency
- ✅ **Automated DSE**: Pareto analysis across accuracy/latency/resources
- ✅ **Bit-width optimization**: 42-bit reduction in multiplier widths
- ✅ **Reproducible**: Full automation with verification at each stage

**Next Steps**: Integration with displaced-muon physics data and final track-finder graph topology.
