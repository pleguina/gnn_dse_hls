# Design Space Exploration Guide

**Last Updated:** January 29, 2026

## Overview

The design space exploration systematically evaluates combinations of:
1. **Algorithm** - Model architecture choices
2. **Quantization** - Bit-width and precision configurations
3. **HLS Implementation** - Hardware optimization directives

This creates a 3-stage filtering pipeline that narrows from ~1,776 theoretical points to a small Pareto-optimal set.

**New in this version:**
- 📊 **Dense vs Sparse Analysis** (Section 3.2) - Comprehensive comparison of message passing implementations, including resource analysis showing linear layers dominate DSP usage (85%), making aggregation optimization less impactful than quantization optimization.

---

## DSE Pipeline Architecture

The `explore_design_space.py` orchestrator runs these steps **automatically** for each design point:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DSE Pipeline Flow                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. TRAINING (if not cached)                                                 │
│     └─► run_pipeline.py --mode train                                         │
│         └─► Saves: build/models/reduced_graphsage_{in}x{hidden}.pt          │
│                                                                              │
│  2. PTQ QUANTIZATION (per-architecture)                                      │
│     └─► quantization_ptq.py --in-channels {in} --hidden-channels {hidden}   │
│         --output-dir build/weights_ptq_per_arch/{in}x{hidden}               │
│         └─► Saves: build/weights_ptq_per_arch/{in}x{hidden}/quant_params.json│
│                                                                              │
│  3. INT8 PARAMETER PREPARATION (per-architecture)                            │
│     └─► prepare_ptq_int8_parameters.py --m-bits {M_BITS}                    │
│         --output-dir build/weights_ptq_per_arch/{in}x{hidden}               │
│         └─► Saves: build/weights_ptq_per_arch/{in}x{hidden}/int8_params.json│
│                                                                              │
│  4. TEST VECTOR GENERATION (per-architecture + implementation)               │
│     └─► generate_test_vectors_per_arch.py                                   │
│         --in-channels {in} --hidden-channels {hidden} --implementation {impl}│
│         └─► Saves: build/test_vectors_arch/{impl}/{in}x{hidden}/            │
│                    - network_input.txt, adj_matrix_int16.txt                │
│                    - weights_layer1.txt, weights_layer2.txt                 │
│                    - bias_layer1_int32.txt, bias_layer2_int32.txt           │
│                    - network_output_{impl}_reference.txt (golden output)    │
│                                                                              │
│  5. BITWIDTH OPTIMIZATION (automatic if bitwidth_analysis.json missing)     │
│     └─► optimize_bitwidths_int8.py --m-bits {M_BITS}                        │
│         └─► Saves: build/hls/bitwidth_analysis.json                         │
│         └─► Saves: hls/auto_generated_bitwidths.h                           │
│                                                                              │
│  6. HLS SYNTHESIS (per-design-point)                                         │
│     └─► Creates: build/hls/dse_{design_id}/                                 │
│         └─► Generates: dse_config.h with architecture-specific parameters   │
│         └─► Vitis HLS with -DDSE_CONFIG -DUSE_OPTIMIZED_BITWIDTHS          │
│         └─► Testbench loads from: build/test_vectors_arch/{impl}/{arch}/    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Per-Architecture Isolation

**Critical design principle:** Each architecture (e.g., 16x16, 16x24, 16x32) has completely isolated storage to prevent data corruption and enable parallel processing.

**Directory structure:**
```
build/
├── weights_ptq_per_arch/          # PTQ weights per architecture
│   ├── 16x16/
│   │   ├── quant_params.json      # Quantization scales
│   │   ├── int8_params.json       # INT8 conversion parameters
│   │   ├── conv1_lin_l_weight.txt # Layer 1 weights (16×16)
│   │   └── conv2_lin_l_weight.txt # Layer 2 weights (7×16)
│   ├── 16x24/                     # Separate 16×24 weights
│   └── 16x32/                     # Separate 16×32 weights
│
├── test_vectors_arch/             # Test vectors per implementation + architecture
│   ├── int8_po2/
│   │   ├── 16x16/                 # INT8 PO2 test vectors for 16×16
│   │   │   ├── network_input.txt  # Input features (8 nodes × 16 features)
│   │   │   ├── weights_layer1.txt # Layer 1 weights (16×16)
│   │   │   ├── weights_layer2.txt # Layer 2 weights (7×16)
│   │   │   └── network_output_int8_po2_reference.txt  # Golden output
│   │   ├── 16x24/                 # Different architecture
│   │   └── 16x32/
│   └── fixed/                     # Fixed-point implementation
│       ├── 16x16/
│       └── 16x24/
│
└── hls/                           # HLS projects per design point
    ├── dse_int_d16x16_m24_u1d_81d450bd/   # Design point ID
    │   ├── dse_config.h           # Generated config with DSE_HIDDEN_FEATURES=16
    │   ├── csim.tcl, synth.tcl    # HLS scripts
    │   └── solution1/             # HLS outputs
    └── dse_int_d16x24_m24_u1d_eff98308/   # Different design point
        └── dse_config.h           # DSE_HIDDEN_FEATURES=24
```

**Why this matters:**
- ✅ **Prevents corruption:** 16x24 weights won't overwrite 16x16 weights
- ✅ **Enables parallelism:** Multiple architectures can train/synthesize simultaneously
- ✅ **Clean caching:** Rerunning 16x16 doesn't invalidate 16x24 results
- ✅ **Reproducibility:** Each architecture has self-contained, versioned data

**Automatic generation:** The DSE automatically generates test vectors when missing:
```python
# DSE checks: Does build/test_vectors_arch/int8_po2/16x16/ exist AND have files?
# If missing or empty → calls generate_test_vectors_per_arch.py
# Result: Fresh test vectors with correct architecture dimensions
```

### How `auto_generated_bitwidths.h` Works

The bitwidth optimizer analyzes actual data ranges during a simulated forward pass and generates optimized bit-widths:

```cpp
// In hls/auto_generated_bitwidths.h (auto-generated)
#define OPT_M_BITS        24    // From M_BITS parameter
#define OPT_ADJ_BITS      16    // Adjacency matrix entries
#define OPT_ACC_BITS      22    // Accumulator (data-driven)
#define OPT_SCALE_BITS    32    // M_BITS + 8 + margin
#define OPT_MULT_BITS     54    // ACC_BITS + SCALE_BITS
```

The HLS code conditionally uses these values:

```cpp
// In hls/graphsage_layer_int8_po2.h
#ifdef USE_OPTIMIZED_BITWIDTHS
  #include "auto_generated_bitwidths.h"
  #define ADJ_BITS   OPT_ADJ_BITS    // Use optimized value
  #define ACC_BITS   OPT_ACC_BITS    // Use optimized value
#else
  #define ADJ_BITS   16              // Conservative default
  #define ACC_BITS   32              // Conservative default
#endif
```

**DSE automatically passes `-DUSE_OPTIMIZED_BITWIDTHS`** to enable the optimized values during synthesis.

### M_BITS Impact on Generated Bitwidths

**Yes, changing M_BITS directly affects the generated values:**

| M_BITS | OPT_SCALE_BITS | OPT_MULT_BITS | Impact |
|--------|----------------|---------------|--------|
| 20     | 28 (20+8)      | ~50           | Smaller multipliers, ~10-15% fewer LUTs |
| 24     | 32 (24+8)      | ~54           | Baseline precision |

The optimizer recalculates all dependent bitwidths when M_BITS changes:
- `SCALE_BITS = M_BITS + 8` (fixed relationship)
- `MULT_BITS = ACC_BITS + SCALE_BITS` (depends on SCALE_BITS)

---

## Stage 1: Algorithm Level

These parameters affect model accuracy, parameter count, and memory footprint.

### `in_channels_reduced`
**What it is:** Input feature dimension after projection layer.

**How it works:**
- Original Cora features: 1433 dimensions
- Projection layer: `Linear(1433 → in_channels_reduced)`
- Reduces memory and computation for GNN layers

**Current values:** `[16, 24]`

**Impact:**
- **Lower (16):** Fewer parameters, faster training, less FPGA memory
- **Higher (24):** More expressive, potentially better accuracy
- **Hardware:** Affects `IN_FEATURES` constant in HLS code

---

### `hidden_channels`
**What it is:** Dimension of hidden layer (output of first SAGEConv, input to second).

**How it works:**
- Layer 1: `SAGEConv(in_channels → hidden_channels)`
- Layer 2: `SAGEConv(hidden_channels → 7 classes)`

**Current values:** `[24, 32, 48]`

**Impact:**
- **Lower (24):** Fewer weights, less computation, fewer DSPs
- **Higher (48):** More capacity, better feature extraction
- **Hardware:** Affects `HIDDEN_FEATURES` and linear layer multiplier count

**Tradeoff:**
- `hidden=24`: ~384 multipliers in linear layers
- `hidden=48`: ~768 multipliers (2× DSP usage)

---

### `dropout`
**What it is:** Dropout probability during training (0.0 to 1.0).

**Current values:** `[0.5]`

**Impact:**
- Training only (disabled during inference)
- Affects regularization and generalization
- No hardware impact

---

### `root_weight`
**What it is:** Whether to include self-loops (add node's own features in aggregation).

**Current values:** `[true]`

**Impact:**
- `true`: Aggregation includes `self_features + neighbor_features`
- `false`: Only neighbor features
- Typically improves accuracy
- Minimal hardware impact (one extra addition)

---

### `num_layers`
**What it is:** Number of SAGEConv layers.

**Current values:** `[2]` (fixed for current implementation)

**Impact:**
- More layers = deeper network, more hops of neighbor info
- Hardware: Each layer needs separate weights, more cycles

---

## Stage 2: Quantization Level

These parameters control fixed-point precision in hardware.

### `M_BITS`
**What it is:** Fractional bits for intermediate fixed-point calculations.

**How it works:**
```
float_value = fixed_point_value / 2^M_BITS
```

**Current values:** `[20, 24]`

**Impact:**
- **Lower (20):**
  - Smaller multipliers (e.g., 28-bit instead of 32-bit)
  - Fewer LUTs per multiplier
  - Slightly lower precision
  
- **Higher (24):**
  - More precision, better accuracy
  - Larger multipliers
  - More resources

**Tradeoff:**
- M=20: ~10-15% fewer LUTs, <0.5% accuracy loss
- M=24: Baseline precision

---

### `K_BITS`
**What it is:** Scaling factor for adjacency matrix normalization.

**Current values:** `[12]` (fixed)

**How it works:**
- Adjacency normalized by sqrt(degree)
- Stored as `int(A_norm * 2^K_BITS)`

**Impact:**
- Higher = more precision for edge weights
- Fixed at 12 gives good accuracy with 16-bit adj_t

---

### `bitwidth_margin`
**What it is:** Safety margin added to minimum required bit-widths.

**Current values:** `[3]`

**How it works:**
```
actual_bitwidth = ceil(log2(max_observed_value)) + margin
```

**Impact:**
- Prevents overflow from unexpected values
- Margin of 3 bits = 8× headroom

---

### `bitwidth_method`
**What it is:** How to determine optimal bit-widths.

**Current values:** `["data_driven"]`

**Options:**
- `"data_driven"`: Analyze actual runtime values
- `"conservative"`: Use worst-case theoretical bounds

---

### `m_bits_relation_to_optimizer`
**IMPORTANT CLARIFICATION:** M_BITS is a **user-specified input** to the quantization system, NOT an output of the bitwidth optimizer.

**How the pipeline works:**

1. **User specifies M_BITS** (fractional precision for fixed-point)
   - Example: `M_BITS=12` means 12 fractional bits

2. **Bitwidth optimizer USES M_BITS to calculate:**
   - `ACC_BITS`: Accumulator width (prevents overflow during sum)
   - `SCALE_BITS`: Scale factor precision
   - `MULT_BITS`: Intermediate multiplication result width
   - `ADJ_BITS`: Adjacency-weighted feature width

3. **These optimized bitwidths are OUTPUTS:**
   - Saved to `auto_generated_bitwidths.h`
   - Used in HLS synthesis

**Analogy:**
- M_BITS = "I want 12 bits of decimal precision"
- Optimizer = "To achieve that without overflow, you need 24-bit accumulators"

**Design space:**
- You explore different M_BITS values (e.g., 8, 10, 12)
- Optimizer automatically calculates safe bitwidths for each
- HLS synthesis shows resulting resource usage

---

## Stage 2.5: Q Format Parameters (Fixed-Point Only)

For the `fixed` implementation, you can directly specify Q(W,I) fixed-point formats instead of using INT8 quantization. This provides fine-grained control over precision and range.

**See [QFORMAT_EXPLORATION_GUIDE.md](QFORMAT_EXPLORATION_GUIDE.md) for comprehensive documentation.**

### Q Format Notation: Q(W,I)
- **W**: Total bit-width (e.g., 16, 24, 32)
- **I**: Integer bits (determines range)
- **F**: Fractional bits = W - I (determines precision)

**Example:** Q(16,8) = 16 total bits, 8 integer, 8 fractional
- Range: [-128, 127.996]
- Precision: 1/256 ≈ 0.004

---

### `data_w`, `data_i` (Activation Q Format)
**What it is:** Q format for node feature activations.

**Current values:**
```yaml
hls:
  fixed:
    data_w: [16, 24, 32]
    data_i: [8, 12, 16]
```

**How to choose:**
- **Large I (more integer bits):** Wider range, less precision
  - Example: Q(24,16) = range [-32768, 32767], precision 1/256
- **Small I (more fractional bits):** Narrower range, more precision
  - Example: Q(24,8) = range [-128, 127], precision 1/65536

**Impact:**
- Wider W → More DSPs (larger multipliers)
- More fractional bits (W-I) → Better accuracy (if values fit in range)
- Too few integer bits → Clipping/overflow → Accuracy loss

---

### `weight_w`, `weight_i` (Weight Q Format)
**What it is:** Q format for learned weights.

**Current values:**
```yaml
hls:
  fixed:
    weight_w: [16, 24]
    weight_i: [4, 8]
```

**Why different from activations:**
- Weights are typically smaller values (normalized)
- Can use fewer integer bits (e.g., I=4 for range [-8, 7.9])
- More fractional bits for precision

**Impact:**
- `weight_w × data_w` determines multiplier size
- Example: 24-bit weights × 24-bit data = 48-bit product
- Accumulator must be sized accordingly

---

### `acc_w`, `acc_i` (Accumulator Q Format)
**What it is:** Q format for intermediate sums.

**Current values:**
```yaml
hls:
  fixed:
    acc_w: [32, 48]
    acc_i: [16, 24]
```

**Why larger:**
- Accumulation increases magnitude (sum of many terms)
- Must prevent overflow: `acc_i >= data_i + log2(num_neighbors)`
- Example: 8 neighbors with I=8 → need at least I=11 for accumulator

**Impact:**
- Larger accumulators → More LUT usage (wide adders)
- Too small → Overflow → Catastrophic accuracy loss
- Too large → Wasted resources

**DSE automatically validates:** I < W and F >= 4 (minimum precision)

---

### Q Format Exploration Workflow

**Option 1: Fast software evaluation** (recommended first step)
```bash
python src/evaluate_qformat.py \
    --config configs/model_config.yaml \
    --data-w 16 24 32 \
    --data-i 8 12 16 \
    --weight-w 16 24 \
    --weight-i 4 8 \
    --acc-w 32 48 \
    --acc-i 16 24
```
**Result:** Accuracy for each Q format in ~5-10 minutes (no HLS)

**Option 2: Full HLS synthesis**
```bash
python src/explore_design_space.py \
    --config configs/design_space_qformat.yaml
```
**Result:** Accuracy + resource usage, ~30 min per design point

**Recommended strategy:**
1. Use `evaluate_qformat.py` to find Q formats with acceptable accuracy
2. Use those Q formats in HLS DSE to measure real resource usage
3. Select Pareto-optimal points (accuracy vs DSP/LUT)

---

## Stage 3: HLS Implementation

These parameters control hardware synthesis and optimization.

---

## 3.1 Implementation Selection

### `implementations`
**What it is:** Which HLS code variant to synthesize.

**Current values:** `["int8_po2", "fixed"]`

**Options:**

#### `int8_po2` (Recommended)
- **Datatype:** INT8 activations/weights
- **Scaling:** Power-of-2 scales (bit-shifts, no DSP multipliers)
- **Pros:**
  - No DSP48s for scaling (~500-700 DSP savings!)
  - Shorter critical path
  - Better timing
- **Cons:**
  - Small quantization error from PO2 approximation (~0.2% accuracy)
- **Use case:** Best for DSP-constrained designs

#### `fixed`
- **Datatype:** Configurable `ap_fixed<W,I>` types
- **Scaling:** Handled by wider accumulators
- **Pros:**
  - Flexible precision (can go lower than INT8)
  - Good for exploring bit-width tradeoffs
- **Cons:**
  - Uses more DSPs than int8_po2
- **Use case:** Research, precision exploration

#### `int8` (Skipped)
- Like int8_po2 but with arbitrary scales (uses DSP multipliers)
- Worse than int8_po2 in every way

#### `float` (Skipped)
- FP32 - doesn't fit on device (283% DSP usage)

#### `ptq` (Skipped)
- Known bugs, broken implementation

---

## 3.2 Message Passing: Dense vs Sparse Implementation

A critical design decision for GNN hardware is how to implement the message passing (aggregation) phase. The current implementation uses **dense adjacency matrices**, but sparse implementations are worth considering for larger or sparser graphs.

### Current Dense Implementation

**File:** `hls/graphsage_layer_int8_po2.h` (and other variants)

**Structure:**
```cpp
// Dense adjacency: iterate over all possible neighbors
for (int i = 0; i < N_NODES; i++) {          // nodes
  for (int f = 0; f < N_FEATURES; f++) {     // features
    for (int j = 0; j < N_NODES; j++) {      // all neighbors
      acc += adjacency[i][j] * features[j][f];
    }
  }
}
```

**Characteristics:**
- **Operations per layer:**
  - Layer 1: 8 nodes × 16 features × 8 neighbors = 1,024 multiplications
  - Layer 2: 8 nodes × 24 features × 8 neighbors = 1,536 multiplications
  - **Total: 2,560 multiplications**

- **For Cora 8-node subgraph (20 edges, 68.75% sparsity):**
  - Useful multiplications: 320 + 480 = 800
  - **Wasted operations: 1,760 (69%!)** - multiplying by zero adjacency values

- **Hardware costs (from synthesis reports):**
  - Aggregation DSPs: ~200-300 (15% of total)
  - Linear layer DSPs: ~1,500-2,000 (85% of total)
  - **Total: ~1,700-2,300 DSPs**

**Advantages:**
- ✅ Simple, predictable hardware
- ✅ Fully unrollable (static loop bounds)
- ✅ Achieves II=1 with pipelining
- ✅ Latency: N_nodes × N_features cycles (~128 cycles for Layer 1)
- ✅ Known graph structure at synthesis time (can optimize away zeros)

**Disadvantages:**
- ❌ Wastes computation on zero adjacency values (69% for Cora subgraph)
- ❌ Wastes memory storing full adjacency matrix
- ❌ DSP usage scales as O(N²) even for sparse graphs

---

### Alternative: Sparse Neighbor-List Implementation

**File:** `hls/graphsage_sparse.h` (exists but not integrated in DSE)

**Structure:**
```cpp
// Neighbor list format
int neighbors[N_NODES][MAX_DEGREE] = {
  {1, 5, 7, -1},    // node 0 has 3 neighbors
  {0, 4, 6, -1},    // node 1 has 3 neighbors  
  {5, -1, -1, -1},  // node 2 has 1 neighbor
  // ...
};

// Sparse aggregation
for (int i = 0; i < N_NODES; i++) {
  for (int f = 0; f < N_FEATURES; f++) {
    for (int d = 0; d < MAX_DEGREE; d++) {
      int j = neighbors[i][d];
      if (j >= 0) {  // valid neighbor
        acc += edge_weights[i][d] * features[j][f];
      }
    }
  }
}
```

**Characteristics:**
- **Operations per layer (Cora subgraph, max_degree=4):**
  - Layer 1: 8 nodes × 16 features × 4 max_deg = 512 operations
  - Layer 2: 8 nodes × 24 features × 4 max_deg = 768 operations
  - **Total: 1,280 operations** (50% reduction vs dense)
  - Actual useful: 800 (still ~37% wasted due to max_degree padding)

- **Memory savings:**
  - Dense: 8×8 = 64 adjacency entries
  - Sparse: 8×4 = 32 entries (50% reduction)

**Advantages:**
- ✅ 50% fewer operations than dense (for max_degree << N_nodes)
- ✅ Estimated DSP savings: ~100-150 vs ~200-300 (50% reduction in aggregation)
- ✅ Scales better for larger graphs (O(N×D) vs O(N²), D=max_degree)
- ✅ Memory-efficient for sparse graphs
- ✅ Reconfigurable - can change graph structure without resynthesis

**Disadvantages:**
- ❌ More complex control logic (conditional on j >= 0)
- ❌ May hurt II (initiation interval) due to data dependencies
- ❌ Irregular memory access patterns (neighbors[i][d])
- ❌ Still wastes computation when actual_degree < MAX_DEGREE

**Latency comparison:**
- **Option A: Unroll d loop (max_degree)**
  - Same II=1 as dense
  - Same latency: 8 × 16 = 128 cycles
  - Trades DSPs for better utilization

- **Option B: Pipeline d loop**
  - May hurt II due to conditional (j >= 0)
  - Latency: 8 × 16 × 4 = 512 cycles (4× slower!)
  - But uses fewer resources

---

### Resource Analysis: Where are the DSPs?

**Critical insight:** Aggregation is only 15% of total DSP usage!

```
DSP Breakdown (from synthesis reports):
┌─────────────────────┬──────────┬─────────┐
│ Component           │ DSPs     │ % Total │
├─────────────────────┼──────────┼─────────┤
│ Aggregation         │ 200-300  │ 15%     │
│ Linear layers       │ 1500-2000│ 85%     │
│ Total               │ 1700-2300│ 100%    │
└─────────────────────┴──────────┴─────────┘

Sparse aggregation savings:
- Aggregation: 50% of 15% = 7.5% total DSP savings
- Linear layers: 0% savings (unchanged)
- Net benefit: ~150 DSPs saved (7.5% of 2000)
```

**Implication:** Optimizing aggregation has limited impact. The linear layers dominate!

---

### Recommendations

#### ✅ **Keep Dense Implementation** if:
1. **Latency is critical** - Need <1μs inference (dense achieves ~128 cycles)
2. **DSP budget is sufficient** - ~2,000 DSPs available (VU13P has 12,288)
3. **Graph is known at synthesis time** - Can optimize structure
4. **You value simplicity** - Predictable, well-tested hardware

#### ✅ **Switch to Sparse Implementation** if:
1. **DSP resources are tight** - Approaching device limits
2. **Scaling to larger graphs** - 16-32+ nodes where O(N²) becomes prohibitive
3. **Latency budget allows 2-4× increase** - Can pipeline the d loop
4. **Graph changes between runs** - Need reconfigurable hardware

#### 🎯 **Best Optimization: Focus on Linear Layers!**

Instead of optimizing aggregation (15% of DSPs):
- **Reduce bit-width:** INT4 weights saves 50% DSPs on linear layers
- **Mixed precision:** INT8 activations + INT4 weights
- **Result:** 40-50% total DSP savings (vs 7.5% from sparse aggregation)

**Recommended hybrid approach:**
1. Keep **dense aggregation** (simple, fast, only 15% of DSPs)
2. Optimize **linear layers** with INT4 weights (40% DSP savings)
3. Use **PO2 scaling** (eliminates scale multipliers)

**Expected result:**
- 40% DSP reduction (vs 7.5% from sparse)
- Same latency (dense, II=1)
- Simple hardware (no irregular access)
- <1% accuracy loss (INT4 with QAT)

---

### Future Work: When to Revisit Sparse

Consider sparse implementation when:
- Graph size exceeds 32 nodes (dense becomes O(N²) = 1024 iterations)
- Multiple smaller graphs processed in parallel (sparse saves memory bandwidth)
- Dynamic graphs (run-time reconfiguration needed)
- Targeting smaller FPGAs (e.g., Zynq) with <5,000 DSPs

For current Cora subgraph (8 nodes, 20 edges):
- **Dense is optimal** - simple, fast, only 15% of DSP budget

---

## 3.3 Loop Unrolling

Loop unrolling creates parallel hardware by replicating computation units.

### `unroll.nodes`
**What it is:** How many node iterations to unroll in parallel.

**Current values:** `[1, 8]`

**How it works:**
```cpp
// unroll=1 (sequential)
for (int i = 0; i < 8; i++) {
    // Process one node per cycle
}

// unroll=8 (fully parallel)
for (int i = 0; i < 8; i++) {
#pragma HLS UNROLL
    // Process all 8 nodes simultaneously
}
```

**Impact:**
- `unroll=1`: 
  - Latency: 8 cycles
  - Resources: 1× multipliers
  - II can be 1 if pipelined
  
- `unroll=8`:
  - Latency: 1 cycle
  - Resources: 8× multipliers
  - Fully parallel

**Tradeoff:**
- 8× unroll = 8× DSPs but 8× faster

---

### `unroll.features_agg` and `unroll.features_lin`
**What it is:** Feature dimension unroll factor for aggregation and linear layers.

**Current values:** `[1, 4]`

**How it works:**
```cpp
// Aggregation: for each feature
AGG_F: for (int f = 0; f < N_FEATURES; f++) {
#pragma HLS UNROLL factor=4
    // Unroll by 4: process 4 features in parallel
}
```

**Impact:**
- `unroll=1`: Sequential, minimal resources
- `unroll=4`: Process 4 features/cycle, 4× multipliers

**Demand calculation:**
```
AGG demand = unroll_nodes × unroll_features_agg
LIN demand = unroll_nodes × unroll_features_lin

Example: nodes=8, feat=4 → demand=32 multipliers
```

---

### `unroll.features_relu`
**What it is:** ReLU unroll factor.

**Current values:** `[1]`

**Why fixed at 1:**
- ReLU is cheap (comparator + mux)
- Not worth the area for parallel ReLU units
- Doesn't affect latency much

---

## 3.4 Pipelining

Pipelining allows new iterations to start before previous ones finish.

### `pipeline.agg_ii` and `pipeline.lin_ii`
**What it is:** Initiation Interval (cycles between starting new iterations).

**Current values:** `[1]`

**How it works:**
```cpp
for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
    // New iteration starts every cycle
}
```

**II values:**
- `II=1`: New iteration every cycle (best throughput)
- `II=2`: New iteration every 2 cycles (if dependencies exist)
- `II>2`: Typically indicates performance issue

**Impact:**
- Lower II = higher throughput
- II=1 is goal, but requires sufficient parallelism
- If allocation limits multipliers, II may increase

---

## 3.5 Resource Binding

Controls which hardware resources implement operations. This is a critical parameter for trading DSP usage vs LUT usage.

### `bind_storage_agg` and `bind_storage_lin`
**What it is:** Control whether multiplications use DSP48 blocks or fabric (LUTs).

**Current values:** `["auto"]`

**Options:**

#### `"auto"` (Default)
- Let HLS decide based on heuristics
- Usually picks DSP for 16-bit+ multiplies
- Usually picks fabric for 8-bit multiplies
- Good starting point

#### `"dsp"` (Explicit DSP)
- Force all multiplies to use DSP48E2 blocks
- **Pros:**
  - Faster (single-cycle multiply)
  - Lower LUT usage
  - Better timing (shorter critical path)
- **Cons:**
  - Limited DSP48s (12,288 on VU13P)
  - May cause resource overflow
- **Use case:** When DSPs available, latency critical

#### `"fabric"` (LUT-based)
- Force multiplies to use LUT-based implementation
- **Pros:**
  - Saves DSPs for other operations
  - LUTs are more abundant (~1.7M on VU13P)
- **Cons:**
  - Slower (multi-cycle multiply, more logic levels)
  - Higher LUT usage (~50-100 LUTs per 8-bit multiply)
  - May hurt Fmax
- **Use case:** DSP-constrained designs

---

### Which Operations Are Controlled?

#### `bind_storage_agg`
- Controls **aggregation layer multiplications**: `adjacency × features`
- In code: `edge_weight[d] * x_buffer[j][f]`
- Typical usage: ~200-400 DSPs if bound to DSP

#### `bind_storage_lin`
- Controls **linear layer SCALE multiplications**: `feature × scale_factor`
- In code: `scaled = (lin_result * scale[f]) >> SCALE_BITS`
- **NOT used in int8_po2** (power-of-2 scales = bit-shifts, no multipliers)
- **Used in int8, fixed** implementations
- Typical usage: ~300-500 DSPs if bound to DSP

**Important:** Linear layer weight multiplies (`x * W`) are separate and not controlled by bind pragmas (always optimized by HLS).

---

### Bind Strategy by Implementation

| Implementation | bind_storage_agg | bind_storage_lin | Rationale |
|----------------|------------------|------------------|------------|
| **int8_po2**   | "dsp" or "auto" | N/A (no scale mults) | PO2 scales = bit-shifts, no multipliers needed |
| **int8**       | "dsp" or "auto" | "fabric" | Scale mults can use fabric, save DSPs for critical path |
| **fixed**      | "dsp" or "auto" | "dsp" or "auto" | Depends on Q format and available DSPs |

**Current presentation config:** Uses "auto" for simplicity (not exploring fabric binding).

---

### DSP vs Fabric Trade-off

**Example:** INT8 non-PO2 with 16 hidden channels

| Bind Config | DSP Usage | LUT Usage | Latency | Fmax |
|-------------|-----------|-----------|---------|------|
| agg=dsp, lin=dsp | 2,200 | 45k | 100 cycles | 380 MHz |
| agg=dsp, lin=fabric | 1,700 | 78k | 105 cycles | 360 MHz |
| agg=fabric, lin=fabric | 1,200 | 125k | 115 cycles | 340 MHz |

**Pareto front:** agg=dsp + lin=fabric often best compromise.

**When to explore fabric binding:**
- DSP usage >80% (need to reduce)
- LUT usage <50% (have headroom)
- Latency budget allows 10-15% increase
- Fmax >300 MHz target (timing not critical)

---

## 3.6 Resource Allocation (DSP Reuse)

**Most important for controlling resource/performance tradeoff!**

### Are Unroll and Allocation Redundant?

**No - they control different aspects:**

1. **Unroll factors** → Set DEMAND (how much parallelism the algorithm exposes)
   - Controls loop trip counts
   - Affects memory access patterns
   - Determines pipeline structure

2. **Allocation limits** → Set SUPPLY (how many physical multipliers HLS creates)
   - Controls hardware resources
   - Forces time-multiplexing when supply < demand

3. **Reuse factor** → DERIVED metric (demand / supply)
   - Not a parameter, it's the result
   - Indicates how many times each multiplier is used

**Example showing they're NOT redundant:**

| Config | Nodes | Features | Demand | Alloc | Reuse | Loop Structure | Resources |
|--------|-------|----------|--------|-------|-------|----------------|-----------|
| A      | 8     | 4        | 32     | 16    | 2×    | 8 node iterations, 4 feat/iter | 16 DSPs |
| B      | 4     | 8        | 32     | 16    | 2×    | 4 node iterations, 8 feat/iter | 16 DSPs |
| C      | 1     | 32       | 32     | 16    | 2×    | 1 node iteration, 32 feat/iter | 16 DSPs |

All have same reuse (2×) and same DSPs (16), but **DIFFERENT latency and II**:
- Config A: Better for node-parallel workloads, 8 outer loop iterations
- Config B: Better for feature-parallel workloads, 4 outer loop iterations  
- Config C: No node parallelism, sequential nodes, 1 outer loop iteration (highest latency)

**Concrete Code Example:**

```cpp
// Config A: unroll_nodes=8, unroll_feat=4, alloc=16
// Loop structure: 8 parallel nodes, 4 parallel features
AGG_I: for (int i = 0; i < 8; i++) {
#pragma HLS UNROLL  // All 8 nodes in parallel
    AGG_F: for (int f = 0; f < 16; f += 4) {  // Process 4 features at a time
#pragma HLS UNROLL factor=4
        // 8 nodes × 4 features = 32 multipliers needed
        // But only 16 allocated → each used 2× (time-multiplexed)
        // Latency: 1 outer iteration × 4 inner iterations = 4 cycles
    }
}

// Config C: unroll_nodes=1, unroll_feat=32, alloc=16
// Loop structure: 1 node at a time, 32 parallel features
AGG_I: for (int i = 0; i < 8; i++) {  // Sequential nodes
    AGG_F: for (int f = 0; f < 16; f += 32) {
#pragma HLS UNROLL factor=32  // All features in parallel
        // 1 node × 32 features = 32 multipliers needed
        // But only 16 allocated → each used 2× (time-multiplexed)
        // Latency: 8 outer iterations × 1 inner iteration = 8 cycles
    }
}
```

**Result:**
- Same resources (16 DSPs)
- Same reuse factor (2×)
- **Different latency:** Config A = 4 cycles, Config C = 8 cycles
- Different memory access patterns, different II potential

**However, we DO filter truly redundant cases:**
```
If demand=4 and alloc=16 → SKIP (limit > demand, no effect)
If demand=4 and alloc=4  → KEEP (baseline, no reuse)
If demand=4 and alloc=2  → KEEP (2× reuse)
```

### `allocation.agg_mul_limit` and `allocation.lin_mul_limit`
**What it is:** Maximum number of multiplier instances (forces time-multiplexing).

**Current values:** `["unlimited", 16, 4, 1]`

**How it works:**
```cpp
// Without allocation limit (unlimited):
// For demand=32, creates 32 parallel multipliers
AGG_J: for (int j = 0; j < 32; j++) {
#pragma HLS UNROLL
    result += a[j] * b[j];  // 32 multipliers
}

// With allocation limit=8:
// Only 8 multiplier instances, reused 4 times
#pragma HLS allocation instances=mul limit=8 operation
AGG_J: for (int j = 0; j < 32; j++) {
#pragma HLS UNROLL
    result += a[j] * b[j];  // 8 multipliers × 4 reuse
}
```

**Reuse factor:**
```
reuse = demand / limit
demand = unroll_nodes × unroll_features

Example:
  nodes=8, features=4 → demand=32
  limit=8 → reuse=4× (each multiplier used 4 times)
```

**Impact:**

#### `"unlimited"` (baseline)
- No reuse, full parallelism
- **DSPs:** Maximum (32 for demand=32)
- **Latency:** Minimum
- **II:** 1
- **Use case:** Latency-critical, DSPs available

#### `limit=16` (2× reuse)
- Moderate sharing
- **DSPs:** 16 (50% savings)
- **Latency:** ~2× higher
- **II:** May stay at 1 with good pipelining
- **Use case:** Balanced tradeoff

#### `limit=4` (8× reuse)
- High sharing
- **DSPs:** 4 (87.5% savings)
- **Latency:** ~8× higher
- **II:** Likely increases to 2-4
- **Use case:** Extreme DSP constraints

#### `limit=1` (maximum reuse)
- Single multiplier shared for everything
- **DSPs:** 1 (96.9% savings)
- **Latency:** ~32× higher
- **II:** Very high (8+)
- **Use case:** Reference only, usually too slow

**Validation:**
- Only valid if `demand % limit == 0` (clean divisor)
- Invalid configs automatically filtered
- Example: demand=32, limit=24 → INVALID (32%24≠0)

---

## 3.7 Target Device

### `target_device`
**Current value:** `"xcvu13p-fsga2577-1-e"`

**Device specs:**
- Virtex UltraScale+ VU13P
- **DSP48E2:** 12,288 slices
- **LUTs:** ~1.7M
- **BRAM:** 36 Mb
- **Clock:** 200-500 MHz typical

### `target_clock_ns`
**Current value:** `2.77` (361 MHz)

**Impact:**
- Aggressive timing target
- May require retiming, pipelining
- Lower = higher throughput but harder to meet timing

---

## Filtering and Pareto Analysis

### Stage 1 Filter: Accuracy Threshold
```yaml
algorithm:
  accuracy_threshold: 0.60  # Keep models ≥60% accuracy
```

**Effect:**
- 6 algo configs → typically 3-4 pass to Stage 2

### Stage 2 Filter: Accuracy + Bitwidth Feasibility
- Checks if PTQ/QAT quantization maintains accuracy
- Verifies bitwidths don't cause overflow
- ~12 configs → typically 2-3 pass to Stage 3

### Stage 3 Filter: HLS Feasibility + Pareto
```yaml
hls:
  min_fmax_mhz: 200         # Reject designs below 200 MHz
  max_dsp_pct: 100          # Reject designs over 100% DSP usage
```

**Pareto objectives:**
```yaml
pareto:
  objectives:
    accuracy: max           # Higher is better
    latency_cycles: min     # Lower is better
    dsp_used: min           # Lower is better
```

**Result:**
- Only non-dominated points on Pareto frontier
- Typically 10-30 final designs from ~500 HLS syntheses

---

## Expected Design Space Size

### Before Filtering
```
Algorithm:  6 configs (in_channels × hidden_channels)
Quant:      2 configs (M_BITS)
HLS:        2 × 8 × 16 = 256 (impl × unroll × allocation)

Total: 6 × 2 × 256 = 3,072 theoretical points
After allocation filtering: 1,776 valid points
```

### After Staged Filtering

**Stage 1 (Python training):**
- Evaluate: 6 models
- Pass: ~3-4 (above accuracy threshold)
- Time: ~5 min/model = 30 min total

**Stage 2 (Quantization):**
- Evaluate: 3-4 models × 2 M_BITS = 6-8 configs
- Pass: ~2-3 (acceptable quantization loss)
- Time: ~1 min/config = 10 min total

**Stage 3 (HLS synthesis):**
- Evaluate: 2-3 configs × 256 HLS variations = 512-768 syntheses
- Pass (Pareto): ~10-30 designs
- Time: ~30 min/synthesis = 256-384 hours (10-16 days)

**Parallelization:**
- With 16 parallel jobs: ~16-24 hours for full exploration

---

## Output Data and Result Storage

### Per-Design-Point Artifacts

For each evaluated design point, the following files are saved:

**Directory structure:**
```
build/design_points/
  design_<id>/
    config.json           # Complete design point parameters
    summary.txt           # Human-readable summary
    training_history.json # Algorithm training metrics (if trained)
    quantization.json     # PTQ/QAT metrics (if quantized)
    hls_report.json       # HLS synthesis results (if synthesized)
    vivado_hls.log        # Full HLS log
```

---

### `config.json` - Complete Parameter Storage

**Contains ALL 50+ design point parameters:**

```json
{
  "id": 42,
  // Algorithm Level
  "in_channels_reduced": 16,
  "hidden_channels": 24,
  "dropout": 0.3,
  "root_weight": true,
  "num_layers": 2,
  "accuracy": 0.782,
  
  // Quantization Level
  "m_bits": 12,
  "k_bits": 8,
  "bitwidth_margin": 3,
  "bitwidth_method": "data_driven",
  
  // Q Format (if fixed implementation)
  "data_w": 24,
  "data_i": 12,
  "weight_w": 16,
  "weight_i": 8,
  "acc_w": 32,
  "acc_i": 16,
  
  // HLS Implementation
  "implementation": "int8_po2",
  "unroll_nodes": 8,
  "unroll_features_agg": 4,
  "unroll_features_lin": 4,
  "unroll_features_relu": 1,
  "pipeline_agg_ii": 1,
  "pipeline_lin_ii": 1,
  "bind_storage_agg": "dsp",
  "bind_storage_lin": "auto",
  "allocation_agg_mul_limit": 16,
  "allocation_lin_mul_limit": 16,
  
  // HLS Results
  "hls_status": "success",
  "latency_cycles": 145,
  "latency_us": 0.402,
  "fmax_mhz": 361,
  "dsp_used": 1824,
  "dsp_pct": 14.8,
  "lut_used": 52340,
  "lut_pct": 3.1,
  "ff_used": 28456,
  "ff_pct": 0.8,
  "bram_used": 12,
  "uram_used": 0,
  
  // Derived Metrics
  "agg_demand": 32,
  "lin_demand": 32,
  "agg_reuse_factor": 2.0,
  "lin_reuse_factor": 2.0
}
```

**Usage for post-processing:**
```python
import json

# Load specific design point
with open('build/design_points/design_42/config.json') as f:
    dp = json.load(f)
    
print(f"Accuracy: {dp['accuracy']:.3f}")
print(f"DSPs: {dp['dsp_used']} ({dp['dsp_pct']:.1f}%)")
print(f"Latency: {dp['latency_us']:.3f} μs")
print(f"Q Format: Q({dp['data_w']},{dp['data_i']})")
```

---

### `summary.txt` - Human-Readable Summary

**Example for fixed-point implementation:**
```
Design Point 42
===============

Algorithm:
  in_channels: 16
  hidden_channels: 24
  root_weight: true
  accuracy: 78.2%

Quantization:
  M_BITS: 12
  Method: data_driven

Q Format (Fixed-Point):
  Activations: Q(24,12) - 12 fractional bits
  Weights: Q(16,8) - 8 fractional bits
  Accumulators: Q(32,16) - 16 fractional bits

HLS:
  Implementation: fixed
  Unroll: nodes=8, features=4
  Allocation: agg=16, lin=16
  Reuse: 2.0x

Results:
  Latency: 145 cycles (0.402 μs @ 361 MHz)
  DSP: 1824 (14.8%)
  LUT: 52340 (3.1%)
  FF: 28456 (0.8%)
  BRAM: 12
```

---

### Aggregated Results

**`design_space_results.json`:** All design points in one file
```json
[
  {"id": 1, "accuracy": 0.782, "dsp_used": 1824, ...},
  {"id": 2, "accuracy": 0.779, "dsp_used": 912, ...},
  ...
]
```

**`design_space_results.csv`:** Same data, CSV format for Excel/analysis
```csv
id,accuracy,dsp_used,dsp_pct,latency_cycles,fmax_mhz,...
1,0.782,1824,14.8,145,361,...
2,0.779,912,7.4,289,361,...
```

**Pareto-optimal subset:** `pareto_optimal.json` and `pareto_optimal.csv`
- Only non-dominated designs (best accuracy/latency/DSP trade-offs)
- Typically 10-30 designs from 500+ evaluated

---

## Visualization and Plotting

### Recommended Plots for Presentation

#### 1. **Pareto Front: Accuracy vs DSP Usage**
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('build/pareto_optimal.csv')
plt.scatter(df['dsp_used'], df['accuracy'], c=df['latency_cycles'])
plt.xlabel('DSP Usage')
plt.ylabel('Accuracy')
plt.colorbar(label='Latency (cycles)')
plt.title('Pareto-Optimal Designs')
```
**Shows:** Best trade-off between accuracy and resources.

---

#### 2. **Latency vs DSP (Color = Accuracy)**
```python
plt.scatter(df['dsp_used'], df['latency_cycles'], c=df['accuracy'], cmap='viridis')
plt.xlabel('DSP Usage')
plt.ylabel('Latency (cycles)')
plt.colorbar(label='Accuracy')
plt.title('Resource-Latency Trade-off')
```
**Shows:** How allocation reuse affects latency.

---

#### 3. **Unroll Factor Impact**
```python
for unroll in [1, 4, 8]:
    subset = df[df['unroll_nodes'] == unroll]
    plt.plot(subset['allocation_agg_mul_limit'], subset['latency_cycles'], 
             label=f'Unroll={unroll}', marker='o')
plt.xlabel('Allocation Limit')
plt.ylabel('Latency (cycles)')
plt.legend()
plt.title('Effect of Unrolling on Latency')
```
**Shows:** Parallelism scaling.

---

#### 4. **DSP Scaling: Unroll vs Allocation**
```python
plt.figure(figsize=(10, 6))
for alloc in ['unlimited', 16, 4, 1]:
    subset = df[df['allocation_agg_mul_limit'] == alloc]
    plt.plot(subset['unroll_nodes'], subset['dsp_used'], 
             label=f'Alloc={alloc}', marker='s')
plt.xlabel('Unroll Factor (nodes)')
plt.ylabel('DSP Usage')
plt.legend()
plt.title('DSP Usage: Demand vs Allocation')
```
**Shows:** Allocation limits cap DSP usage regardless of unroll.

---

#### 5. **Bind Operation Trade-off** (if explored)
```python
for bind in ['dsp', 'fabric']:
    subset = df[df['bind_storage_lin'] == bind]
    plt.scatter(subset['dsp_used'], subset['lut_used'], label=bind, alpha=0.7)
plt.xlabel('DSP Usage')
plt.ylabel('LUT Usage')
plt.legend()
plt.title('DSP vs LUT: Bind Operation Impact')
```
**Shows:** Fabric binding trades DSPs for LUTs.

---

#### 6. **Q Format Precision Analysis** (if explored)
```python
df['fractional_bits'] = df['data_w'] - df['data_i']
plt.scatter(df['fractional_bits'], df['accuracy'], c=df['dsp_used'], cmap='plasma')
plt.xlabel('Fractional Bits (Precision)')
plt.ylabel('Accuracy')
plt.colorbar(label='DSP Usage')
plt.title('Q Format: Precision vs Accuracy')
```
**Shows:** How fractional bits affect accuracy and resources.

---

#### 7. **Architecture Scaling**
```python
for hidden in [16, 24, 32]:
    subset = df[df['hidden_channels'] == hidden]
    plt.plot(subset['allocation_agg_mul_limit'], subset['dsp_used'],
             label=f'Hidden={hidden}', marker='d')
plt.xlabel('Allocation Limit')
plt.ylabel('DSP Usage')
plt.legend()
plt.title('Model Size Impact on Resources')
```
**Shows:** Larger models need more DSPs at same allocation.

---

### Suggested Plot Subset for Presentation

For a 10-minute presentation, show:
1. **Pareto Front** (accuracy vs DSP) - main result
2. **Latency vs DSP** (color = accuracy) - performance trade-off
3. **Unroll impact** - show parallelism benefit
4. **Allocation impact** - show resource reuse

Optional if you explored:
5. **Bind operations** - DSP vs LUT trade-off
6. **Q format** - precision vs accuracy

---

## Usage Examples

### Quick exploration (skip HLS)
```bash
python src/explore_design_space.py \
    --config configs/design_space.yaml \
    --skip-hls
```

### Full exploration
```bash
python src/explore_design_space.py \
    --config configs/design_space.yaml \
    --parallel-jobs 16
```

### Dry run (preview design points)
```bash
python src/explore_design_space.py \
    --config configs/design_space.yaml \
    --dry-run
```

### Resume interrupted run
```bash
python src/explore_design_space.py \
    --config configs/design_space.yaml \
    --resume
```

---

## Recommended Configurations

### For fastest exploration (testing pipeline)
```yaml
algorithm:
  in_channels_reduced: [16]
  hidden_channels: [24]
hls:
  unroll:
    nodes: [1]
    features_agg: [1]
  allocation:
    agg_mul_limit: ["unlimited"]
    lin_mul_limit: ["unlimited"]
```
**Result:** ~4 design points, ~2 hours with HLS

### For balanced exploration
```yaml
algorithm:
  in_channels_reduced: [16, 24]
  hidden_channels: [24, 32]
hls:
  unroll:
    nodes: [1, 8]
    features_agg: [1, 4]
  allocation:
    agg_mul_limit: ["unlimited", 16]
    lin_mul_limit: ["unlimited", 16]
```
**Result:** ~200 design points, ~4 days with 16 parallel jobs

### For comprehensive exploration (current config)
```yaml
# Full config as in design_space.yaml
```
**Result:** ~1,776 design points, ~10-16 days with 16 parallel jobs

---

## Summary

The design space exploration provides:
1. **Algorithm optimization:** Find best architecture for accuracy
2. **Quantization optimization:** Minimize bit-widths while preserving accuracy
3. **HLS optimization:** Explore latency/resource tradeoffs via unroll/allocation

The **allocation limits** are the key knob for trading DSPs vs latency:
- Start with `unlimited` (baseline)
- Reduce to 16, 4, or 1 to progressively save DSPs at cost of latency
- Pareto analysis shows the best tradeoff for your constraints
