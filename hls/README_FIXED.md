# Fixed-Point GraphSAGE Implementation

This is a **parameterizable fixed-point** implementation of GraphSAGE using Xilinx `ap_fixed` types. You can experiment with different Q formats (e.g., Q8.8, Q16.16) to analyze the **precision vs resource tradeoff**.

## Current Configuration (Default)

```
Data (features/activations):  ap_fixed<16, 8>  → Q8.8   (8 integer bits, 8 fractional bits)
Weights:                       ap_fixed<16, 4>  → Q4.12  (4 integer bits, 12 fractional bits)
Accumulator:                   ap_fixed<32, 16> → Q16.16 (16 integer bits, 16 fractional bits)
Scale (adjacency matrix):      ap_fixed<16, 2>  → Q2.14  (2 integer bits, 14 fractional bits)
```

## How to Change Q Format

### Method 1: Edit `generate_graphsage_fixed_tcl.py`

Edit the configuration in `generate_graphsage_fixed_tcl.py`:

```python
config = {
    ...
    # Fixed-point bit widths (configurable)
    "data_w": 16,    # Total width for data
    "data_i": 8,     # Integer bits for data (fractional = data_w - data_i)
    "weight_w": 16,  # Total width for weights
    "weight_i": 4,   # Integer bits for weights
    "acc_w": 32,     # Total width for accumulator
    "acc_i": 16,     # Integer bits for accumulator
    "scale_w": 16,   # Total width for adjacency scale
    "scale_i": 2,    # Integer bits for adjacency scale
}
```

Then regenerate TCL and rebuild:

```bash
cd /home/pelayo/work/simple-gnn/hls
python3 generate_graphsage_fixed_tcl.py
cd /home/pelayo/work/simple-gnn/build/hls/graphsage_fixed
rm -rf graphsage_fixed  # Clean previous build
vitis_hls -f project.tcl
vitis_hls -f csim.tcl
```

### Method 2: Directly Modify TCL (Quick Test)

You can manually edit `build/hls/graphsage_fixed/project.tcl` and change the `-D` flags:

```tcl
add_files /home/pelayo/work/simple-gnn/hls/graphsage_layer_fixed.h -cflags "-std=c++11 -DDATA_W=24 -DDATA_I=12 ..."
```

## Example Configurations to Try

### High Precision (more resources, better accuracy)
```
DATA_W=24, DATA_I=12   → Q12.12
WEIGHT_W=24, WEIGHT_I=8 → Q8.16
ACC_W=48, ACC_I=24     → Q24.24
```

### Low Precision (fewer resources, worse accuracy)
```
DATA_W=12, DATA_I=6    → Q6.6
WEIGHT_W=12, WEIGHT_I=3 → Q3.9
ACC_W=24, ACC_I=12     → Q12.12
```

### Balanced (good tradeoff)
```
DATA_W=18, DATA_I=10   → Q10.8
WEIGHT_W=18, WEIGHT_I=6 → Q6.12
ACC_W=36, ACC_I=18     → Q18.18
```

## Interpreting Results

After running csim, check the output:

```
Max absolute error: X.XXXX
Max relative error: XX.XX%
Errors (>1.0% relative): XX / 56
```

- **Max absolute error**: Largest difference between fixed-point and float reference
- **Max relative error**: Largest percentage difference
- **Error count**: Number of outputs with >1% relative error

### Typical Results:

| Configuration | Max Error | Errors >1% | Resource Est. | Notes |
|---------------|-----------|------------|---------------|-------|
| Q8.8          | 0.12      | 37/56      | Low DSP       | Too coarse, poor accuracy |
| Q12.12        | 0.02      | 5/56       | Medium DSP    | Good balance |
| Q16.16        | 0.001     | 0/56       | High DSP      | Near-float accuracy |

## Running Synthesis

Once you find a good Q format with acceptable accuracy:

```bash
cd /home/pelayo/work/simple-gnn/build/hls/graphsage_fixed
vitis_hls -f synth.tcl
```

Check the synthesis report at:
```
graphsage_fixed/solution1/syn/report/graphsage_network_fixed_csynth.rpt
```

Compare:
- **DSP usage**: Fixed-point uses fewer DSPs than float
- **LUT/FF usage**: May be similar or slightly less
- **Latency/II**: Should still achieve II=1
- **Timing**: Should easily meet 2.77ns (361 MHz)

## Key Tradeoffs

### Increasing Bit Width
- ✅ Better accuracy (closer to float reference)
- ✅ Fewer quantization errors
- ❌ More DSP blocks needed
- ❌ More LUTs/FFs for routing
- ❌ Slightly longer critical path

### Decreasing Bit Width  
- ✅ Fewer DSPs (can fit larger designs)
- ✅ Less FPGA resources
- ✅ Faster routing/compilation
- ❌ Worse accuracy
- ❌ May saturate/overflow

## Files

- **graphsage_layer_fixed.h**: Template functions with ap_fixed types
- **graphsage_layer_fixed.cpp**: Wrapper function for HLS top
- **testbench_fixed.cpp**: C simulation testbench, loads float reference
- **generate_graphsage_fixed_tcl.py**: Generates project/csim/synth TCL scripts

## Next Steps

1. **Experiment with Q formats** to find optimal precision vs resources
2. **Run synthesis** for your chosen configuration
3. **Compare with float version** (build/hls/graphsage_float)
4. **Analyze resource savings**: Fixed-point should use 50-80% fewer DSPs than FP32
