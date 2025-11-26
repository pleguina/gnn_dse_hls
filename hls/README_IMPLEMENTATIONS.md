# GraphSAGE HLS Implementations Summary

This directory contains **three** different HLS implementations of GraphSAGE:

## 1. Float (FP32) - BASELINE ✅ WORKING
**Files**: `graphsage_layer_float.{h,cpp}`, `testbench_float.cpp`  
**Status**: ✅ Csim PASSED, ✅ Synthesis PASSED  
**Build dir**: `build/hls/graphsage_float/`

- **Precision**: 32-bit floating-point
- **Accuracy**: Exact match with PyTorch reference
- **Resources**: 34,880 DSPs (283% over-budget for VU13P), 4.45M FFs, 2.14M LUTs
- **Performance**: II=1, 602 cycles latency, 361 MHz
- **Problem**: **Does NOT fit** on VU13P due to DSP usage

## 2. Fixed-Point (ap_fixed) - PARAMETERIZABLE ✅ WORKING
**Files**: `graphsage_layer_fixed.{h,cpp}`, `testbench_fixed.cpp`  
**Status**: ✅ Csim PASSED (with reduced accuracy), ⏳ Synthesis PENDING  
**Build dir**: `build/hls/graphsage_fixed/`  
**TCL Generator**: `generate_graphsage_fixed_tcl.py`

- **Precision**: Configurable Q format (default Q8.8 for data, Q4.12 for weights)
- **Accuracy**: 37/56 outputs >1% error with default Q8.8 (needs higher precision)
- **Purpose**: Experiment with different bit widths to find optimal precision/resource tradeoff
- **How to change**: Edit `generate_graphsage_fixed_tcl.py` config, regenerate TCL
- **Expected resources**: 50-80% less DSPs than float (depending on bit width)

### Fixed-Point Configurations to Try:
```python
# In generate_graphsage_fixed_tcl.py:
"data_w": 24, "data_i": 12   # Q12.12 - better accuracy
"weight_w": 24, "weight_i": 8  # Q8.16
"acc_w": 48, "acc_i": 24      # Q24.24
```

## 3. INT8 PTQ (Post-Training Quantization) - ❌ NOT WORKING YET
**Files**: `graphsage_layer_ptq.{h,cpp}`, `testbench_ptq.cpp`  
**Status**: ❌ Csim FAILED (all outputs wrong), ❌ Synthesis NOT RUN  
**Build dir**: `build/hls/graphsage_ptq/`  
**TCL Generator**: `generate_graphsage_ptq_tcl.py`

- **Precision**: INT8 weights/activations, INT32 accumulators
- **Quantization**: Symmetric (zero_point=0), per-tensor scales
- **Accuracy**: ❌ COMPLETELY WRONG (max error 110 LSB out of ±127 range)
- **Problem**: Quantization logic has fundamental bugs (scale handling incorrect)
- **Expected resources**: ~90% fewer DSPs than float, ~60% less area overall
- **Status**: **NEEDS DEBUGGING** - don't use this version yet!

### PTQ Issues to Fix:
1. Aggregation scale mismatch (input vs output scales)
2. Bias quantization incorrect (should be INT32 in accumulator scale, not INT8)
3. Test vectors may be using wrong reference files

---

## Recommended Workflow

### For Experimentation (Precision vs Resources):
Use **Fixed-Point** version - it's working and parameterizable:
```bash
cd /home/pelayo/work/simple-gnn/hls
python3 generate_graphsage_fixed_tcl.py
cd /home/pelayo/work/simple-gnn/build/hls/graphsage_fixed
rm -rf graphsage_fixed
vitis_hls -f project.tcl
vitis_hls -f csim.tcl  # Check accuracy
vitis_hls -f synth.tcl # Synthesize
```

### For Production (Best Accuracy):
Use **Float** version - it works perfectly but needs a bigger FPGA:
```bash
cd /home/pelayo/work/simple-gnn/build/hls/graphsage_float
vitis_hls -f csim.tcl  # Already tested
vitis_hls -f synth.tcl # Already synthesized
```

### For INT8 PTQ (DON'T USE YET):
**Status**: ❌ Broken - needs debugging
The PTQ implementation has fundamental bugs and produces completely wrong results.
Needs investigation of the quantization logic before it can be used.

---

## File Naming Convention

- `*_float.*` → FP32 floating-point
- `*_fixed.*` → Parameterizable ap_fixed 
- `*_ptq.*` → INT8 Post-Training Quantization (PTQ)
- **NOTE**: No `*_qat.*` files - QAT was a naming mistake, everything is PTQ

---

## Next Steps

1. ✅ **Fixed-point**: Try Q12.12 or higher precision to improve accuracy
2. ✅ **Fixed-point**: Run synthesis to see actual resource usage
3. ❌ **PTQ**: Debug quantization logic (scale handling, bias format)
4. ❌ **PTQ**: Verify test vector generation matches HLS implementation
5. 🔍 **All**: Compare synthesis results (DSP/LUT/FF usage, timing)
