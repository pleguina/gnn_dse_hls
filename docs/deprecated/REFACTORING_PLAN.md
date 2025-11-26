# Repository Audit and Refactoring Plan

## Current State Analysis

### Models (.pth files in build/models/)
1. **base_graphsage_best.pth** - Full base model
2. **reduced_graphsage_best.pth** - Reduced model (with root_weight)
3. **reduced_graphsage_no_root_best.pth** - Reduced model (HLS-compatible, no root_weight)
4. **reduced_graphsage_qat_no_root_best.pth** - QAT trained model

### Quantized Weights Directories
1. **build/quantized/** - PTQ with root_weight (not HLS-compatible)
2. **build/quantized_no_root/** - PTQ without root_weight (HLS-compatible)
3. **build/quantized_qat/** - QAT quantized weights
4. **build/quantized_int8/** - NEW: Integer-only PTQ (INT32 biases, INT16 adjacency, fixed-point scales)
5. **build/weights_float/** - NEW: Float biases exported for INT8 conversion

### Test Vectors Directories
1. **build/test_vectors/** - Original test vectors
2. **build/test_vectors_float/** - Float model test vectors
3. **build/test_vectors_ptq/** - PTQ test vectors (quantized forward pass)
4. **build/test_vectors_qat/** - QAT test vectors
5. **build/test_vectors_int8/** - NEW: Integer-only PTQ test vectors

### Training History Files
1. **base_model_history.json** - Base model training
2. **reduced_model_history.json** - Reduced model (with root)
3. **reduced_model_no_root_history.json** - Reduced model (no root)
4. **qat_model_qat_no_root_history.json** - QAT training

---

## Problems Identified

### 1. Naming Confusion
- "quantized" vs "quantized_no_root" vs "quantized_qat" - not clear which is which
- "test_vectors" vs "test_vectors_ptq" - redundant naming
- PTQ not explicit in directory names

### 2. Mixed Implementations
- Multiple PTQ variants (float PTQ, int8 PTQ)
- Not clear which one to use for HLS
- Integer-only PTQ added but not integrated into main pipeline

### 3. Directory Structure Issues
- Too many "quantized_*" directories
- Test vectors scattered across multiple directories
- Not clear which artifacts belong to which model variant

---

## Proposed Refactoring

### New Directory Structure

```
build/
├── models/                              # All trained models
│   ├── 1_base_full.pth                 # Base full model (128 hidden)
│   ├── 2_reduced_with_root.pth         # Reduced (24 hidden, with root_weight)
│   ├── 3_reduced_no_root.pth           # Reduced (24 hidden, HLS-compatible)
│   └── 4_qat_no_root.pth               # QAT trained (HLS-compatible)
│
├── training_history/                    # Training curves
│   ├── base_full_history.json
│   ├── reduced_with_root_history.json
│   ├── reduced_no_root_history.json
│   └── qat_no_root_history.json
│
├── weights/                             # Exported weights for each variant
│   ├── float/                          # Float weights (for reference)
│   │   ├── reduced_no_root_*.txt      # Layer weights and biases
│   │   └── qat_no_root_*.txt
│   │
│   ├── ptq_float/                      # PTQ with float quant/dequant
│   │   ├── weights_*.txt              # INT8 weights
│   │   ├── bias_*.txt                 # INT8 biases (output domain)
│   │   └── quant_params.json          # Scales and zero points
│   │
│   ├── ptq_int8/                       # PTQ with integer-only ops
│   │   ├── weights_*.txt              # INT8 weights (same as ptq_float)
│   │   ├── bias_layer1_int32.txt      # INT32 biases (accumulator domain)
│   │   ├── bias_layer2_int32.txt
│   │   ├── adj_matrix_int16.txt       # INT16 fixed-point adjacency
│   │   └── int8_params.json           # M, K, eff_scale_fp, beta_fp
│   │
│   └── qat/                            # QAT quantized weights
│       ├── weights_*.txt
│       ├── bias_*.txt
│       └── quant_params.json
│
├── test_vectors/                        # Test vectors for each variant
│   ├── float/                          # Float model (no quantization)
│   │   ├── input.txt
│   │   ├── output_reference.txt
│   │   └── adjacency.txt
│   │
│   ├── ptq_float/                      # PTQ with float ops
│   │   ├── input_int8.txt
│   │   ├── output_reference_int8.txt
│   │   ├── adjacency_float.txt
│   │   ├── scales.txt
│   │   └── intermediates/             # Layer outputs for debugging
│   │
│   ├── ptq_int8/                       # PTQ with integer-only ops
│   │   ├── input_int8.txt
│   │   ├── output_reference_int8.txt
│   │   ├── adjacency_int16.txt        # Fixed-point adjacency
│   │   ├── config.txt                 # M, K parameters
│   │   └── intermediates/
│   │
│   └── qat/                            # QAT test vectors
│       ├── input_int8.txt
│       ├── output_reference_int8.txt
│       ├── adjacency_float.txt
│       └── scales.txt
│
├── subgraph/                            # Extracted subgraph
│   ├── subgraph_data.npz
│   └── visualization.png
│
├── plots/                               # Training curves and analysis
│   └── model_comparison.png
│
└── hls/                                 # HLS C header files
    ├── weights_float.h                 # Float implementation
    ├── weights_ptq_float.h             # PTQ with float quant/dequant
    ├── weights_ptq_int8.h              # PTQ integer-only
    └── weights_qat.h                   # QAT weights
```

### Renamed Source Files

```
src/
├── train.py                            # Trains all float models
├── train_qat.py                        # QAT training
│
├── export_weights_float.py             # Export float weights
├── export_weights_ptq_float.py         # Export PTQ (float ops) weights
├── export_weights_ptq_int8.py          # Export PTQ (integer-only) weights
├── export_weights_qat.py               # Export QAT weights
│
├── quantization_ptq.py                 # PTQ quantization (renamed from quantization.py)
├── quantization_qat.py                 # QAT quantization
├── prepare_ptq_int8_parameters.py      # Convert PTQ to integer-only
│
└── ...

tests/
├── generate_test_vectors_float.py       # Float model test vectors
├── generate_test_vectors_ptq_float.py   # PTQ with float ops
├── generate_test_vectors_ptq_int8.py    # PTQ integer-only (emulator)
├── generate_test_vectors_qat.py         # QAT test vectors
│
├── evaluate_ptq_float_accuracy.py       # Compare PTQ float vs float model
├── evaluate_ptq_int8_accuracy.py        # Compare PTQ int8 vs PTQ float
├── evaluate_qat_accuracy.py             # Compare QAT vs float model
│
└── ...
```

---

## Model Variants Clarification

### 1. Float Models (No Quantization)
- **Purpose**: Baseline accuracy, training experimentation
- **Use case**: Understanding model capacity
- **Files**: `models/1_base_full.pth`, `models/3_reduced_no_root.pth`

### 2. PTQ with Float Ops (Dequant → Float → Quant)
- **Purpose**: INT8 weights, but uses float arithmetic
- **Implementation**: 
  ```python
  x_float = x_int8 * scale_in              # Dequantize
  agg_float = matmul(adj_float, x_float)   # Float ops
  agg_int8 = round(agg_float / scale_out)  # Quantize
  ```
- **HLS**: Requires floating-point units (DSPs)
- **Files**: `weights/ptq_float/`, `test_vectors/ptq_float/`

### 3. PTQ with Integer-Only Ops (Fixed-Point)
- **Purpose**: Pure integer datapath for HLS
- **Implementation**:
  ```c
  int32_t tmp = 0;
  for (int j = 0; j < N; j++)
      tmp += adj_int16[i][j] * x_int8[j];   // INT16 * INT8
  int64_t scaled = tmp * beta_fp;            // INT32 * INT32
  int8_t result = (scaled + (1<<(M-1))) >> M; // Shift right
  ```
- **HLS**: No floating-point, significant resource savings
- **Parameters**: M=20 (scale fractional bits), K=4096 (adjacency scale)
- **Files**: `weights/ptq_int8/`, `test_vectors/ptq_int8/`
- **Accuracy**: Slightly different from PTQ float due to fixed-point approximation

### 4. QAT (Quantization-Aware Training)
- **Purpose**: Train with fake quantization for better accuracy
- **Implementation**: Simulates quantization during training
- **HLS**: Can use either float ops or integer ops
- **Files**: `weights/qat/`, `test_vectors/qat/`

---

## Refactoring Steps

### Phase 1: Clean Build Directory
1. Backup current build/
2. Create new structure with explicit names
3. Move/rename existing files to new structure

### Phase 2: Rename Source Files
1. Rename quantization.py → quantization_ptq.py
2. Split functionality into explicit export scripts
3. Update imports in all files

### Phase 3: Update Pipeline
1. Rewrite run_pipeline.py with clear steps
2. Each step explicitly named (train_float, export_ptq_float, etc.)
3. Clear output showing which variant is being processed

### Phase 4: Update Documentation
1. Create MODELS.md explaining each variant
2. Create DIRECTORY_STRUCTURE.md
3. Update README.md with new structure

### Phase 5: Validation
1. Run full pipeline
2. Verify all artifacts generated correctly
3. Check HLS compatibility

---

## Implementation Priority

### High Priority (Do First)
- [ ] Backup current build/
- [ ] Create refactoring script to rename/move files
- [ ] Update run_pipeline.py with explicit naming
- [ ] Test pipeline runs without errors

### Medium Priority
- [ ] Rename source files for clarity
- [ ] Update all imports
- [ ] Create MODELS.md documentation

### Low Priority (Nice to Have)
- [ ] Consolidate redundant scripts
- [ ] Add validation checks between steps
- [ ] Create cleanup script to remove old artifacts

---

## Questions for User

1. **Preferred naming convention**: Do you prefer underscores or hyphens?
   - `ptq_float` vs `ptq-float`
   - `reduced_no_root` vs `reduced-no-root`

2. **Keep old structure temporarily?**: Should we keep backups during transition?

3. **Priority on model variants**: Which variants are most important?
   - Float (baseline)
   - PTQ Float (easier HLS, more DSPs)
   - PTQ Int8 (harder HLS, fewer resources)
   - QAT (best accuracy)

4. **HLS target**: Which variant should be the primary HLS target?
   - PTQ Int8 for maximum efficiency?
   - QAT for maximum accuracy?

---

## Suggested Next Steps

1. Review this audit
2. Confirm directory structure and naming
3. Create refactoring script (I can generate this)
4. Run refactoring
5. Test pipeline with new structure
6. Update all documentation
