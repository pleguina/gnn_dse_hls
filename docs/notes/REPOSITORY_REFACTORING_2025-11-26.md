# Repository Refactoring - November 26, 2025

## Overview

Complete repository reorganization to improve clarity and maintainability with explicit naming conventions.

## Changes Made

### 1. Documentation Organization

Created `docs/` directory with subdirectories:

**docs/guides/** - User-facing guides
- `CLEAN_BUILD_GUIDE.md` - How to do a clean build
- `QUICKSTART.md` - Quick start guide
- `VISUALIZATION_GUIDE.md` - Visualization tools guide

**docs/specs/** - Technical specifications
- `QUANTIZATION_SPEC.md` - Quantization technical spec
- `MODELS_AND_IMPLEMENTATIONS.md` - Model architecture details
- `HLS_VARIANTS_AND_TEST_VECTORS.md` - HLS variant documentation
- `INTEGER_PTQ_PIPELINE.md` - Integer-only PTQ pipeline spec

**docs/notes/** - Development notes and historical context
- `CURRENT_STATUS.md` - Project status
- `DIRECTORY_STRUCTURE.md` - Directory structure explanation
- `PTQ_ACCURACY_SUMMARY.md` - PTQ accuracy analysis
- `PTQ_ROUNDING_ANALYSIS.md` - Rounding behavior analysis
- `QAT_TRAINING_NOTES.md` - QAT training notes
- `SAGECONV_ADJACENCY_FIX.md` - Adjacency matrix fix documentation
- `build_instructions.txt` - Build notes
- `next_steps_ptq_hls.txt` - Next steps for PTQ HLS
- `qat_problem_and_fixes.txt` - QAT issues and fixes
- `QAT_scope.txt` - QAT scope definition
- `quantization_solution.txt` - Quantization solution notes
- `quantization.txt` - General quantization notes
- `adjacent_matrix.txt` - Adjacency matrix notes

**docs/deprecated/** - Obsolete documentation
- `CONFIG_MIGRATION.md`
- `CONFIG_MIGRATION_COMPLETE.md`
- `REFACTORING_PLAN.md`
- `TEST_VECTOR_GENERATORS_AUDIT.md`
- `MODEL_VERSIONS.md`

### 2. Source Code Renaming

**Renamed files in src/:**
- `quantization.py` → `quantization_ptq.py` (clearer PTQ purpose)
- `prepare_int8_parameters.py` → `prepare_ptq_int8_parameters.py` (explicit PTQ)

**Moved to src/deprecated/:**
- `pytorch_messagepassing.py` - PyTorch source (reference only)
- `pytorch_SAGEConv.py` - PyTorch source (reference only)
- `pruning.py` - Not currently used

### 3. Test File Renaming

**Renamed for clarity:**
- `debug_hls_python_comparison.py` → `compare_hls_vs_python.py`
- `debug_node6_python.py` → `debug_ptq_node6.py`
- `evaluate_int8_ptq_accuracy.py` → `evaluate_ptq_int8_accuracy.py`
- `python_testbench.py` → `compare_python_vs_hls_testbench.py`
- `generate_ptq_test_vectors_clean.py` → `generate_test_vectors_ptq_float.py`
- `integer_ptq_emulator.py` → `generate_test_vectors_ptq_int8.py`

**Moved to tests/deprecated/:**
- `test_aggregation_methods.py` - Early investigation script
- `trace_sageconv.py` - Early debug script

**Moved to tests/old_generators/:**
- `generate_test_vectors.py` - Wrong approach (uses float model)
- `generate_ptq_test_vectors.py` - Duplicate of clean version

### 4. Build Directory Renaming

**Explicit naming convention: variant_purpose**

**Weight directories:**
- `build/quantized_no_root/` → `build/weights_ptq_float/`
- `build/quantized_int8/` → `build/weights_ptq_int8/`
- `build/quantized_qat/` → `build/weights_qat/`

**Test vector directories:**
- `build/test_vectors/` → `build/test_vectors_old/` (deprecated)
- `build/test_vectors_ptq/` → `build/test_vectors_ptq_float/`
- `build/test_vectors_int8/` → `build/test_vectors_ptq_int8/`

**Model files (build/models/):**
- `base_graphsage_best.pth` → `1_base_full.pth`
- `reduced_graphsage_best.pth` → `2_reduced_with_root.pth`
- `reduced_graphsage_no_root_best.pth` → `3_reduced_no_root.pth`
- `reduced_graphsage_qat_no_root_best.pth` → `4_qat_no_root.pth`

**Training history:**
- Moved to `build/training_history/` with numbered prefixes

### 5. Updated References

**Files with updated paths:**
- `run_pipeline.py` - All script names and output paths
- `src/prepare_ptq_int8_parameters.py` - Input/output directories
- `tests/generate_test_vectors_ptq_float.py` - Output directory and imports
- `tests/generate_test_vectors_ptq_int8.py` - All directory paths
- `tests/generate_all_test_vectors.py` - Documentation and paths
- `tests/evaluate_ptq_int8_accuracy.py` - Input file paths
- `tests/debug_ptq_node6.py` - Import statement
- `src/analyze_models.py` - Import statement
- `verify_both_versions.py` - Directory paths
- `README.md` - Complete project structure

## Naming Conventions

### HLS Variants

Three distinct HLS implementations:

1. **float** - Full precision baseline
   - Files: `*_float.cpp`, `*_float.h`
   - Data: `test_vectors_float/`, `weights_float/`

2. **ptq_float** - PTQ with float quant/dequant operations
   - Files: `graphsage_layer.cpp`, `graphsage_layer.h`
   - Data: `test_vectors_ptq_float/`, `weights_ptq_float/`
   - Uses INT8 data but float operations (dequant → ops → quant)

3. **ptq_int8** - PTQ with integer-only operations
   - Files: (to be created) `graphsage_layer_int8.cpp`, `graphsage_layer_int8.h`
   - Data: `test_vectors_ptq_int8/`, `weights_ptq_int8/`
   - Uses INT8/INT16/INT32 with fixed-point arithmetic

4. **qat** - Quantization-aware training
   - Data: `test_vectors_qat/`, `weights_qat/`
   - Uses QAT-trained model weights

### File Naming Patterns

- **generate_test_vectors_{variant}.py** - Test vector generators
- **weights_{variant}/** - Weight export directories
- **test_vectors_{variant}/** - Test vector directories
- **{number}_{description}.pth** - Model checkpoints (numbered training order)

## Benefits

1. **Clarity** - File names explicitly state their purpose
2. **Consistency** - Uniform naming across all directories
3. **Organization** - Documentation properly categorized
4. **Maintainability** - Deprecated code clearly separated
5. **Discoverability** - Easy to find relevant files

## Migration Notes

- All old paths updated in Python scripts
- Pipeline generates files with new naming from clean build
- Deprecated files preserved for reference but moved out of main workspace
- Documentation structure mirrors code organization

## Next Steps

1. Test full pipeline with new naming
2. Update any remaining documentation references
3. Create HLS integer-only implementation files
4. Remove old deprecated directories after validation
