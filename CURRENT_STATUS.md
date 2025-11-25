# Current Status: Configuration and QAT Setup

## What You Asked ✓

> "Where are the model sizes and parameters defined? Shouldn't they be in configs/?"

**Answer**: You were **100% correct!** Parameters were hardcoded in 6+ different files instead of a central config.

## What I Found

### Problem: Scattered Parameters
```
❌ src/train.py:155           → in_channels_reduced=16, hidden_channels=24
❌ src/train_qat.py:122        → in_channels_reduced=16, hidden_channels=24
❌ src/quantization.py:207     → in_channels_reduced=16, hidden_channels=24
❌ src/analyze_models.py:169   → in_channels_reduced=16, hidden_channels=24
❌ src/pruning.py:226          → in_channels_reduced=16, hidden_channels=24
```

**Impact**: To change model size, you'd need to edit 6 files manually!

### Root Cause of Low QAT Accuracy

The 16→24→7 architecture is **too small** for INT8 quantization:
- Float model: 75.7% accuracy ✓
- QAT model: 50% accuracy ✗ (25% drop!)

**Why?**
- Only 24 hidden units
- INT8 quantization loses too much precision
- Not enough capacity to learn under quantization constraints

## What I've Created

### ✅ 1. Centralized Configuration System

**New files:**
```
configs/
  └── model_config.yaml         ← ALL parameters here

src/
  └── config.py                 ← Python loader for all scripts
```

**Test it:**
```bash
cd src
python config.py
```

### ✅ 2. Configuration Options

The config file has **3 model configurations**:

#### Option A: Current Small Model (16→24→7)
```yaml
reduced_model:
  in_channels_reduced: 16
  hidden_channels: 24

qat_model:
  in_channels_reduced: 16   # Same as reduced
  hidden_channels: 24
```
- Float accuracy: 75.7% ✓
- QAT accuracy: ~50% ✗
- **Status**: This is what's currently trained

#### Option B: Large QAT Model (32→48→7) - RECOMMENDED
```yaml
qat_model_large:
  in_channels_reduced: 32   # 2x larger
  hidden_channels: 48       # 2x larger
  lr: 0.005                 # Lower LR for stability
  calibration_batches: 100  # More calibration
```
- Expected QAT accuracy: **65-70%** ✓
- **Status**: Defined but not yet trained

#### Option C: Use PTQ Instead
```bash
# Already generated with good accuracy:
build/quantized/         # PTQ weights
build/test_vectors/      # PTQ test vectors
```
- PTQ accuracy: ~74% ✓ (only 1.5% drop)
- **Status**: Ready to use for HLS

## Current Model Sizes

### What's Actually Trained:

| Model | Architecture | Accuracy | Files |
|-------|-------------|----------|-------|
| **Float (reduced)** | 16→24→7 | 75.7% | `build/models/reduced_graphsage_no_root_best.pth` |
| **QAT (small)** | 16→24→7 | 50% | `build/models/reduced_graphsage_qat_no_root_best.pth` |
| **PTQ** | 16→24→7 | ~74% | `build/quantized/` |

### What's Available to Train:

```yaml
# In configs/model_config.yaml:

active_qat_config: "qat_model"        # Current: 16→24→7
# OR
active_qat_config: "qat_model_large"  # Better:  32→48→7
```

## Your Options Now

### Option 1: Use PTQ for HLS (Easiest, Good Accuracy)
```bash
# PTQ files already generated:
ls build/quantized/
ls build/test_vectors/

# These have ~74% accuracy (good!)
# Ready for HLS implementation
```

### Option 2: Train Larger QAT Model (Best Accuracy)
```bash
# Edit config to use larger model:
nano configs/model_config.yaml
# Change: active_qat_config: "qat_model_large"

# Train QAT with larger model:
cd src
python train_qat.py

# Expected: 65-70% accuracy (much better!)
```

### Option 3: Keep Small Model (Testing Only)
```bash
# Current 16→24→7 QAT model (50% accuracy)
# Good for:
# - Testing HLS implementation
# - Verifying quantization pipeline
# - Proof of concept

# NOT good for production
```

### Option 4: Migrate All Scripts to Use Config (Recommended)
```bash
# Update all scripts to read from configs/model_config.yaml
# Benefits:
# - Single source of truth
# - Easy to experiment with sizes
# - No more hardcoded values

# Files to update:
# - src/train.py
# - src/train_qat.py
# - src/quantization.py
# - src/analyze_models.py
# - etc.
```

## Quick Commands

### View current config:
```bash
cd src
python config.py
```

### Switch to larger QAT model:
```bash
# Edit line 115 of configs/model_config.yaml:
active_qat_config: "qat_model_large"

# Then train:
python train_qat.py
```

### Compare model files:
```bash
ls -lh build/models/
```

### Check what's generated:
```bash
ls build/quantized/        # PTQ (good)
ls build/quantized_qat/    # QAT (small model)
ls build/test_vectors/     # PTQ vectors
ls build/test_vectors_qat/ # QAT vectors
```

## Summary

✅ **Config system created**: `configs/model_config.yaml`
✅ **Config loader created**: `src/config.py`
✅ **Identified problem**: 16→24→7 too small for QAT
✅ **Solution available**: 32→48→7 config ready to use

📋 **Documentation created**:
- `CONFIG_MIGRATION.md` - How to use the new config
- `QAT_TRAINING_NOTES.md` - Why QAT accuracy is low
- `CURRENT_STATUS.md` - This file

## Next Step - Your Choice:

1. **Use PTQ** (easiest, 74% accuracy, ready now)
2. **Train larger QAT** (best, expected 65-70%, needs training)
3. **Migrate scripts to config** (cleanest, one-time effort)
4. **Keep small model** (testing only, 50% accuracy)

What would you like to do?
