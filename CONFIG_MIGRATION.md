# Configuration Migration Guide

## Problem: Parameters Hardcoded Everywhere

### Current State (Before Migration)

Model parameters are **hardcoded** in multiple files:

```
src/train.py:155           in_channels_reduced=16, hidden_channels=24
src/train_qat.py:122       in_channels_reduced=16, hidden_channels=24
src/quantization.py:207    in_channels_reduced=16, hidden_channels=24
src/analyze_models.py:169  in_channels_reduced=16, hidden_channels=24
src/pruning.py:226         in_channels_reduced=16, hidden_channels=24
```

**Problems:**
- Changing model size requires editing 6+ files
- Easy to have inconsistencies between Float, PTQ, and QAT models
- No easy way to experiment with different configurations
- Parameters buried in function signatures

## Solution: Centralized Configuration

### New Structure

```
configs/
  └── model_config.yaml      # All parameters in one place

src/
  └── config.py              # Configuration loader
```

### How to Use the New Config

#### 1. View Current Configuration

```bash
cd src
python config.py
```

Output:
```
============================================================
Configuration Summary
============================================================

Dataset: Cora
  Features: 1433
  Classes: 7

Reduced Model:
  Architecture: 16 → 24 → 7
  Root weight: False
  Epochs: 200, LR: 0.01

QAT Model (active: qat_model):
  Architecture: 16 → 24 → 7
  Quantization: INT8 weights, INT8 acts
  Epochs: 200, LR: 0.01
  Calibration batches: 50
============================================================
```

#### 2. Use Config in Python Scripts

**Old way** (hardcoded):
```python
# train_qat.py - OLD
def train_qat_model(epochs=200, lr=0.01, in_channels_reduced=16,
                   hidden_channels=24, dropout=0.5, root_weight=False):
    ...
```

**New way** (from config):
```python
# train_qat.py - NEW
from config import get_config

config = get_config()

model = ReducedGraphSAGEQAT(
    in_channels=config.num_features,
    in_channels_reduced=config.qat_in_channels,
    hidden_channels=config.qat_hidden_channels,
    out_channels=config.num_classes,
    dropout=config.qat_dropout,
    root_weight=config.qat_root_weight,
)

optimizer = torch.optim.Adam(model.parameters(),
                              lr=config.qat_lr,
                              weight_decay=config.get('qat_model.weight_decay'))
```

#### 3. Change Model Size for Better QAT

**Option A: Edit config file directly**
```bash
# Edit configs/model_config.yaml
nano configs/model_config.yaml

# Change:
active_qat_config: "qat_model_large"  # Instead of "qat_model"
```

**Option B: Use larger predefined config**
The config file has two QAT configurations:
- `qat_model`: 16 → 24 → 7 (current, low accuracy)
- `qat_model_large`: 32 → 48 → 7 (better accuracy)

Just change the `active_qat_config` line!

#### 4. Create Custom Configuration

Add to `configs/model_config.yaml`:

```yaml
# Custom QAT Configuration
qat_model_custom:
  in_channels_reduced: 24
  hidden_channels: 36
  out_channels: 7
  dropout: 0.5
  use_projection: true
  root_weight: false

  num_bits_acts: 8
  num_bits_weights: 8

  epochs: 200
  lr: 0.005
  weight_decay: 5e-4

  calibration_batches: 75

# Then activate it:
active_qat_config: "qat_model_custom"
```

## Configuration Options

### Available Parameters

```yaml
dataset:
  name: "Cora"
  num_features: 1433
  num_classes: 7

reduced_model:
  in_channels_reduced: 16   # Input projection size
  hidden_channels: 24       # Hidden layer size
  dropout: 0.5
  root_weight: false        # HLS-compatible
  epochs: 200
  lr: 0.01

qat_model:
  in_channels_reduced: 16   # Should match reduced_model
  hidden_channels: 24
  num_bits_acts: 8          # Activation bits
  num_bits_weights: 8       # Weight bits
  epochs: 200
  lr: 0.01
  calibration_batches: 50

qat_model_large:
  in_channels_reduced: 32   # Larger for better accuracy
  hidden_channels: 48
  lr: 0.005                 # Lower LR for stability
  calibration_batches: 100  # More calibration
```

## Migration Status

### ✅ Done
- [x] Created `configs/model_config.yaml`
- [x] Created `src/config.py` loader
- [x] Installed PyYAML
- [x] Tested configuration loading

### ⏳ TODO: Update Scripts

Files that need migration to use config:

1. **src/train_qat.py** - QAT training
2. **src/quantization_qat.py** - QAT export
3. **src/train.py** - Float model training
4. **src/quantization.py** - PTQ
5. **src/analyze_models.py** - Analysis
6. **src/pruning.py** - Pruning

## Quick Start After Migration

### Run QAT with Small Model (16→24→7):
```bash
# Check current config
cd src
python config.py

# Train QAT
python train_qat.py  # Uses config automatically
```

### Run QAT with Large Model (32→48→7):
```bash
# Edit configs/model_config.yaml:
# Change: active_qat_config: "qat_model_large"

# Train QAT
cd src
python train_qat.py  # Now uses larger model!
```

### Compare Configurations:
```bash
# Small model
active_qat_config: "qat_model"
# Expected accuracy: 50% (current issue)

# Large model
active_qat_config: "qat_model_large"
# Expected accuracy: 65-70% (much better!)
```

## Benefits

1. **Single Source of Truth**: All parameters in one file
2. **Easy Experimentation**: Change config, not code
3. **Consistency**: All models use same config
4. **Documentation**: Config file documents all parameters
5. **Version Control**: Easy to track parameter changes

## Next Steps

Would you like me to:
1. ✅ Keep current 16→24→7 configuration
2. 🔄 Switch to 32→48→7 for better QAT accuracy
3. 🎯 Migrate all scripts to use config (recommended)
4. 🧪 Create experiment configs for different sizes
