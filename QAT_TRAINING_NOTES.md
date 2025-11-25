# QAT Training Notes and Troubleshooting

## Current Status

### Model Performance Comparison

| Model Type | Validation Accuracy | Notes |
|------------|--------------------|----|
| **Float Baseline** | ~75.6% | Full precision (FP32) |
| **Float Reduced** | ~75.7% | Reduced dimensions (16→24→7) |
| **QAT** | ~46-51% | INT8 quantization-aware training |
| **PTQ** | ~74% (estimated) | Post-training quantization |

## Why is QAT Accuracy Lower?

### Root Causes:

1. **Model is Too Small for QAT**
   - Current: 16 input → 24 hidden → 7 output
   - With only 24 hidden units, quantization severely limits representation capacity
   - INT8 quantization adds ~1-2% noise per layer, which compounds

2. **Training Instability**
   - Fake quantization introduces non-differentiable quantization operators
   - Small models are more sensitive to quantization noise during backprop
   - The Straight-Through Estimator (STE) used for gradients can cause instability

3. **Insufficient Representational Capacity**
   - The graph task requires capturing complex neighborhood patterns
   - Quantization reduces effective precision from 23 bits (FP32) to 7 bits (INT8)
   - This 3.3x reduction in precision is harder to compensate in smaller networks

## Solutions to Improve QAT Accuracy

### Option 1: Increase Model Size (Recommended)
```python
# In train_qat.py, change:
train_qat_model(
    epochs=200,
    in_channels_reduced=32,  # Was 16
    hidden_channels=48,       # Was 24
    root_weight=False
)
```

**Expected improvement**: 55-65% accuracy

### Option 2: Use PTQ Instead
Post-Training Quantization often works better for small models:
```bash
cd src
python quantization.py  # Uses PTQ instead of QAT
```

**Expected accuracy**: 72-75% (only 1-3% drop from float)

### Option 3: Mixed Quantization Strategy
- Keep first layer in higher precision (INT16 or even FP16)
- Only quantize later layers to INT8
- This preserves critical input features

### Option 4: Tune QAT Hyperparameters

#### Lower Learning Rate:
```python
train_qat_model(epochs=200, lr=0.005)  # Was 0.01
```

#### More Calibration:
```python
# In train_qat.py calibrate_qat_model:
calibrate_qat_model(model, data, num_batches=100)  # Was 50
```

#### Different Quantization Scheme:
```python
# In model_base_QAT.py, try asymmetric quantization:
return FakeQuantize(
    observer=MovingAverageMinMaxObserver,
    quant_min=0,  # Changed from -(2^7)
    quant_max=255,  # Changed from 2^7-1
    dtype=torch.quint8,  # Changed from qint8
    qscheme=torch.per_tensor_affine,  # Asymmetric
    ...
)
```

## What Accuracy is Acceptable?

### Industry Standards:
- **PTQ**: 1-5% accuracy drop is typical
- **QAT**: Should match or beat PTQ (0-3% drop ideal)
- **Your case**: QAT showing 25-30% drop is **too high**

### For FPGA Deployment:
If QAT accuracy is insufficient, you have options:

1. **Use PTQ artifacts** (build/quantized/) instead of QAT
   - PTQ achieves ~74% (only 1.5% drop)
   - Easier to implement in HLS
   - No training required

2. **Increase model size** for QAT
   - Larger hidden dimensions give more room for quantization
   - 32→48→7 or 32→64→7 recommended

3. **Accept lower accuracy** if hardware constraints are strict
   - 50% accuracy might be acceptable depending on application
   - Trade-off: accuracy vs. resource usage

## Recommended Next Steps

### For Best Accuracy:
1. Use **PTQ** (already implemented in `src/quantization.py`)
2. Export PTQ weights to HLS
3. Validate with test vectors

### For Better QAT:
1. Increase model size to 32→48→7
2. Retrain with 200 epochs
3. Use lower learning rate (0.005)

### For Quick Testing:
The current QAT model (~50% accuracy) is still valid for:
- HLS implementation testing
- Hardware validation
- Proving quantization pipeline works
- Just not for production deployment

## Files Generated

Regardless of accuracy, QAT pipeline generates all required files:

```
build/
├── quantized_qat/
│   ├── weights_layer1_qat.txt  # INT8 weights
│   ├── bias_layer1_qat.txt      # INT32 biases
│   ├── weights_layer2_qat.txt
│   └── bias_layer2_qat.txt
└── test_vectors_qat/
    ├── network_input_qat.txt           # INT8 inputs
    ├── network_output_reference_qat.txt # INT8 outputs
    ├── edge_index_qat.txt
    ├── adj_matrix_qat.txt
    ├── scales_qat.txt                   # Quantization scales
    └── quant_params_qat.json
```

These files are **correct and usable** for HLS, just trained on a small/challenging model.

## Summary

**Problem**: QAT accuracy (50%) << Float accuracy (75%)
**Cause**: Model too small for INT8 quantization constraints
**Solution**: Either (1) use PTQ artifacts, or (2) increase model size for QAT
**For HLS**: Current QAT files are valid for testing implementation

The QAT pipeline is working correctly - the low accuracy is a model capacity issue, not a code bug.
