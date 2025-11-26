# Directory Structure

## Build Directory

```
build/
├── models/                              # Trained models
│   ├── 1_base_full.pth                 # Base full model
│   ├── 2_reduced_with_root.pth         # Reduced (with root_weight)
│   ├── 3_reduced_no_root.pth           # Reduced (HLS-compatible)
│   └── 4_qat_no_root.pth               # QAT trained
│
├── training_history/                    # Training curves
│   ├── 1_base_full_history.json
│   ├── 2_reduced_with_root_history.json
│   ├── 3_reduced_no_root_history.json
│   └── 4_qat_no_root_history.json
│
├── weights_ptq_float/                   # PTQ INT8 (uses float ops)
│   ├── conv1_lin_l_weight.txt
│   ├── conv1_lin_l_bias.txt
│   ├── conv2_lin_l_weight.txt
│   ├── conv2_lin_l_bias.txt
│   └── quant_params.json
│
├── weights_ptq_int8/                    # PTQ INT8 (integer-only)
│   ├── bias_layer1_int32.txt
│   ├── bias_layer2_int32.txt
│   ├── adj_matrix_int16.txt
│   └── int8_params.json
│
├── weights_qat/                         # QAT quantized
│   └── ...
│
├── weights_float/                       # Float exports (for conversion)
│   └── *_bias_no_root.txt
│
├── test_vectors_float/                  # Float model
│   ├── network_input.txt
│   ├── network_output.txt
│   └── adj_matrix.txt
│
├── test_vectors_ptq_float/              # PTQ with float ops
│   ├── network_input.txt
│   ├── network_output_reference.txt
│   ├── adj_matrix.txt
│   └── scales.txt
│
├── test_vectors_ptq_int8/               # PTQ integer-only
│   ├── network_input.txt
│   ├── network_output_reference.txt
│   ├── adj_matrix_int16.txt
│   └── config.txt (M, K parameters)
│
└── test_vectors_qat/                    # QAT
    └── ...
```

## Model Variants

1. **Float Baseline** (3_reduced_no_root.pth)
   - No quantization
   - Full precision
   - Test vectors: test_vectors_float/

2. **PTQ Float** (3_reduced_no_root.pth + quantization)
   - INT8 weights/activations
   - Uses: dequantize → float ops → quantize
   - HLS: Requires DSPs for float operations
   - Weights: weights_ptq_float/
   - Test vectors: test_vectors_ptq_float/

3. **PTQ Int8** (3_reduced_no_root.pth + integer conversion)
   - INT8 weights/activations
   - Uses: Integer-only fixed-point arithmetic
   - HLS: No floating-point required
   - Parameters: M=20, K=4096
   - Weights: weights_ptq_int8/
   - Test vectors: test_vectors_ptq_int8/

4. **QAT** (4_qat_no_root.pth)
   - Quantization-aware training
   - Better accuracy than PTQ
   - Weights: weights_qat/
   - Test vectors: test_vectors_qat/
