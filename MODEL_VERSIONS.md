# GraphSAGE Model Versions

This project provides two versions of the reduced GraphSAGE model to support different use cases.

## Version 1: With root_weight=True (Default PyG)

**Files:**
- Model: `build/models/reduced_graphsage_best.pth`
- Quantized: `build/quantized/` 
- Weights header: `build/hls/weights.h`
- Training plot: `build/plots/reduced_model_training.png`

**Formula:**
```
out = W_l * h_agg + b_l + W_r * x_i
```

**Characteristics:**
- Uses both `lin_l` and `lin_r` weights in SAGEConv
- Default PyTorch Geometric behavior
- Better accuracy (typically ~1-2% higher)
- More parameters and computation
- **Not directly compatible with current HLS implementation**

**When to use:**
- For maximum accuracy in Python/PyTorch
- For comparison and baseline measurements
- When HLS constraints don't apply

## Version 2: Without root_weight (HLS-Compatible) ✅

**Files:**
- Model: `build/models/reduced_graphsage_no_root_best.pth`
- Quantized: `build/quantized_no_root/`
- Weights header: `build/hls/weights_no_root.h`
- Training plot: `build/plots/reduced_model_no_root_training.png`

**Formula:**
```
out = W_l * h_agg + b_l
```

**Characteristics:**
- Uses only `lin_l` weights (no `lin_r`)
- Simpler computation
- Directly maps to HLS implementation
- Slightly lower accuracy (~1-2% drop)
- Fewer parameters
- **Use this for HLS/FPGA implementation**

**When to use:**
- **For HLS synthesis and FPGA deployment** ✅
- When hardware resource constraints matter
- For simpler, more predictable hardware mapping

## Quantization Details

Both versions use INT8 quantization with the following scheme:

### Activation Scales
- `scale_in`: Input feature scale
- `scale_hidden`: Hidden activation scale  
- `scale_out`: Output activation scale

### Weight Scales
- `scale_w1`: Conv1 weights scale (layer 1)
- `scale_w2`: Conv2 weights scale (layer 2)

### Formula (per quantization.txt)
For a linear layer `y = W * x + b`:

1. **Integer accumulation:**
   ```
   acc = sum(x_int8 * W_int8) + b_int32
   ```

2. **Real value reconstruction:**
   ```
   y_real = acc * (scale_x * scale_w)
   ```

3. **Requantization to next layer:**
   ```
   y_int8 = quantize(y_real, scale_y)
   ```

## Training Both Versions

The training script automatically trains both versions:

```bash
cd src
python3 train.py
```

This will generate both model variants.

## Quantizing Both Versions

### HLS-compatible (no root_weight):
```bash
cd src
python3 quantization.py
```

### With root_weight:
```bash
cd src
python3 quantization.py --use-root-weight
```

## Run Full Pipeline

The pipeline script trains and quantizes both versions:

```bash
python3 run_pipeline.py
```

## Recommendation for HLS

**Use the `_no_root` version for HLS implementation:**

1. Simpler hardware mapping
2. Matches the formula in `quantization.txt`
3. Direct correspondence to HLS `graphsage_network` implementation
4. No additional `lin_r` computation needed

The accuracy difference is minimal (~1-2%) but the hardware implementation is significantly simpler and more efficient.

## Verification

To verify which version you're using:

```python
import torch

# Load checkpoint
checkpoint = torch.load('build/models/reduced_graphsage_no_root_best.pth')

# Check if root_weight is stored
if 'root_weight' in checkpoint:
    print(f"Model uses root_weight = {checkpoint['root_weight']}")
else:
    print("Model is older version (likely has root_weight=True)")
```

Or check the quantized parameters:

```bash
# HLS-compatible version should NOT have lin_r weights
cat build/quantized_no_root/quant_params.json | grep lin_r
# Should return nothing

# Standard version WILL have lin_r weights  
cat build/quantized/quant_params.json | grep lin_r
# Should show conv1.lin_r and conv2.lin_r
```
