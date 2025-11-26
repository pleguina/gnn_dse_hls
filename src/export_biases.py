"""
Quick script to export float biases from existing trained model checkpoint.
Only exports biases - weights already have PTQ INT8 versions we can reuse.
"""

import torch
import numpy as np
from pathlib import Path

# Create output directory
output_dir = Path("../build/weights_float")
output_dir.mkdir(parents=True, exist_ok=True)

# Load trained model checkpoint
checkpoint_path = Path("../build/models/reduced_graphsage_no_root_best.pth")

print("="*80)
print("EXPORT FLOAT BIASES FROM TRAINED MODEL")
print("="*80)
print(f"\nLoading checkpoint: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = checkpoint['model_state_dict']

print(f"\nModel parameters:")
for name, param in state_dict.items():
    print(f"  {name}: {param.shape}")

# Extract and save biases
biases = {
    'conv1_lin_l_bias_no_root.txt': state_dict['conv1.lin_l.bias'],
    'conv2_lin_l_bias_no_root.txt': state_dict['conv2.lin_l.bias']
}

print(f"\nExporting biases to {output_dir}/")
for filename, param in biases.items():
    # Convert to numpy
    bias_np = param.cpu().numpy()
    
    # Save as text file
    np.savetxt(output_dir / filename, bias_np.flatten(), fmt='%.6f')
    
    # Save shape metadata
    with open(output_dir / f"{filename}.shape", 'w') as f:
        if len(bias_np.shape) == 1:
            f.write(str(bias_np.shape[0]))
        else:
            f.write(','.join(map(str, bias_np.shape)))
    
    print(f"  ✓ {filename}: shape={bias_np.shape}, range=[{bias_np.min():.6f}, {bias_np.max():.6f}]")

print(f"\n" + "="*80)
print("SUCCESS - Float biases exported!")
print("="*80)
print(f"\nNext step: Run prepare_int8_parameters.py to convert to INT32")
print(f"  cd /home/pelayo/work/simple-gnn/src")
print(f"  python3 prepare_int8_parameters.py")
