#!/usr/bin/env python3
"""
Debug script to print all intermediate values for node 6 in Python PTQ
"""

import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_base import ReducedGraphSAGE
from quantization import quantize_tensor
from subgraph_extraction import extract_fixed_subgraph
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

# Load everything
dataset = Planetoid(root='../data', name='Cora', transform=NormalizeFeatures())
data = dataset[0]
subgraph_data = extract_fixed_subgraph(data, num_nodes=8, center_node=0, num_hops=2)

model = ReducedGraphSAGE(
    in_channels=dataset.num_features,
    in_channels_reduced=16,
    hidden_channels=24,
    out_channels=dataset.num_classes,
    dropout=0.5,
    use_projection=True,
    root_weight=False
)

checkpoint = torch.load('../build/models/reduced_graphsage_no_root_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get features after projection
adj_matrix = subgraph_data['adj_matrix']
features = subgraph_data['x']

with torch.no_grad():
    features_tensor = model.projection(torch.from_numpy(features).float())
    features_tensor = torch.relu(features_tensor)

features_quant, scale_in, _ = quantize_tensor(features_tensor, num_bits=8)

# Get weights
conv1 = model.conv1
weights1 = conv1.lin_l.weight.data
bias1 = conv1.lin_l.bias.data
weights1_quant, scale_w1, _ = quantize_tensor(weights1, num_bits=8)
bias1_quant = (bias1 / (scale_in * scale_w1)).to(torch.int32)

conv2 = model.conv2
weights2 = conv2.lin_l.weight.data
bias2 = conv2.lin_l.bias.data
weights2_quant, scale_w2, _ = quantize_tensor(weights2, num_bits=8)
scale_hidden = 0.1
bias2_quant = (bias2 / (scale_hidden * scale_w2)).to(torch.int32)
scale_out = 0.1

NODE = 6

print("\n" + "="*80)
print(f"PYTHON PTQ - NODE {NODE} - DETAILED DEBUG")
print("="*80)

print(f"\nScales:")
print(f"  scale_in:     {scale_in}")
print(f"  scale_w1:     {scale_w1}")
print(f"  scale_hidden: {scale_hidden}")
print(f"  scale_w2:     {scale_w2}")
print(f"  scale_out:    {scale_out}")

# === LAYER 1 ===
print("\n" + "-"*80)
print("LAYER 1 - AGGREGATION")
print("-"*80)

# Input quantized
print(f"Input_int8[{NODE},:] = {features_quant[NODE,:].tolist()}")

# Dequantize
x_float = features_quant.float() * scale_in
print(f"Input_float[{NODE},:3] = {x_float[NODE,:3].tolist()}")

# Aggregate
adj_tensor = torch.from_numpy(adj_matrix).float()
print(f"Adjacency[{NODE},:] = {adj_tensor[NODE,:].tolist()}")

agg_float = torch.matmul(adj_tensor, x_float)
print(f"Agg1_float[{NODE},:3] = {agg_float[NODE,:3].tolist()}")

# Quantize to hidden scale
agg_before_round = agg_float / scale_hidden
print(f"Agg1_before_round[{NODE},:3] = {agg_before_round[NODE,:3].tolist()}")

agg1_int8 = torch.round(agg_before_round).clamp(-128, 127).to(torch.int8)
print(f"Agg1_int8[{NODE},:] = {agg1_int8[NODE,:].tolist()}")

print("\n" + "-"*80)
print("LAYER 1 - LINEAR")
print("-"*80)

# Matrix multiply in int32
acc1_int32 = torch.mm(agg1_int8.to(torch.int32), weights1_quant.t().to(torch.int32)) + bias1_quant
print(f"Acc1_int32[{NODE},:3] = {acc1_int32[NODE,:3].tolist()}")

# Requantize
scale_factor_l1 = (scale_hidden * scale_w1) / scale_hidden
print(f"Scale_factor_L1 = {scale_factor_l1}")

hidden_before_round = acc1_int32.float() * scale_factor_l1
print(f"Hidden_before_round[{NODE},:3] = {hidden_before_round[NODE,:3].tolist()}")

hidden_int8 = torch.round(hidden_before_round).clamp(-128, 127).to(torch.int8)
print(f"Hidden_before_relu[{NODE},:3] = {hidden_int8[NODE,:3].tolist()}")

# ReLU
hidden_int8 = torch.clamp(hidden_int8, min=0)
print(f"Hidden_int8[{NODE},:] = {hidden_int8[NODE,:].tolist()}")

# === LAYER 2 ===
print("\n" + "-"*80)
print("LAYER 2 - AGGREGATION")
print("-"*80)

# Dequantize hidden
hidden_float = hidden_int8.float() * scale_hidden
print(f"Hidden_float[{NODE},:3] = {hidden_float[NODE,:3].tolist()}")

# Aggregate
agg2_float = torch.matmul(adj_tensor, hidden_float)
print(f"Agg2_float[{NODE},:3] = {agg2_float[NODE,:3].tolist()}")
print(f"Agg2_float[{NODE},[8,13,16,19]] = {agg2_float[NODE,[8,13,16,19]].tolist()}")

# Quantize
agg2_before_round = agg2_float / scale_out
print(f"Agg2_before_round[{NODE},:3] = {agg2_before_round[NODE,:3].tolist()}")
print(f"Agg2_before_round[{NODE},[8,13,16,19]] = {agg2_before_round[NODE,[8,13,16,19]].tolist()}")

agg2_int8 = torch.round(agg2_before_round).clamp(-128, 127).to(torch.int8)
print(f"Agg2_int8[{NODE},:] = {agg2_int8[NODE,:].tolist()}")

print("\n" + "-"*80)
print("LAYER 2 - LINEAR")
print("-"*80)

# Matrix multiply in int32
acc2_int32 = torch.mm(agg2_int8.to(torch.int32), weights2_quant.t().to(torch.int32)) + bias2_quant
print(f"Acc2_int32[{NODE},:] = {acc2_int32[NODE,:].tolist()}")

# Requantize
scale_factor_l2 = (scale_out * scale_w2) / scale_out
print(f"Scale_factor_L2 = {scale_factor_l2}")

output_before_round = acc2_int32.float() * scale_factor_l2
print(f"Output_before_round[{NODE},:] = {output_before_round[NODE,:].tolist()}")

output_int8 = torch.round(output_before_round).clamp(-128, 127).to(torch.int8)
print(f"Output_int8[{NODE},:] = {output_int8[NODE,:].tolist()}")

print("\n" + "="*80)
print("FINAL RESULT")
print("="*80)
print(f"Python output[{NODE},:] = {output_int8[NODE,:].tolist()}")
print(f"Expected from file:      [-36, -7, -105, -29, -114, 120, -128]")
