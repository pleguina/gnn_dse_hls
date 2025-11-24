"""
Structured pruning utilities for GNN models.
Implements channel-wise pruning for FPGA-friendly architectures.
"""

import torch
import torch.nn as nn
import numpy as np
from model_base import ReducedGraphSAGE


def compute_channel_importance(layer, method='l1'):
    """
    Compute importance scores for output channels of a layer.

    Args:
        layer: Linear or Conv layer
        method: Importance metric ('l1', 'l2', or 'variance')

    Returns:
        importance: Importance score for each output channel
    """
    weight = layer.weight.data  # Shape: [out_channels, in_channels]

    if method == 'l1':
        # L1 norm of weights for each output channel
        importance = torch.norm(weight, p=1, dim=1)
    elif method == 'l2':
        # L2 norm of weights for each output channel
        importance = torch.norm(weight, p=2, dim=1)
    elif method == 'variance':
        # Variance of weights for each output channel
        importance = torch.var(weight, dim=1)
    else:
        raise ValueError(f"Unknown method: {method}")

    return importance


def prune_channels(model, layer_name, keep_ratio=0.75, method='l1'):
    """
    Prune channels from a specific layer.

    Args:
        model: PyTorch model
        layer_name: Name of layer to prune
        keep_ratio: Ratio of channels to keep (0.75 = keep 75%, prune 25%)
        method: Importance metric

    Returns:
        pruned_indices: Indices of channels that were kept
    """
    # Get the layer
    layer = dict(model.named_modules())[layer_name]

    # Compute importance
    importance = compute_channel_importance(layer, method=method)

    # Determine how many channels to keep
    num_channels = importance.shape[0]
    num_keep = int(num_channels * keep_ratio)

    # Get indices of most important channels
    _, indices = torch.topk(importance, num_keep, largest=True)
    pruned_indices = indices.sort()[0]  # Sort for stability

    return pruned_indices


def apply_structured_pruning(model, pruning_config):
    """
    Apply structured pruning to model based on configuration.

    Args:
        model: PyTorch model
        pruning_config: Dictionary specifying pruning for each layer
                       e.g., {'conv1': 0.75, 'conv2': 0.75}

    Returns:
        pruned_model: New model with pruned architecture
        pruning_masks: Dictionary of pruning masks for each layer
    """
    pruning_masks = {}

    for layer_name, keep_ratio in pruning_config.items():
        pruned_indices = prune_channels(model, layer_name, keep_ratio)
        pruning_masks[layer_name] = pruned_indices
        print(f"Pruned {layer_name}: keeping {len(pruned_indices)} channels")

    return pruning_masks


def create_pruned_model(original_model, pruning_masks):
    """
    Create a new model with pruned architecture.

    Args:
        original_model: Original ReducedGraphSAGE model
        pruning_masks: Dictionary of channel indices to keep

    Returns:
        pruned_model: New model with reduced channel dimensions
    """
    # For simplicity, we'll create a new model with adjusted dimensions
    # In practice, you'd need to handle the specific layer connections

    # Get dimensions from pruning masks
    if 'conv1' in pruning_masks:
        hidden_channels = len(pruning_masks['conv1'])
    else:
        hidden_channels = original_model.conv1.out_channels

    if 'conv2' in pruning_masks:
        out_channels = len(pruning_masks['conv2'])
    else:
        out_channels = original_model.conv2.out_channels

    # Create new model with pruned dimensions
    pruned_model = ReducedGraphSAGE(
        in_channels=original_model.projection.in_features if hasattr(original_model, 'projection') else original_model.conv1.in_channels,
        in_channels_reduced=original_model.projection.out_features if hasattr(original_model, 'projection') else original_model.conv1.in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        dropout=original_model.dropout,
        use_projection=original_model.use_projection
    )

    return pruned_model


def copy_pruned_weights(pruned_model, original_model, pruning_masks):
    """
    Copy weights from original model to pruned model according to pruning masks.

    Args:
        pruned_model: New model with reduced dimensions
        original_model: Original trained model
        pruning_masks: Dictionary of channel indices to keep
    """
    pruned_model.eval()
    original_model.eval()

    with torch.no_grad():
        # Copy projection layer if it exists
        if hasattr(pruned_model, 'projection') and hasattr(original_model, 'projection'):
            pruned_model.projection.weight.data = original_model.projection.weight.data.clone()
            if original_model.projection.bias is not None:
                pruned_model.projection.bias.data = original_model.projection.bias.data.clone()

        # Copy conv1 weights
        if 'conv1' in pruning_masks:
            indices = pruning_masks['conv1']
            # Output channels pruning
            pruned_model.conv1.lin_l.weight.data = original_model.conv1.lin_l.weight.data[indices].clone()
            pruned_model.conv1.lin_r.weight.data = original_model.conv1.lin_r.weight.data[indices].clone()
            if original_model.conv1.lin_l.bias is not None:
                pruned_model.conv1.lin_l.bias.data = original_model.conv1.lin_l.bias.data[indices].clone()
                pruned_model.conv1.lin_r.bias.data = original_model.conv1.lin_r.bias.data[indices].clone()

        # Copy conv2 weights (with input channel pruning based on conv1 output)
        if 'conv1' in pruning_masks:
            in_indices = pruning_masks['conv1']
            pruned_model.conv2.lin_l.weight.data = original_model.conv2.lin_l.weight.data[:, in_indices].clone()
            pruned_model.conv2.lin_r.weight.data = original_model.conv2.lin_r.weight.data[:, in_indices].clone()

            if 'conv2' in pruning_masks:
                out_indices = pruning_masks['conv2']
                pruned_model.conv2.lin_l.weight.data = pruned_model.conv2.lin_l.weight.data[out_indices].clone()
                pruned_model.conv2.lin_r.weight.data = pruned_model.conv2.lin_r.weight.data[out_indices].clone()
                if original_model.conv2.lin_l.bias is not None:
                    pruned_model.conv2.lin_l.bias.data = original_model.conv2.lin_l.bias.data[out_indices].clone()
                    pruned_model.conv2.lin_r.bias.data = original_model.conv2.lin_r.bias.data[out_indices].clone()

    print("Weights copied to pruned model")


def fine_tune_pruned_model(model, data, epochs=50, lr=0.001):
    """
    Fine-tune pruned model to recover accuracy.

    Args:
        model: Pruned model
        data: Training data
        epochs: Number of fine-tuning epochs
        lr: Learning rate

    Returns:
        model: Fine-tuned model
    """
    import torch.nn.functional as F

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                pred = out.argmax(dim=1)
                val_correct = pred[data.val_mask] == data.y[data.val_mask]
                val_acc = int(val_correct.sum()) / int(data.val_mask.sum())
            model.train()
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}')

    return model


if __name__ == '__main__':
    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures

    # Load dataset
    dataset = Planetoid(root='./data', name='Cora', transform=NormalizeFeatures())
    data = dataset[0]

    # Load trained model
    model = ReducedGraphSAGE(
        in_channels=dataset.num_features,
        in_channels_reduced=16,
        hidden_channels=24,
        out_channels=dataset.num_classes,
        dropout=0.5,
        use_projection=True
    )

    try:
        checkpoint = torch.load('../build/models/reduced_graphsage_best.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded trained model")
    except:
        print("Warning: Could not load trained model")

    # Define pruning configuration (keep 75% of channels)
    pruning_config = {
        'conv1': 0.75,  # Keep 75% of hidden channels (24 -> 18)
    }

    print("\nApplying structured pruning...")
    pruning_masks = apply_structured_pruning(model, pruning_config)

    # Create pruned model
    pruned_model = create_pruned_model(model, pruning_masks)
    copy_pruned_weights(pruned_model, model, pruning_masks)

    print(f"\nOriginal model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Pruned model parameters: {sum(p.numel() for p in pruned_model.parameters())}")

    # Fine-tune
    print("\nFine-tuning pruned model...")
    pruned_model = fine_tune_pruned_model(pruned_model, data, epochs=50)

    # Save pruned model
    torch.save({
        'model_state_dict': pruned_model.state_dict(),
        'pruning_masks': pruning_masks,
        'architecture': {
            'in_channels_reduced': 16,
            'hidden_channels': len(pruning_masks['conv1']),
            'out_channels': dataset.num_classes
        }
    }, '../build/models/pruned_graphsage.pth')

    print("\nPruned model saved to ./models/pruned_graphsage.pth")
