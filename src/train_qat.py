"""
Training script for QAT (Quantization-Aware Training) GraphSAGE model.
Trains with fake quantization to simulate INT8 hardware constraints.
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data
import os
import json

# Fix for PyTorch 2.6+ weights_only default change
torch.serialization.add_safe_globals([Data])

from model_qat import ReducedGraphSAGEQAT
from visualization import plot_training_curves


def train_qat(model, data, optimizer, enable_fake_quant=True):
    """
    Train the QAT model for one epoch.

    Args:
        enable_fake_quant: If True, fake quantization is active during training
    """
    model.train()
    if enable_fake_quant:
        model.enable_fake_quant()
        model.disable_observer()  # Observers only during calibration

    optimizer.zero_grad()

    # Forward pass
    out = model(data.x, data.edge_index)

    # Compute loss only on training nodes
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss.item()


def test_qat(model, data, enable_fake_quant=True):
    """
    Evaluate the QAT model on train, val, and test sets.

    Args:
        enable_fake_quant: If True, evaluate with fake quantization enabled
    """
    model.eval()

    if enable_fake_quant:
        model.enable_fake_quant()
    else:
        model.disable_fake_quant()

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

        # Compute accuracy for each split
        train_correct = pred[data.train_mask] == data.y[data.train_mask]
        train_acc = int(train_correct.sum()) / int(data.train_mask.sum())

        val_correct = pred[data.val_mask] == data.y[data.val_mask]
        val_acc = int(val_correct.sum()) / int(data.val_mask.sum())

        test_correct = pred[data.test_mask] == data.y[data.test_mask]
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())

    return train_acc, val_acc, test_acc


def calibrate_qat_model(model, data, num_batches=10):
    """
    Calibrate QAT observers to collect activation statistics.
    This should be done before training to initialize scale/zero-point.
    """
    print("\nCalibrating QAT observers...")
    model.eval()
    model.enable_observer()
    model.enable_fake_quant()

    with torch.no_grad():
        # Run several forward passes to collect statistics
        for _ in range(num_batches):
            _ = model(data.x, data.edge_index)

    model.disable_observer()
    print("Calibration complete!")


def load_cora_dataset(root='../data'):
    """Load and prepare Cora dataset."""
    dataset = Planetoid(root=root, name='Cora', transform=NormalizeFeatures())
    data = dataset[0]

    print(f'Dataset: {dataset}')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Training nodes: {data.train_mask.sum()}')
    print(f'Validation nodes: {data.val_mask.sum()}')
    print(f'Test nodes: {data.test_mask.sum()}')

    return dataset, data


def train_qat_model(
    epochs=200,
    lr=0.01,
    in_channels_reduced=16,
    hidden_channels=24,
    dropout=0.5,
    root_weight=False
):
    """
    Train QAT-enabled GraphSAGE model for FPGA implementation.

    Args:
        root_weight: If False (default), HLS-compatible (W_l * h_agg + b_l only)
                     If True, uses full GraphSAGE (W_l * h_agg + b_l + W_r * x_i)
    """
    suffix = "_qat" if root_weight else "_qat_no_root"
    print("\n" + "="*60)
    print(f"Training QAT GraphSAGE Model (FPGA-friendly)")
    print("="*60)
    if not root_weight:
        print("NOTE: Using root_weight=False for HLS compatibility")
        print("      Formula: out = W_l * h_agg + b_l")
    else:
        print("NOTE: Using root_weight=True (full GraphSAGE)")
        print("      Formula: out = W_l * h_agg + b_l + W_r * x_i")
    print("QAT: Fake quantization simulates INT8 arithmetic during training")

    # Load dataset
    dataset, data = load_cora_dataset()

    # Create QAT model
    model = ReducedGraphSAGEQAT(
        in_channels=dataset.num_features,
        in_channels_reduced=in_channels_reduced,
        hidden_channels=hidden_channels,
        out_channels=dataset.num_classes,
        dropout=dropout,
        use_projection=True,
        root_weight=root_weight,
        num_bits_acts=8,
        num_bits_weights=8,
    )

    print(f'\nModel architecture:\n{model}')
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')

    # Calibrate observers before training
    calibrate_qat_model(model, data, num_batches=10)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # Training loop with history tracking
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'test_acc': [],
        'train_acc_float': [],  # Accuracy without fake quant
        'val_acc_float': [],
        'test_acc_float': []
    }

    best_val_acc = 0
    for epoch in range(1, epochs + 1):
        # Train with fake quantization
        loss = train_qat(model, data, optimizer, enable_fake_quant=True)

        # Test with fake quantization (simulates INT8)
        train_acc, val_acc, test_acc = test_qat(model, data, enable_fake_quant=True)

        # Also test without fake quant to see float accuracy
        train_acc_float, val_acc_float, test_acc_float = test_qat(model, data, enable_fake_quant=False)

        # Track history
        history['train_loss'].append(loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['test_acc'].append(test_acc)
        history['train_acc_float'].append(train_acc_float)
        history['val_acc_float'].append(val_acc_float)
        history['test_acc_float'].append(test_acc_float)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save best model
            model_path = f'../build/models/reduced_graphsage{suffix}_best.pth'
            os.makedirs('../build/models', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_acc_float': val_acc_float,
                'root_weight': root_weight,
                'in_channels_reduced': in_channels_reduced,
                'hidden_channels': hidden_channels,
                'out_channels': dataset.num_classes,
            }, model_path)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
            print(f'  QAT (INT8-sim): Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
            print(f'  Float:          Train: {train_acc_float:.4f}, Val: {val_acc_float:.4f}, Test: {test_acc_float:.4f}')

    # Load best model and evaluate
    model_path = f'../build/models/reduced_graphsage{suffix}_best.pth'
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    train_acc, val_acc, test_acc = test_qat(model, data, enable_fake_quant=True)
    train_acc_float, val_acc_float, test_acc_float = test_qat(model, data, enable_fake_quant=False)

    print(f'\nBest model performance:')
    print(f'QAT (INT8-sim): Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    print(f'Float:          Train: {train_acc_float:.4f}, Val: {val_acc_float:.4f}, Test: {test_acc_float:.4f}')
    print(f'Quantization gap: {val_acc_float - val_acc:.4f} (lower is better)')

    # Save training history
    history_path = f'../build/qat_model{suffix}_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Plot training curves
    os.makedirs('../build/plots', exist_ok=True)
    plot_path = f'../build/plots/qat_model{suffix}_training.png'
    plot_training_curves(history, save_path=plot_path)

    return model, data, history


if __name__ == '__main__':
    # Train QAT model WITHOUT root_weight (HLS-compatible)
    print("\n" + "="*60)
    print("Training QAT Model WITHOUT root_weight (HLS-compatible)")
    print("="*60)
    qat_model, data, qat_history = train_qat_model(epochs=200, root_weight=False)

    print("\n" + "="*60)
    print("QAT Training Complete!")
    print("="*60)
    print("Saved models:")
    print("  - ../build/models/reduced_graphsage_qat_no_root_best.pth")
    print("\nTraining plots:")
    print("  - ../build/plots/qat_model_qat_no_root_training.png")
    print("\nNext steps:")
    print("  1. Run quantization export: python quantization_qat.py")
    print("  2. Generate test vectors for HLS verification")
    print("="*60)
