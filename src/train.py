"""
Training script for GraphSAGE models on Cora dataset.
Implements training for both base and reduced models.
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data
import os
import sys
import json

# Fix for PyTorch 2.6+ weights_only default change
torch.serialization.add_safe_globals([Data])

from model_base import GraphSAGE, ReducedGraphSAGE
from visualization import plot_training_curves


def train(model, data, optimizer):
    """Train the model for one epoch."""
    model.train()
    optimizer.zero_grad()

    # Forward pass
    out = model(data.x, data.edge_index)

    # Compute loss only on training nodes
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model, data):
    """Evaluate the model on train, val, and test sets."""
    model.eval()

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
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Training nodes: {data.train_mask.sum()}')
    print(f'Validation nodes: {data.val_mask.sum()}')
    print(f'Test nodes: {data.test_mask.sum()}')

    return dataset, data


def train_base_model(epochs=200, lr=0.01, hidden_channels=64, dropout=0.5):
    """Train base GraphSAGE model on full Cora dataset."""
    print("\n" + "="*60)
    print("Training Base GraphSAGE Model")
    print("="*60)

    # Load dataset
    dataset, data = load_cora_dataset()

    # Create model
    model = GraphSAGE(
        in_channels=dataset.num_features,
        hidden_channels=hidden_channels,
        out_channels=dataset.num_classes,
        dropout=dropout
    )

    print(f'\nModel architecture:\n{model}')
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # Training loop with history tracking
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'test_acc': []
    }

    best_val_acc = 0
    for epoch in range(1, epochs + 1):
        loss = train(model, data, optimizer)
        train_acc, val_acc, test_acc = test(model, data)

        # Track history
        history['train_loss'].append(loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['test_acc'].append(test_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, '../build/models/base_graphsage_best.pth')

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                  f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

    # Load best model and evaluate
    checkpoint = torch.load('../build/models/base_graphsage_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    train_acc, val_acc, test_acc = test(model, data)

    print(f'\nBest model performance:')
    print(f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

    # Save training history
    os.makedirs('../outputs', exist_ok=True)
    with open('../build/base_model_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Plot training curves
    plot_training_curves(history, save_path='../build/plots/base_model_training.png')

    return model, data, history


def train_reduced_model(epochs=200, lr=0.01, in_channels_reduced=16,
                       hidden_channels=24, dropout=0.5):
    """Train reduced GraphSAGE model for FPGA implementation."""
    print("\n" + "="*60)
    print("Training Reduced GraphSAGE Model (FPGA-friendly)")
    print("="*60)

    # Load dataset
    dataset, data = load_cora_dataset()

    # Create reduced model
    model = ReducedGraphSAGE(
        in_channels=dataset.num_features,
        in_channels_reduced=in_channels_reduced,
        hidden_channels=hidden_channels,
        out_channels=dataset.num_classes,
        dropout=dropout,
        use_projection=True
    )

    print(f'\nModel architecture:\n{model}')
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # Training loop with history tracking
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'test_acc': []
    }

    best_val_acc = 0
    for epoch in range(1, epochs + 1):
        loss = train(model, data, optimizer)
        train_acc, val_acc, test_acc = test(model, data)

        # Track history
        history['train_loss'].append(loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['test_acc'].append(test_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, '../build/models/reduced_graphsage_best.pth')

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                  f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

    # Load best model and evaluate
    checkpoint = torch.load('../build/models/reduced_graphsage_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    train_acc, val_acc, test_acc = test(model, data)

    print(f'\nBest model performance:')
    print(f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

    # Save training history
    with open('../build/reduced_model_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Plot training curves
    plot_training_curves(history, save_path='../build/plots/reduced_model_training.png')

    return model, data, history


if __name__ == '__main__':
    # Train base model
    base_model, data, base_history = train_base_model(epochs=200)

    # Train reduced model for FPGA
    reduced_model, data, reduced_history = train_reduced_model(epochs=200)

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("Saved models:")
    print("  - ../models/base_graphsage_best.pth")
    print("  - ../models/reduced_graphsage_best.pth")
    print("\nTraining plots:")
    print("  - ../outputs/plots/base_model_training.png")
    print("  - ../outputs/plots/reduced_model_training.png")
