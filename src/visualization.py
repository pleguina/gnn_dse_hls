"""
Visualization utilities for model analysis and comparison.
Generates plots for training curves, model efficiency, quantization effects, etc.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import json
import os
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def plot_training_curves(history, save_path='../build/plots/training_curves.png'):
    """
    Plot training and validation curves.

    Args:
        history: Dictionary with keys 'train_loss', 'train_acc', 'val_acc', 'test_acc'
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss over Epochs')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].plot(epochs, history['test_acc'], 'g-', label='Test Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy over Epochs')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved training curves to {save_path}")
    plt.close()


def plot_model_comparison(model_stats, save_path='../build/plots/model_comparison.png'):
    """
    Compare different model variants (base, reduced, pruned, quantized).

    Args:
        model_stats: List of dicts with keys 'name', 'accuracy', 'parameters', 'memory_mb'
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    names = [stat['name'] for stat in model_stats]
    accuracies = [stat['accuracy'] for stat in model_stats]
    parameters = [stat['parameters'] / 1e3 for stat in model_stats]  # Convert to K
    memory = [stat['memory_mb'] for stat in model_stats]

    colors = sns.color_palette("husl", len(names))

    # Accuracy comparison
    bars1 = axes[0].bar(names, accuracies, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Test Accuracy')
    axes[0].set_title('Model Accuracy Comparison')
    axes[0].set_ylim([0, 1.0])
    axes[0].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=9)

    # Parameters comparison
    bars2 = axes[1].bar(names, parameters, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Parameters (K)')
    axes[1].set_title('Model Size (Parameters)')
    axes[1].grid(True, alpha=0.3, axis='y')

    for bar, param in zip(bars2, parameters):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{param:.1f}K', ha='center', va='bottom', fontsize=9)

    # Memory comparison
    bars3 = axes[2].bar(names, memory, color=colors, alpha=0.7, edgecolor='black')
    axes[2].set_ylabel('Memory (MB)')
    axes[2].set_title('Model Memory Footprint')
    axes[2].grid(True, alpha=0.3, axis='y')

    for bar, mem in zip(bars3, memory):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{mem:.2f}MB', ha='center', va='bottom', fontsize=9)

    # Rotate x-axis labels if needed
    for ax in axes:
        ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved model comparison to {save_path}")
    plt.close()


def plot_efficiency_analysis(model_stats, save_path='../build/plots/efficiency_analysis.png'):
    """
    Plot efficiency metrics: accuracy vs parameters and accuracy vs memory.

    Args:
        model_stats: List of dicts with keys 'name', 'accuracy', 'parameters', 'memory_mb'
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    names = [stat['name'] for stat in model_stats]
    accuracies = [stat['accuracy'] * 100 for stat in model_stats]  # Convert to percentage
    parameters = [stat['parameters'] / 1e3 for stat in model_stats]
    memory = [stat['memory_mb'] for stat in model_stats]

    colors = sns.color_palette("husl", len(names))

    # Accuracy vs Parameters
    axes[0].scatter(parameters, accuracies, s=200, c=colors, alpha=0.7, edgecolors='black', linewidths=2)
    for i, name in enumerate(names):
        axes[0].annotate(name, (parameters[i], accuracies[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
    axes[0].set_xlabel('Model Parameters (K)')
    axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].set_title('Accuracy vs Model Size\n(Higher & Left is Better)')
    axes[0].grid(True, alpha=0.3)

    # Accuracy vs Memory
    axes[1].scatter(memory, accuracies, s=200, c=colors, alpha=0.7, edgecolors='black', linewidths=2)
    for i, name in enumerate(names):
        axes[1].annotate(name, (memory[i], accuracies[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
    axes[1].set_xlabel('Memory Footprint (MB)')
    axes[1].set_ylabel('Test Accuracy (%)')
    axes[1].set_title('Accuracy vs Memory Usage\n(Higher & Left is Better)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved efficiency analysis to {save_path}")
    plt.close()


def plot_quantization_error(original_weights, quantized_weights, layer_name,
                           save_path='../build/plots/quantization_error.png'):
    """
    Analyze quantization error for weights.

    Args:
        original_weights: Original float weights (numpy array)
        quantized_weights: Quantized weights (numpy array)
        layer_name: Name of the layer
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Compute errors
    errors = original_weights.flatten() - quantized_weights.flatten()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Original weights distribution
    axes[0, 0].hist(original_weights.flatten(), bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Weight Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'{layer_name} - Original Weights Distribution')
    axes[0, 0].grid(True, alpha=0.3)

    # Quantized weights distribution
    axes[0, 1].hist(quantized_weights.flatten(), bins=50, color='red', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Weight Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'{layer_name} - Quantized Weights Distribution')
    axes[0, 1].grid(True, alpha=0.3)

    # Error distribution
    axes[1, 0].hist(errors, bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Quantization Error')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'{layer_name} - Quantization Error Distribution')
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].grid(True, alpha=0.3)

    # Error statistics
    stats_text = f'Mean Error: {np.mean(errors):.6f}\n'
    stats_text += f'Std Error: {np.std(errors):.6f}\n'
    stats_text += f'Max Error: {np.max(np.abs(errors)):.6f}\n'
    stats_text += f'RMSE: {np.sqrt(np.mean(errors**2)):.6f}'

    axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                   fontsize=14, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')
    axes[1, 1].set_title(f'{layer_name} - Error Statistics')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved quantization error analysis to {save_path}")
    plt.close()


def plot_pruning_effect(original_channels, pruned_channels, layer_names,
                       save_path='../build/plots/pruning_effect.png'):
    """
    Visualize the effect of pruning on each layer.

    Args:
        original_channels: List of original channel counts
        pruned_channels: List of pruned channel counts
        layer_names: List of layer names
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(layer_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, original_channels, width, label='Original',
                  color='blue', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, pruned_channels, width, label='After Pruning',
                  color='red', alpha=0.7, edgecolor='black')

    ax.set_xlabel('Layer')
    ax.set_ylabel('Number of Channels')
    ax.set_title('Effect of Structured Pruning on Layer Channels')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels and pruning percentage
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()

        ax.text(bar1.get_x() + bar1.get_width()/2., height1,
               f'{int(height1)}', ha='center', va='bottom', fontsize=9)
        ax.text(bar2.get_x() + bar2.get_width()/2., height2,
               f'{int(height2)}', ha='center', va='bottom', fontsize=9)

        # Pruning percentage
        prune_pct = (1 - height2/height1) * 100
        ax.text(i, max(height1, height2) * 1.1,
               f'-{prune_pct:.1f}%', ha='center', fontsize=9,
               color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved pruning effect plot to {save_path}")
    plt.close()


def plot_accuracy_degradation(base_acc, model_accs, model_names,
                              save_path='../build/plots/accuracy_degradation.png'):
    """
    Show accuracy degradation from optimizations.

    Args:
        base_acc: Base model accuracy (float)
        model_accs: List of accuracies for other models
        model_names: List of model names
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    all_names = ['Base'] + model_names
    all_accs = [base_acc] + model_accs
    degradations = [0] + [(base_acc - acc) * 100 for acc in model_accs]

    colors = ['green'] + ['orange' if d < 5 else 'red' for d in degradations[1:]]

    bars = ax.bar(all_names, [a * 100 for a in all_accs], color=colors,
                  alpha=0.7, edgecolor='black', linewidth=2)

    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Accuracy Comparison: Impact of Optimizations')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=base_acc * 100, color='blue', linestyle='--', linewidth=2, label='Base Model')

    # Add value labels
    for i, (bar, acc, deg) in enumerate(zip(bars, all_accs, degradations)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{acc*100:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

        if i > 0:  # Skip base model
            ax.text(bar.get_x() + bar.get_width()/2., height - 5,
                   f'({deg:+.2f}%)', ha='center', va='top', fontsize=8, color='darkred')

    ax.legend()
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved accuracy degradation plot to {save_path}")
    plt.close()


def plot_resource_utilization(model_stats, save_path='../build/plots/resource_utilization.png'):
    """
    Stacked bar chart showing resource breakdown.

    Args:
        model_stats: List of dicts with resource information
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    names = [stat['name'] for stat in model_stats]

    # Normalized parameters by layer type
    layer_params = []
    for stat in model_stats:
        layers = stat.get('layer_breakdown', {'conv': 0, 'linear': 0, 'other': 0})
        layer_params.append([layers.get('conv', 0), layers.get('linear', 0), layers.get('other', 0)])

    layer_params = np.array(layer_params).T / 1e3  # Convert to K

    colors = ['#ff9999', '#66b3ff', '#99ff99']
    labels = ['Conv Layers', 'Linear Layers', 'Other']

    bottom = np.zeros(len(names))
    for i, (params, color, label) in enumerate(zip(layer_params, colors, labels)):
        bars = ax.bar(names, params, bottom=bottom, label=label,
                     color=color, alpha=0.8, edgecolor='black')
        bottom += params

        # Add labels
        for j, (bar, p) in enumerate(zip(bars, params)):
            if p > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.,
                       bottom[j] - height/2,
                       f'{p:.1f}K', ha='center', va='center', fontsize=8)

    ax.set_ylabel('Parameters (K)')
    ax.set_title('Model Resource Breakdown by Layer Type')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved resource utilization plot to {save_path}")
    plt.close()


def generate_summary_report(model_stats, save_path='../build/plots/summary_report.png'):
    """
    Generate a comprehensive summary figure with multiple subplots.

    Args:
        model_stats: List of model statistics
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    names = [stat['name'] for stat in model_stats]
    accuracies = [stat['accuracy'] * 100 for stat in model_stats]
    parameters = [stat['parameters'] / 1e3 for stat in model_stats]
    memory = [stat['memory_mb'] for stat in model_stats]
    colors = sns.color_palette("husl", len(names))

    # Accuracy comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(names, accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Test Accuracy')
    ax1.tick_params(axis='x', rotation=15)
    ax1.grid(True, alpha=0.3, axis='y')

    # Parameters comparison
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(names, parameters, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Parameters (K)')
    ax2.set_title('Model Size')
    ax2.tick_params(axis='x', rotation=15)
    ax2.grid(True, alpha=0.3, axis='y')

    # Memory comparison
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(names, memory, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Memory (MB)')
    ax3.set_title('Memory Footprint')
    ax3.tick_params(axis='x', rotation=15)
    ax3.grid(True, alpha=0.3, axis='y')

    # Efficiency: Accuracy vs Parameters
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.scatter(parameters, accuracies, s=300, c=colors, alpha=0.7, edgecolors='black', linewidths=2)
    for i, name in enumerate(names):
        ax4.annotate(name, (parameters[i], accuracies[i]),
                    xytext=(8, 8), textcoords='offset points', fontsize=10)
    ax4.set_xlabel('Parameters (K)')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('Model Efficiency: Accuracy vs Size')
    ax4.grid(True, alpha=0.3)

    # Summary table
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('tight')
    ax5.axis('off')

    table_data = []
    for stat in model_stats:
        reduction = (1 - stat['parameters'] / model_stats[0]['parameters']) * 100
        table_data.append([
            stat['name'][:10],
            f"{stat['accuracy']*100:.2f}%",
            f"{stat['parameters']/1e3:.1f}K",
            f"{reduction:.1f}%"
        ])

    table = ax5.table(cellText=table_data,
                     colLabels=['Model', 'Acc', 'Params', 'Reduction'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax5.set_title('Summary Statistics', pad=20)

    # Speedup / Compression ratio
    ax6 = fig.add_subplot(gs[2, :])
    base_params = model_stats[0]['parameters']
    compression_ratios = [base_params / stat['parameters'] for stat in model_stats]

    bars = ax6.barh(names, compression_ratios, color=colors, alpha=0.7, edgecolor='black')
    ax6.set_xlabel('Compression Ratio (vs Base Model)')
    ax6.set_title('Model Compression Achieved')
    ax6.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Base Model')
    ax6.grid(True, alpha=0.3, axis='x')
    ax6.legend()

    for bar, ratio in zip(bars, compression_ratios):
        width = bar.get_width()
        ax6.text(width, bar.get_y() + bar.get_height()/2.,
                f'{ratio:.2f}x', ha='left', va='center', fontsize=10, fontweight='bold')

    fig.suptitle('GraphSAGE Model Optimization Summary', fontsize=16, fontweight='bold')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved summary report to {save_path}")
    plt.close()


def save_stats_json(model_stats, save_path='../build/plots/model_stats.json'):
    """Save model statistics to JSON file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(model_stats, f, indent=2)

    print(f"Saved model statistics to {save_path}")
