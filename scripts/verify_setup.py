#!/usr/bin/env python3
"""
Verification script to test that the environment is set up correctly.
"""

import sys
import os

# Add src to path
sys.path.insert(0, 'src')

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        import torch
        import torch_geometric
        import numpy
        import scipy
        from model_base import GraphSAGE, ReducedGraphSAGE
        print("✓ All imports successful")
        print(f"  - PyTorch version: {torch.__version__}")
        print(f"  - PyTorch Geometric version: {torch_geometric.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_dataset():
    """Test that Cora dataset can be loaded."""
    print("\nTesting dataset loading...")
    try:
        from torch_geometric.datasets import Planetoid
        from torch_geometric.transforms import NormalizeFeatures

        dataset = Planetoid(root='./data', name='Cora', transform=NormalizeFeatures())
        data = dataset[0]

        print("✓ Cora dataset loaded successfully")
        print(f"  - Nodes: {data.num_nodes}")
        print(f"  - Edges: {data.num_edges}")
        print(f"  - Features: {dataset.num_features}")
        print(f"  - Classes: {dataset.num_classes}")
        return True
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False


def test_model():
    """Test that models can be instantiated and run forward pass."""
    print("\nTesting model instantiation...")
    try:
        import torch
        from model_base import GraphSAGE, ReducedGraphSAGE
        from torch_geometric.datasets import Planetoid
        from torch_geometric.transforms import NormalizeFeatures

        # Load dataset
        dataset = Planetoid(root='./data', name='Cora', transform=NormalizeFeatures())
        data = dataset[0]

        # Test base model
        model = GraphSAGE(
            in_channels=dataset.num_features,
            hidden_channels=16,
            out_channels=dataset.num_classes,
            dropout=0.5
        )

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)

        print("✓ Base GraphSAGE model works")
        print(f"  - Input shape: {data.x.shape}")
        print(f"  - Output shape: {out.shape}")

        # Test reduced model
        reduced_model = ReducedGraphSAGE(
            in_channels=dataset.num_features,
            in_channels_reduced=16,
            hidden_channels=24,
            out_channels=dataset.num_classes,
            dropout=0.5,
            use_projection=True
        )

        reduced_model.eval()
        with torch.no_grad():
            out = reduced_model(data.x, data.edge_index)

        print("✓ Reduced GraphSAGE model works")
        print(f"  - Output shape: {out.shape}")

        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_directory_structure():
    """Test that all required directories exist."""
    print("\nTesting directory structure...")
    required_dirs = ['src', 'hls', 'tests', 'models', 'data', 'outputs', 'configs']
    all_exist = True

    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✓ {dir_name}/ exists")
        else:
            print(f"✗ {dir_name}/ missing")
            all_exist = False

    return all_exist


def main():
    print("="*60)
    print("Simple GNN Setup Verification")
    print("="*60)

    results = []
    results.append(("Imports", test_imports()))
    results.append(("Directory Structure", test_directory_structure()))
    results.append(("Dataset Loading", test_dataset()))
    results.append(("Model Forward Pass", test_model()))

    print("\n" + "="*60)
    print("Verification Results")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n✓ All tests passed! Your environment is ready.")
        print("\nNext steps:")
        print("  1. Run training: python run_pipeline.py")
        print("  2. Or train manually: cd src && python train.py")
        print("  3. See README.md for detailed instructions")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
