"""
Test Brevitas model import and forward pass with synthetic data.
Verifies that Brevitas models can be created and run inference.
"""

import sys
import os
import torch

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # Define a dummy skipif decorator
    class pytest:
        class mark:
            @staticmethod
            def skipif(condition, reason=""):
                def decorator(func):
                    if condition:
                        def wrapper(*args, **kwargs):
                            print(f"SKIPPED: {reason}")
                            return None
                        return wrapper
                    return func
                return decorator

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from brevitas_models import (
        BrevitasQuantConfig, 
        check_brevitas_available,
        BrevitasReducedGraphSAGE
    )
    BREVITAS_AVAILABLE = check_brevitas_available()
except ImportError as e:
    print(f"Import error: {e}")
    BREVITAS_AVAILABLE = False


@pytest.mark.skipif(not BREVITAS_AVAILABLE, reason="Brevitas not installed")
def test_brevitas_model_creation():
    """Test Brevitas model creation."""
    config = BrevitasQuantConfig(weight_bit_width=8, act_bit_width=8)
    
    model = BrevitasReducedGraphSAGE(
        in_channels=16,
        in_channels_reduced=8,
        hidden_channels=12,
        out_channels=4,
        dropout=0.5,
        use_projection=True,
        root_weight=False,
        quant_config=config
    )
    
    assert model is not None
    assert model.use_projection == True
    assert model.root_weight == False
    print("✓ Model creation successful")


@pytest.mark.skipif(not BREVITAS_AVAILABLE, reason="Brevitas not installed")
def test_brevitas_forward_no_projection_no_root():
    """Test forward pass without projection and without root weight."""
    config = BrevitasQuantConfig(weight_bit_width=8, act_bit_width=8)
    
    model = BrevitasReducedGraphSAGE(
        in_channels=16,
        in_channels_reduced=16,
        hidden_channels=12,
        out_channels=4,
        dropout=0.0,  # Disable dropout for deterministic test
        use_projection=False,
        root_weight=False,
        quant_config=config
    )
    
    # Synthetic graph data
    num_nodes = 8
    num_edges = 16
    x = torch.randn(num_nodes, 16)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
    
    assert out.shape == (num_nodes, 4)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    print(f"✓ Forward pass successful: {x.shape} -> {out.shape}")


@pytest.mark.skipif(not BREVITAS_AVAILABLE, reason="Brevitas not installed")
def test_brevitas_forward_with_projection_no_root():
    """Test forward pass with projection but without root weight."""
    config = BrevitasQuantConfig(weight_bit_width=8, act_bit_width=8)
    
    model = BrevitasReducedGraphSAGE(
        in_channels=32,
        in_channels_reduced=16,
        hidden_channels=12,
        out_channels=4,
        dropout=0.0,
        use_projection=True,
        root_weight=False,
        quant_config=config
    )
    
    # Synthetic graph data
    num_nodes = 8
    num_edges = 16
    x = torch.randn(num_nodes, 32)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
    
    assert out.shape == (num_nodes, 4)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    print(f"✓ Forward pass with projection successful: {x.shape} -> {out.shape}")


@pytest.mark.skipif(not BREVITAS_AVAILABLE, reason="Brevitas not installed")
def test_brevitas_forward_with_projection_with_root():
    """Test forward pass with both projection and root weight."""
    config = BrevitasQuantConfig(weight_bit_width=8, act_bit_width=8)
    
    model = BrevitasReducedGraphSAGE(
        in_channels=32,
        in_channels_reduced=16,
        hidden_channels=12,
        out_channels=4,
        dropout=0.0,
        use_projection=True,
        root_weight=True,
        quant_config=config
    )
    
    # Synthetic graph data
    num_nodes = 8
    num_edges = 16
    x = torch.randn(num_nodes, 32)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
    
    assert out.shape == (num_nodes, 4)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    print(f"✓ Forward pass with projection and root weight successful")


@pytest.mark.skipif(not BREVITAS_AVAILABLE, reason="Brevitas not installed")
def test_brevitas_different_bit_widths():
    """Test that model works with different bit width configurations."""
    # Try different bit widths - only test 8-bit if generic quantizers not available
    test_configs = [(8, 8)]  # Always test 8-bit
    
    # If generic quantizers available, test other bit widths
    try:
        from brevitas.quant import IntWeightPerTensorFloat
        test_configs.extend([(4, 4), (16, 16)])
    except ImportError:
        print("  Note: Only testing 8-bit (generic quantizers not available)")
    
    for w_bits, a_bits in test_configs:
        config = BrevitasQuantConfig(weight_bit_width=w_bits, act_bit_width=a_bits)
        
        model = BrevitasReducedGraphSAGE(
            in_channels=16,
            in_channels_reduced=8,
            hidden_channels=12,
            out_channels=4,
            dropout=0.0,
            use_projection=True,
            root_weight=False,
            quant_config=config
        )
        
        # Synthetic graph data
        num_nodes = 8
        num_edges = 16
        x = torch.randn(num_nodes, 16)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)
        
        assert out.shape == (num_nodes, 4)
        print(f"✓ Model works with w_bits={w_bits}, a_bits={a_bits}")


if __name__ == '__main__':
    # Run tests directly
    print("Testing Brevitas model import and forward pass...")
    print(f"Brevitas available: {BREVITAS_AVAILABLE}")
    
    if not BREVITAS_AVAILABLE:
        print("⚠ Brevitas not installed. Skipping tests.")
        print("Install with: pip install brevitas")
        sys.exit(0)
    
    print("\n" + "="*60)
    test_brevitas_model_creation()
    print("\n" + "="*60)
    test_brevitas_forward_no_projection_no_root()
    print("\n" + "="*60)
    test_brevitas_forward_with_projection_no_root()
    print("\n" + "="*60)
    test_brevitas_forward_with_projection_with_root()
    print("\n" + "="*60)
    test_brevitas_different_bit_widths()
    print("\n" + "="*60)
    print("\n✓ All tests passed!")
