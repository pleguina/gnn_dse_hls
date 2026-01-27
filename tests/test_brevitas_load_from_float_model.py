"""
Test loading weights from float model into Brevitas model.
Verifies that weight transfer and inference matching works correctly.
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
    from model_base import ReducedGraphSAGE
    from brevitas_models import (
        BrevitasQuantConfig,
        check_brevitas_available,
        BrevitasReducedGraphSAGE,
        load_from_float_model
    )
    BREVITAS_AVAILABLE = check_brevitas_available()
except ImportError as e:
    print(f"Import error: {e}")
    BREVITAS_AVAILABLE = False


@pytest.mark.skipif(not BREVITAS_AVAILABLE, reason="Brevitas not installed")
def test_load_weights_no_projection_no_root():
    """Test weight loading without projection and without root weight."""
    # Create float model
    float_model = ReducedGraphSAGE(
        in_channels=16,
        in_channels_reduced=16,
        hidden_channels=12,
        out_channels=4,
        dropout=0.0,
        use_projection=False,
        root_weight=False
    )
    
    # Create Brevitas model with matching architecture
    config = BrevitasQuantConfig(weight_bit_width=8, act_bit_width=8)
    brevitas_model = BrevitasReducedGraphSAGE(
        in_channels=16,
        in_channels_reduced=16,
        hidden_channels=12,
        out_channels=4,
        dropout=0.0,
        use_projection=False,
        root_weight=False,
        quant_config=config
    )
    
    # Load weights from float to Brevitas
    load_from_float_model(brevitas_model, float_model)
    
    # Verify weights were copied
    float_model.eval()
    brevitas_model.eval()
    
    # Check conv1 neighbor weights
    assert torch.allclose(
        brevitas_model.conv1.lin_neighbor.weight.data,
        float_model.conv1.lin_l.weight.data,
        atol=1e-6
    )
    
    # Check conv2 neighbor weights
    assert torch.allclose(
        brevitas_model.conv2.lin_neighbor.weight.data,
        float_model.conv2.lin_l.weight.data,
        atol=1e-6
    )
    
    print("✓ Weight loading verified (no projection, no root)")


@pytest.mark.skipif(not BREVITAS_AVAILABLE, reason="Brevitas not installed")
def test_load_weights_with_projection_no_root():
    """Test weight loading with projection but without root weight."""
    # Create float model
    float_model = ReducedGraphSAGE(
        in_channels=32,
        in_channels_reduced=16,
        hidden_channels=12,
        out_channels=4,
        dropout=0.0,
        use_projection=True,
        root_weight=False
    )
    
    # Create Brevitas model
    config = BrevitasQuantConfig(weight_bit_width=8, act_bit_width=8)
    brevitas_model = BrevitasReducedGraphSAGE(
        in_channels=32,
        in_channels_reduced=16,
        hidden_channels=12,
        out_channels=4,
        dropout=0.0,
        use_projection=True,
        root_weight=False,
        quant_config=config
    )
    
    # Load weights
    load_from_float_model(brevitas_model, float_model)
    
    # Verify projection weights
    assert torch.allclose(
        brevitas_model.projection.weight.data,
        float_model.projection.weight.data,
        atol=1e-6
    )
    
    # Verify conv weights
    assert torch.allclose(
        brevitas_model.conv1.lin_neighbor.weight.data,
        float_model.conv1.lin_l.weight.data,
        atol=1e-6
    )
    
    print("✓ Weight loading verified (with projection, no root)")


@pytest.mark.skipif(not BREVITAS_AVAILABLE, reason="Brevitas not installed")
def test_load_weights_with_projection_with_root():
    """Test weight loading with both projection and root weight."""
    # Create float model
    float_model = ReducedGraphSAGE(
        in_channels=32,
        in_channels_reduced=16,
        hidden_channels=12,
        out_channels=4,
        dropout=0.0,
        use_projection=True,
        root_weight=True
    )
    
    # Create Brevitas model
    config = BrevitasQuantConfig(weight_bit_width=8, act_bit_width=8)
    brevitas_model = BrevitasReducedGraphSAGE(
        in_channels=32,
        in_channels_reduced=16,
        hidden_channels=12,
        out_channels=4,
        dropout=0.0,
        use_projection=True,
        root_weight=True,
        quant_config=config
    )
    
    # Load weights
    load_from_float_model(brevitas_model, float_model)
    
    # Verify all weights including root
    assert torch.allclose(
        brevitas_model.projection.weight.data,
        float_model.projection.weight.data,
        atol=1e-6
    )
    
    assert torch.allclose(
        brevitas_model.conv1.lin_neighbor.weight.data,
        float_model.conv1.lin_l.weight.data,
        atol=1e-6
    )
    
    if float_model.conv1.lin_r is not None:
        assert torch.allclose(
            brevitas_model.conv1.lin_root.weight.data,
            float_model.conv1.lin_r.weight.data,
            atol=1e-6
        )
    
    print("✓ Weight loading verified (with projection and root)")


@pytest.mark.skipif(not BREVITAS_AVAILABLE, reason="Brevitas not installed")
def test_inference_comparison_high_precision():
    """
    Compare float and Brevitas model outputs with quantization.
    
    With 8-bit quantization, we verify the functional mapping is correct
    and check that outputs are in reasonable range.
    """
    # Create models with matching architecture
    float_model = ReducedGraphSAGE(
        in_channels=32,
        in_channels_reduced=16,
        hidden_channels=12,
        out_channels=4,
        dropout=0.0,
        use_projection=True,
        root_weight=False
    )
    
    # Use 8-bit quantization (highest precision available with Int8 presets)
    config = BrevitasQuantConfig(
        weight_bit_width=8,
        act_bit_width=8,
        return_quant_tensor=True
    )
    
    brevitas_model = BrevitasReducedGraphSAGE(
        in_channels=32,
        in_channels_reduced=16,
        hidden_channels=12,
        out_channels=4,
        dropout=0.0,
        use_projection=True,
        root_weight=False,
        quant_config=config
    )
    
    # Load weights
    load_from_float_model(brevitas_model, float_model)
    
    # Create synthetic test data
    num_nodes = 16
    num_edges = 32
    torch.manual_seed(42)
    x = torch.randn(num_nodes, 32)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Run inference
    float_model.eval()
    brevitas_model.eval()
    
    with torch.no_grad():
        float_out = float_model(x, edge_index)
        brevitas_out = brevitas_model(x, edge_index)
    
    # With 8-bit quantization, expect some difference but outputs should be reasonable
    max_diff = torch.max(torch.abs(float_out - brevitas_out)).item()
    mean_diff = torch.mean(torch.abs(float_out - brevitas_out)).item()
    
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    
    # With 8-bit quantization, differences should be reasonable
    assert max_diff < 10.0, f"Max diff {max_diff} too large"
    assert mean_diff < 5.0, f"Mean diff {mean_diff} too large"
    assert not torch.isnan(brevitas_out).any(), "NaN in brevitas output"
    assert not torch.isinf(brevitas_out).any(), "Inf in brevitas output"
    
    print("✓ Inference comparison passed (8-bit quantization)")


@pytest.mark.skipif(not BREVITAS_AVAILABLE, reason="Brevitas not installed")
def test_inference_shapes_match():
    """Test that output shapes match between float and Brevitas models."""
    # Create models
    float_model = ReducedGraphSAGE(
        in_channels=32,
        in_channels_reduced=16,
        hidden_channels=12,
        out_channels=4,
        dropout=0.0,
        use_projection=True,
        root_weight=False
    )
    
    config = BrevitasQuantConfig(weight_bit_width=8, act_bit_width=8)
    brevitas_model = BrevitasReducedGraphSAGE(
        in_channels=32,
        in_channels_reduced=16,
        hidden_channels=12,
        out_channels=4,
        dropout=0.0,
        use_projection=True,
        root_weight=False,
        quant_config=config
    )
    
    # Test with different graph sizes
    for num_nodes in [8, 16, 32]:
        num_edges = num_nodes * 2
        x = torch.randn(num_nodes, 32)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        float_model.eval()
        brevitas_model.eval()
        
        with torch.no_grad():
            float_out = float_model(x, edge_index)
            brevitas_out = brevitas_model(x, edge_index)
        
        assert float_out.shape == brevitas_out.shape, \
            f"Shape mismatch: {float_out.shape} vs {brevitas_out.shape}"
        assert float_out.shape == (num_nodes, 4)
    
    print("✓ Output shapes match across different graph sizes")


if __name__ == '__main__':
    # Run tests directly
    print("Testing weight loading from float model to Brevitas model...")
    print(f"Brevitas available: {BREVITAS_AVAILABLE}")
    
    if not BREVITAS_AVAILABLE:
        print("⚠ Brevitas not installed. Skipping tests.")
        print("Install with: pip install brevitas")
        sys.exit(0)
    
    print("\n" + "="*60)
    test_load_weights_no_projection_no_root()
    print("\n" + "="*60)
    test_load_weights_with_projection_no_root()
    print("\n" + "="*60)
    test_load_weights_with_projection_with_root()
    print("\n" + "="*60)
    test_inference_comparison_high_precision()
    print("\n" + "="*60)
    test_inference_shapes_match()
    print("\n" + "="*60)
    print("\n✓ All tests passed!")
