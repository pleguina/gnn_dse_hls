"""
Brevitas quantization layer helpers for GraphSAGE models.
Utilities to build quantized linear layers and activations.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

BREVITAS_AVAILABLE = False

try:
    from brevitas.nn import QuantLinear, QuantReLU, QuantIdentity

    # Use fixed-point quantizers for power-of-two scales
    # These are needed for proper fixed-point hardware export
    try:
        from brevitas.quant.fixed_point import Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint
        _HAS_FIXED_POINT_QUANT = True
    except Exception:
        _HAS_FIXED_POINT_QUANT = False
        # Fallback to float quantizers (not ideal for hardware)
        from brevitas.quant import Int8WeightPerTensorFloat as Int8WeightPerTensorFixedPoint
        from brevitas.quant import Int8ActPerTensorFloat as Int8ActPerTensorFixedPoint

    BREVITAS_AVAILABLE = True
except ImportError:
    BREVITAS_AVAILABLE = False


@dataclass
class BrevitasQuantConfig:
    """Configuration for Brevitas quantization parameters."""
    weight_bit_width: int = 8
    act_bit_width: int = 8

    # Bias quantization is tricky; keep it off unless you know the quantizer you want.
    enable_bias_quant: bool = False

    # Return QuantTensor to propagate scale/zero-point information downstream.
    return_quant_tensor: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BrevitasQuantConfig":
        return cls(**d)


def _require_brevitas() -> None:
    if not BREVITAS_AVAILABLE:
        raise ImportError("Brevitas is not available. Install with: pip install brevitas")


def make_quant_linear(
    in_features: int,
    out_features: int,
    bias: bool = True,
    quant_config: Optional[BrevitasQuantConfig] = None,
) -> nn.Module:
    """
    Create a Brevitas quantized linear layer.
    """
    _require_brevitas()
    if quant_config is None:
        quant_config = BrevitasQuantConfig()

    if quant_config.enable_bias_quant:
        # IMPORTANT:
        # QuantLinear.bias_quant expects a bias quantizer class/callable, not a dict.
        # You can add one later once you decide the exact bias quantizer.
        raise NotImplementedError(
            "Bias quantization is enabled, but no bias quantizer is configured. "
            "Keep enable_bias_quant=False for now."
        )

    # Use fixed-point quantizers for power-of-two scales
    weight_quant = Int8WeightPerTensorFixedPoint
    act_quant = Int8ActPerTensorFixedPoint

    layer = QuantLinear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        weight_quant=weight_quant,
        weight_bit_width=quant_config.weight_bit_width,
        bias_quant=None,
        # ✅ Let QuantLinear quantize tensor inputs itself
        input_quant=act_quant,
        input_bit_width=quant_config.act_bit_width,
        return_quant_tensor=quant_config.return_quant_tensor,
    )
    return layer


def make_quant_relu(
    quant_config: Optional[BrevitasQuantConfig] = None,
) -> nn.Module:
    """
    Create a Brevitas quantized ReLU activation.
    """
    _require_brevitas()
    if quant_config is None:
        quant_config = BrevitasQuantConfig()

    act_quant = Int8ActPerTensorFixedPoint

    layer = QuantReLU(
        act_quant=act_quant,
        bit_width=quant_config.act_bit_width,
        return_quant_tensor=quant_config.return_quant_tensor,
    )
    return layer


def make_quant_identity(
    quant_config: Optional[BrevitasQuantConfig] = None,
) -> nn.Module:
    """
    Create a Brevitas quantized identity layer (useful to quantize inputs).
    """
    _require_brevitas()
    if quant_config is None:
        quant_config = BrevitasQuantConfig()

    if _HAS_GENERIC_INT_QUANT:
        act_quant = IntActPerTensorFloat
    else:
        if quant_config.act_bit_width != 8:
            raise ValueError(
                "This Brevitas install only provides INT8 preset quantizers. "
                f"Requested act_bit_width={quant_config.act_bit_width}."
            )
        act_quant = Int8ActPerTensorFloat

    layer = QuantIdentity(
        act_quant=act_quant,
        bit_width=quant_config.act_bit_width,
        return_quant_tensor=quant_config.return_quant_tensor,
    )
    return layer


def check_brevitas_available() -> bool:
    return BREVITAS_AVAILABLE


class BrevitasGraphSAGELayer(nn.Module):
    """
    Quantized GraphSAGE layer using explicit mean aggregation + quantized linear transforms.
    
    Quantization scope:
    - Linear transforms (neighbor & optional root): INT weights & activations via Brevitas
    - Mean aggregation: Float arithmetic (scatter_add, division)
    
    This implements:
        out = W_neighbor @ mean(neighbor_features) + b_neighbor [+ W_root @ self_features]
    
    Where W_neighbor and optionally W_root are quantized linear layers.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        root_weight: bool = True,
        quant_config: Optional[BrevitasQuantConfig] = None,
    ):
        super().__init__()
        _require_brevitas()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.root_weight = root_weight
        self.quant_config = quant_config if quant_config else BrevitasQuantConfig()
        
        # Quantized linear for neighbor aggregation
        self.lin_neighbor = make_quant_linear(
            in_channels, out_channels, bias=True, quant_config=self.quant_config
        )
        
        # Optional quantized linear for root (self) connection
        if root_weight:
            self.lin_root = make_quant_linear(
                in_channels, out_channels, bias=False, quant_config=self.quant_config
            )
        else:
            self.lin_root = None
    
    def forward(self, x, edge_index):
        """
        Forward pass with explicit mean aggregation.
        
        Args:
            x: Node features [num_nodes, in_channels] (can be QuantTensor)
            edge_index: Edge connectivity [2, num_edges]
        
        Returns:
            out: Transformed features [num_nodes, out_channels]
        
        Note: QuantTensor scale info is dropped before aggregation (aggregation in float).
        QuantLinear layers will re-quantize their inputs internally.
        """
        # Extract float values (drop QuantTensor if present)
        # Aggregation will be in float; QuantLinear will re-quantize
        if hasattr(x, 'value'):
            x_value = x.value
        else:
            x_value = x
        
        num_nodes = x_value.size(0)
        
        # Manual mean aggregation (message passing in float)
        row, col = edge_index
        
        # Aggregate neighbor features
        aggregated = torch.zeros(num_nodes, self.in_channels, device=x_value.device, dtype=x_value.dtype)
        degree = torch.zeros(num_nodes, device=x_value.device, dtype=x_value.dtype)
        ones = torch.ones(row.size(0), device=x_value.device, dtype=x_value.dtype)
        degree.scatter_add_(0, col, ones)
        degree = degree.clamp(min=1.0)
        
        aggregated.scatter_add_(0, col.unsqueeze(1).expand(-1, self.in_channels), x_value[row])
        aggregated = aggregated / degree.unsqueeze(1)
        
        # Apply quantized linear to aggregated neighbors
        out = self.lin_neighbor(aggregated)
        
        # Add root connection if enabled
        if self.root_weight:
            root_out = self.lin_root(x_value)
            
            # Handle QuantTensor addition
            if hasattr(out, 'value') and hasattr(root_out, 'value'):
                out = out.value + root_out.value
            elif hasattr(out, 'value'):
                out = out.value + root_out
            elif hasattr(root_out, 'value'):
                out = out + root_out.value
            else:
                out = out + root_out
        elif hasattr(out, 'value'):
            out = out.value
        
        return out


class BrevitasReducedGraphSAGE(nn.Module):
    """
    Brevitas-quantized reduced GraphSAGE model for FPGA implementation.
    
    Architecture matches ReducedGraphSAGE but with quantized layers:
    - Optional quantized projection: Linear + QuantReLU
    - QuantSAGE conv1 + QuantReLU + Dropout
    - QuantSAGE conv2
    """
    
    def __init__(
        self,
        in_channels: int,
        in_channels_reduced: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.5,
        use_projection: bool = True,
        root_weight: bool = False,
        quant_config: Optional[BrevitasQuantConfig] = None,
    ):
        super().__init__()
        _require_brevitas()
        
        self.use_projection = use_projection
        self.root_weight = root_weight
        self.dropout = dropout
        self.quant_config = quant_config if quant_config else BrevitasQuantConfig()
        
        # Note: No input QuantIdentity needed - all QuantLinear layers
        # have input_quant configured, so they quantize their inputs internally.
        
        # Optional projection layer
        if use_projection:
            self.projection = make_quant_linear(
                in_channels, in_channels_reduced, bias=True, quant_config=self.quant_config
            )
            self.projection_act = make_quant_relu(quant_config=self.quant_config)
            conv_in = in_channels_reduced
        else:
            self.projection = None
            self.projection_act = None
            conv_in = in_channels
        
        # First SAGE layer
        self.conv1 = BrevitasGraphSAGELayer(
            conv_in, hidden_channels, 
            root_weight=root_weight, 
            quant_config=self.quant_config
        )
        self.conv1_act = make_quant_relu(quant_config=self.quant_config)
        
        # Second SAGE layer
        self.conv2 = BrevitasGraphSAGELayer(
            hidden_channels, out_channels,
            root_weight=root_weight,
            quant_config=self.quant_config
        )
    
    def forward(self, x, edge_index):
        """Forward pass through quantized GraphSAGE."""
        # No explicit input quantization - QuantLinear handles it
        
        # Optional projection
        if self.use_projection:
            x = self.projection(x)
            x = self.projection_act(x)
        
        # First SAGE layer
        x = self.conv1(x, edge_index)
        x = self.conv1_act(x)
        
        # Dropout
        if hasattr(x, 'value'):
            x_value = x.value
        else:
            x_value = x
        x = torch.nn.functional.dropout(x_value, p=self.dropout, training=self.training)
        
        # Second SAGE layer
        x = self.conv2(x, edge_index)
        
        # Return regular tensor
        if hasattr(x, 'value'):
            x = x.value
        
        return x


def load_from_float_model(brevitas_model: BrevitasReducedGraphSAGE, float_model: nn.Module):
    """
    Load weights from a float ReducedGraphSAGE model into a Brevitas model.
    
    Args:
        brevitas_model: Target Brevitas quantized model
        float_model: Source float model (ReducedGraphSAGE)
    """
    float_model.eval()
    brevitas_model.eval()
    
    with torch.no_grad():
        # Copy projection weights if present
        if brevitas_model.use_projection and hasattr(float_model, 'projection'):
            brevitas_model.projection.weight.data.copy_(float_model.projection.weight.data)
            if float_model.projection.bias is not None:
                brevitas_model.projection.bias.data.copy_(float_model.projection.bias.data)
        
        # Copy conv1 weights (PyG SAGEConv has lin_l for neighbor, lin_r for root)
        if hasattr(float_model.conv1, 'lin_l'):
            brevitas_model.conv1.lin_neighbor.weight.data.copy_(float_model.conv1.lin_l.weight.data)
            if float_model.conv1.lin_l.bias is not None:
                brevitas_model.conv1.lin_neighbor.bias.data.copy_(float_model.conv1.lin_l.bias.data)
            
            if brevitas_model.root_weight and hasattr(float_model.conv1, 'lin_r') and float_model.conv1.lin_r is not None:
                brevitas_model.conv1.lin_root.weight.data.copy_(float_model.conv1.lin_r.weight.data)
        
        # Copy conv2 weights
        if hasattr(float_model.conv2, 'lin_l'):
            brevitas_model.conv2.lin_neighbor.weight.data.copy_(float_model.conv2.lin_l.weight.data)
            if float_model.conv2.lin_l.bias is not None:
                brevitas_model.conv2.lin_neighbor.bias.data.copy_(float_model.conv2.lin_l.bias.data)
            
            if brevitas_model.root_weight and hasattr(float_model.conv2, 'lin_r') and float_model.conv2.lin_r is not None:
                brevitas_model.conv2.lin_root.weight.data.copy_(float_model.conv2.lin_r.weight.data)
    
    print("✓ Loaded weights from float model into Brevitas model")


if __name__ == "__main__":
    print(f"Brevitas available: {BREVITAS_AVAILABLE}")
    if BREVITAS_AVAILABLE:
        cfg = BrevitasQuantConfig(weight_bit_width=8, act_bit_width=8, return_quant_tensor=True)
        
        # Test layer creation
        qid = make_quant_identity(cfg)
        qlin = make_quant_linear(16, 32, quant_config=cfg)
        qrelu = make_quant_relu(cfg)
        
        x = torch.randn(4, 16)
        y = qrelu(qlin(qid(x)))
        y_tensor = y.value if hasattr(y, "value") else y
        print("Layer test OK:", x.shape, "->", y_tensor.shape)
        
        # Test model creation
        model = BrevitasReducedGraphSAGE(
            in_channels=32,
            in_channels_reduced=16,
            hidden_channels=12,
            out_channels=4,
            dropout=0.0,
            use_projection=True,
            root_weight=False,
            quant_config=cfg
        )
        
        num_nodes = 8
        num_edges = 16
        x = torch.randn(num_nodes, 32)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)
        
        print("Model test OK:", x.shape, "->", out.shape)
