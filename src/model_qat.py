"""
QAT-enabled GraphSAGE model for hardware-aware quantization training.
Uses SAGEConvQAT layer from model_base_QAT.py for fake quantization during training.
"""

import torch
import torch.nn.functional as F
from model_base_QAT import SAGEConvQAT


class ReducedGraphSAGEQAT(torch.nn.Module):
    """
    QAT-enabled reduced GraphSAGE model for FPGA implementation.
    Uses smaller feature dimensions with quantization-aware training.

    Architecture:
    - Linear projection (optional, for feature reduction)
    - SAGEConvQAT(F_in_reduced, F_hidden) with fake quantization
    - ReLU
    - Dropout (only during training)
    - SAGEConvQAT(F_hidden, F_out) with fake quantization

    This model trains with fake quantization to simulate INT8 arithmetic,
    allowing weights and activations to adapt to quantization constraints.

    Args:
        in_channels: Input feature dimension (1433 for Cora)
        in_channels_reduced: Reduced feature dimension after projection (e.g., 16)
        hidden_channels: Hidden layer dimension (e.g., 24)
        out_channels: Output dimension / number of classes (e.g., 7)
        dropout: Dropout rate during training
        use_projection: Whether to use projection layer
        root_weight: If False, HLS-compatible (W_l * h_agg + b_l only)
        num_bits_acts: Bit width for activation quantization (default 8)
        num_bits_weights: Bit width for weight quantization (default 8)
    """

    def __init__(
        self,
        in_channels,
        in_channels_reduced,
        hidden_channels,
        out_channels,
        dropout=0.5,
        use_projection=True,
        root_weight=False,  # HLS-compatible by default
        num_bits_acts=8,
        num_bits_weights=8,
    ):
        super(ReducedGraphSAGEQAT, self).__init__()

        self.use_projection = use_projection
        self.root_weight = root_weight
        self.dropout = dropout

        # Optional projection layer (stays in float, can be quantized separately)
        if use_projection:
            self.projection = torch.nn.Linear(in_channels, in_channels_reduced)
            conv_in = in_channels_reduced
        else:
            conv_in = in_channels

        # QAT-enabled SAGE layers
        self.conv1 = SAGEConvQAT(
            in_channels=conv_in,
            out_channels=hidden_channels,
            aggr='mean',
            normalize=False,
            root_weight=root_weight,
            project=False,
            bias=True,
            num_bits_acts=num_bits_acts,
            num_bits_weights=num_bits_weights,
        )

        self.conv2 = SAGEConvQAT(
            in_channels=hidden_channels,
            out_channels=out_channels,
            aggr='mean',
            normalize=False,
            root_weight=root_weight,
            project=False,
            bias=True,
            num_bits_acts=num_bits_acts,
            num_bits_weights=num_bits_weights,
        )

    def forward(self, x, edge_index):
        """Forward pass with QAT fake quantization."""
        # Optional projection for feature reduction
        if self.use_projection:
            x = self.projection(x)
            x = F.relu(x)

        # First SAGE layer with QAT
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second SAGE layer with QAT
        x = self.conv2(x, edge_index)

        return x

    def enable_fake_quant(self):
        """Enable fake quantization (for training and calibration)."""
        for module in self.modules():
            if hasattr(module, 'act_in_fake_quant'):
                module.act_in_fake_quant.enable_fake_quant()
                module.act_agg_fake_quant.enable_fake_quant()
                module.act_out_fake_quant.enable_fake_quant()
                module.w_l_fake_quant.enable_fake_quant()
                if module.w_r_fake_quant is not None:
                    module.w_r_fake_quant.enable_fake_quant()

    def disable_fake_quant(self):
        """Disable fake quantization (to measure float accuracy)."""
        for module in self.modules():
            if hasattr(module, 'act_in_fake_quant'):
                module.act_in_fake_quant.disable_fake_quant()
                module.act_agg_fake_quant.disable_fake_quant()
                module.act_out_fake_quant.disable_fake_quant()
                module.w_l_fake_quant.disable_fake_quant()
                if module.w_r_fake_quant is not None:
                    module.w_r_fake_quant.disable_fake_quant()

    def enable_observer(self):
        """Enable observers to collect statistics."""
        for module in self.modules():
            if hasattr(module, 'act_in_fake_quant'):
                module.act_in_fake_quant.enable_observer()
                module.act_agg_fake_quant.enable_observer()
                module.act_out_fake_quant.enable_observer()
                module.w_l_fake_quant.enable_observer()
                if module.w_r_fake_quant is not None:
                    module.w_r_fake_quant.enable_observer()

    def disable_observer(self):
        """Disable observers after calibration."""
        for module in self.modules():
            if hasattr(module, 'act_in_fake_quant'):
                module.act_in_fake_quant.disable_observer()
                module.act_agg_fake_quant.disable_observer()
                module.act_out_fake_quant.disable_observer()
                module.w_l_fake_quant.disable_observer()
                if module.w_r_fake_quant is not None:
                    module.w_r_fake_quant.disable_observer()
