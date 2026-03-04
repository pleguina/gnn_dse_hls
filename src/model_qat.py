"""
QAT-enabled GraphSAGE model for hardware-aware quantization training.
Uses SAGEConvQAT layer from model_base_QAT.py for fake quantization during training.
"""

import torch
import torch.nn.functional as F
import torch.ao.quantization as tq
from model_base_QAT import SAGEConvQAT


def make_act_fakequant(num_bits):
    """
    Create a fake-quantization module for activations.
    Uses symmetric quantization with MovingAverageMinMaxObserver.
    """
    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1
    return tq.FakeQuantize.with_args(
        observer=tq.MovingAverageMinMaxObserver,
        quant_min=qmin,
        quant_max=qmax,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False,
    )()


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
        self.num_bits_acts = num_bits_acts
        self.num_bits_weights = num_bits_weights

        # Optional projection layer with fake-quant for output
        if use_projection:
            self.projection = torch.nn.Linear(in_channels, in_channels_reduced)
            # Add fake-quant for projection output before it enters conv1
            self.proj_act_fake_quant = make_act_fakequant(num_bits_acts)
            conv_in = in_channels_reduced
        else:
            conv_in = in_channels
            self.proj_act_fake_quant = None

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
            # Quantize projection output before it enters conv1
            if self.proj_act_fake_quant is not None:
                x = self.proj_act_fake_quant(x)

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
            if isinstance(module, tq.FakeQuantize):
                module.enable_fake_quant()

    def disable_fake_quant(self):
        """Disable fake quantization (to measure float accuracy)."""
        for module in self.modules():
            if isinstance(module, tq.FakeQuantize):
                module.disable_fake_quant()

    def enable_observer(self):
        """Enable observers to collect statistics."""
        for module in self.modules():
            if isinstance(module, tq.FakeQuantize):
                module.enable_observer()

    def disable_observer(self):
        """Disable observers after calibration."""
        for module in self.modules():
            if isinstance(module, tq.FakeQuantize):
                module.disable_observer()

    def qat_calibrate_then_eval(self, data, num_calib_steps=50):
        """
        Proper post-training calibration for QAT fake-quant modules:
        - observers ON, fake-quant OFF to collect ranges
        - observers OFF, fake-quant ON to simulate quant inference
        
        This is the CORRECT way to evaluate a QAT model.
        
        Args:
            data: Graph data with x, edge_index, and test_mask
            num_calib_steps: Number of calibration passes
            
        Returns:
            test_accuracy: Accuracy on test set with proper quantization
        """
        self.eval()

        # 1) Calibration: observers ON, fake-quant OFF
        self.enable_observer()
        self.disable_fake_quant()

        with torch.no_grad():
            for _ in range(num_calib_steps):
                _ = self(data.x, data.edge_index)

        # 2) Quant inference: observers OFF, fake-quant ON
        self.disable_observer()
        self.enable_fake_quant()

        with torch.no_grad():
            out = self(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            test_correct = pred[data.test_mask] == data.y[data.test_mask]
            acc = int(test_correct.sum()) / int(data.test_mask.sum())

        return acc

    def eval_float_path(self, data):
        """
        Sanity check: evaluate model with fake-quant OFF.
        If this is low (~28%), the problem is in training/checkpoint.
        If this is good (70%+) but quant-path is bad, it's quantization placement/qparams.
        
        Args:
            data: Graph data with x, edge_index, and test_mask
            
        Returns:
            test_accuracy: Float-path accuracy
        """
        self.eval()
        self.disable_fake_quant()
        self.disable_observer()

        with torch.no_grad():
            out = self(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            test_correct = pred[data.test_mask] == data.y[data.test_mask]
            acc = int(test_correct.sum()) / int(data.test_mask.sum())

        return acc
