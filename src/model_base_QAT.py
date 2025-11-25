from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
from torch_geometric.utils import spmm

from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import (
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)


def make_activation_fake_quant(num_bits: int = 8) -> FakeQuantize:
    """
    Symmetric per-tensor fake quantizer for activations.
    Range: [-2^(num_bits-1), 2^(num_bits-1)-1] = [-128, 127] for 8 bits.
    """
    return FakeQuantize(
        observer=MovingAverageMinMaxObserver,
        quant_min=-(2 ** (num_bits - 1)),
        quant_max=(2 ** (num_bits - 1)) - 1,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False,
    )


def make_weight_fake_quant(num_bits: int = 8, per_channel: bool = True) -> FakeQuantize:
    """
    Symmetric fake quantizer for weights.
    By default, uses per-channel quantization on output channels (dim=0),
    which is standard for conv/linear weights.
    """
    if per_channel:
        observer = MovingAveragePerChannelMinMaxObserver
        qscheme = torch.per_channel_symmetric
    else:
        observer = MovingAverageMinMaxObserver
        qscheme = torch.per_tensor_symmetric

    return FakeQuantize(
        observer=observer,
        quant_min=-(2 ** (num_bits - 1)),
        quant_max=(2 ** (num_bits - 1)) - 1,
        dtype=torch.qint8,
        qscheme=qscheme,
        reduce_range=False,
        ch_axis=0,  # per output-channel
    )


class SAGEConvQAT(MessagePassing):
    r"""
    QAT-enabled GraphSAGE operator.

    Behavior matches PyG's SAGEConv (for `root_weight`, `project`, `normalize`,
    `aggr="mean"` etc.), but inserts fake quantization at key points:

        x_in  --(fake quant)--> propagate/aggregate
                      |
                      v
               agg_out --(fake quant)--> lin_l (with quantized weights)
                      |
               (+ lin_r(x_r) with quantized weights if root_weight=True)
                      |
                      v
                 out --(fake quant)--> (optional L2 normalize)

    When training with this module:
    - It simulates INT8 activations & weights, but still runs in float.
    - Gradients flow through Straight-Through Estimator (STE) in FakeQuantize.
    - After training, you can export:
        - quantized weights (INT8)
        - per-layer scales/zero-points
        - and implement the same arithmetic in HLS.
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
        num_bits_acts: int = 8,
        num_bits_weights: int = 8,
        per_channel_weights: bool = True,
        **kwargs,
    ):
        # --- Original SAGEConv fields ---
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if aggr == "lstm":
            kwargs.setdefault("aggr_kwargs", {})
            kwargs["aggr_kwargs"].setdefault("in_channels", in_channels[0])
            kwargs["aggr_kwargs"].setdefault("out_channels", in_channels[0])

        super().__init__(aggr, **kwargs)

        # Optional pre-project layer (same as SAGEConv)
        if self.project:
            if in_channels[0] <= 0:
                raise ValueError(
                    f"'{self.__class__.__name__}' does not support "
                    f"lazy initialization with `project=True`"
                )
            self.lin = Linear(in_channels[0], in_channels[0], bias=True)

        # Aggregation output channels
        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(in_channels[0])
        else:
            aggr_out_channels = in_channels[0]

        # Main GraphSAGE linear transformations (float weights)
        self.lin_l = Linear(aggr_out_channels, out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        # --- QAT components ---

        # Activations fake quantizers
        #   - input features before aggregation
        #   - aggregated features before lin_l
        #   - final output activations
        self.act_in_fake_quant = make_activation_fake_quant(num_bits_acts)
        self.act_agg_fake_quant = make_activation_fake_quant(num_bits_acts)
        self.act_out_fake_quant = make_activation_fake_quant(num_bits_acts)

        # Optional: fake quant after projection
        if self.project:
            self.act_proj_fake_quant = make_activation_fake_quant(num_bits_acts)
        else:
            self.act_proj_fake_quant = None

        # Weight fake quantizers for lin_l and optional lin_r
        self.w_l_fake_quant = make_weight_fake_quant(
            num_bits=num_bits_weights, per_channel=per_channel_weights
        )
        if self.root_weight:
            self.w_r_fake_quant = make_weight_fake_quant(
                num_bits=num_bits_weights, per_channel=per_channel_weights
            )
        else:
            self.w_r_fake_quant = None

        self.reset_parameters()

    def reset_parameters(self):
        # Same resets as SAGEConv + reset fake quant stats
        super().reset_parameters()
        if self.project:
            self.lin.reset_parameters()
            if self.act_proj_fake_quant is not None:
                self.act_proj_fake_quant.activation_post_process.reset_min_max_vals()
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

        # Reset observers for all fake quant modules
        self.act_in_fake_quant.activation_post_process.reset_min_max_vals()
        self.act_agg_fake_quant.activation_post_process.reset_min_max_vals()
        self.act_out_fake_quant.activation_post_process.reset_min_max_vals()
        self.w_l_fake_quant.activation_post_process.reset_min_max_vals()
        if self.w_r_fake_quant is not None:
            self.w_r_fake_quant.activation_post_process.reset_min_max_vals()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        size: Size = None,
    ) -> Tensor:
        """
        Forward pass with QAT:
        - Quantizes inputs (fake) before aggregation.
        - Aggregates neighbors using MessagePassing.
        - Quantizes aggregated features (fake).
        - Applies lin_l with quantized weights (fake).
        - Adds root contribution (lin_r) with quantized weights, if enabled.
        - Quantizes output (fake).
        - Optionally L2-normalizes.

        This is what you want to mirror in HLS later.
        """

        # Ensure x is a pair (x_src, x_tgt)
        if isinstance(x, Tensor):
            x = (x, x)

        # Optional projection in float, then fake-quant
        if self.project and hasattr(self, "lin"):
            # Projection uses float weights (you *could* quantize them too later)
            proj = self.lin(x[0]).relu()
            if self.act_proj_fake_quant is not None:
                proj = self.act_proj_fake_quant(proj)
            x = (proj, x[1])

        # Fake-quant input features (source & target)
        x_src = self.act_in_fake_quant(x[0])
        x_tgt = self.act_in_fake_quant(x[1]) if x[1] is not None else None
        x = (x_src, x_tgt)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)

        # Fake-quant aggregated features (this is what HW will store as INT8)
        out = self.act_agg_fake_quant(out)

        # --- Weight quantization and linear transform (left branch) ---
        # Quantize lin_l weights (and biases if desired; typically biases stay fp32)
        w_l_q = self.w_l_fake_quant(self.lin_l.weight)
        b_l = self.lin_l.bias  # kept as float in QAT; later you convert to int32 domain

        # Perform linear: out = out @ W_l^T + b_l
        out = F.linear(out, w_l_q, b_l)

        # --- Root contribution (right branch) if root_weight=True ---
        x_r = x[1]
        if self.root_weight and x_r is not None:
            # Optionally quantize root activations (here we reuse act_in_fake_quant)
            x_r_q = self.act_in_fake_quant(x_r)

            # Quantize lin_r weights
            w_r_q = self.w_r_fake_quant(self.lin_r.weight)
            # lin_r has no bias in SAGEConv
            out = out + F.linear(x_r_q, w_r_q, None)

        # Fake-quant final output activations (this is what next layer will see as INT8)
        out = self.act_out_fake_quant(out)

        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        # Same as SAGEConv: message is just neighbor features
        return x_j

    def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
        # Same behavior as SAGEConv
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, aggr={self.aggr}, "
            f"root_weight={self.root_weight}, project={self.project})"
        )
