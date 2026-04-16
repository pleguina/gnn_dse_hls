"""
OMTF loss functions.

Three task heads, all optional — pass None to skip a head's contribution.

  node_bce  : per-stub signal/noise classification (trackId != 0)
  edge_bce  : per-pair same-track / cross-track classification
  pt_reg    : pT regression for signal stubs (MSE on log scale)

All losses ignore ambiguous entries when an ambiguous mask is supplied.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def node_bce_loss(
    logits: Tensor,
    labels: Tensor,
    valid_mask: Tensor | None = None,
    ambiguous_mask: Tensor | None = None,
    pos_weight: float | None = None,
) -> Tensor:
    """
    Binary cross-entropy over node (stub) signal/noise labels.

    Parameters
    ----------
    logits        : (N,) raw logits
    labels        : (N,) float — 1.0 for signal stub (trackId != 0), else 0.0
    valid_mask    : (N,) bool — True for real (non-padded) stubs
    ambiguous_mask: (N,) bool — True for stubs with the ambiguous flag set
    pos_weight    : optional scalar weight for positive class

    Returns
    -------
    Scalar loss tensor.
    """
    keep = _build_keep(valid_mask, ambiguous_mask, len(logits), logits.device)
    if keep.sum() == 0:
        return logits.sum() * 0.0

    pw = (torch.tensor([pos_weight], device=logits.device) if pos_weight is not None
          else None)
    return F.binary_cross_entropy_with_logits(
        logits[keep], labels[keep], pos_weight=pw
    )


def edge_bce_loss(
    logits: Tensor,
    labels: Tensor,
    ambiguous_mask: Tensor | None = None,
    pos_weight: float | None = None,
) -> Tensor:
    """
    Binary cross-entropy over edge (pair) same-track labels.

    Parameters
    ----------
    logits        : (E,) raw logits
    labels        : (E,) float — 1.0 for same-track pair
    ambiguous_mask: (E,) bool — True if either endpoint is ambiguous
    pos_weight    : optional scalar weight for positive class

    Returns
    -------
    Scalar loss tensor.
    """
    keep = _build_keep(None, ambiguous_mask, len(logits), logits.device)
    if keep.sum() == 0:
        return logits.sum() * 0.0

    pw = (torch.tensor([pos_weight], device=logits.device) if pos_weight is not None
          else None)
    return F.binary_cross_entropy_with_logits(
        logits[keep], labels[keep], pos_weight=pw
    )


def pt_regression_loss(
    pred_log_pt: Tensor,
    true_pt: Tensor,
    valid_mask: Tensor | None = None,
    signal_mask: Tensor | None = None,
) -> Tensor:
    """
    MSE loss on log(pT) for signal stubs.

    Parameters
    ----------
    pred_log_pt : (N,) predicted log pT
    true_pt     : (N,) true pT (GeV) — loss converts to log scale internally
    valid_mask  : (N,) bool — non-padded stubs
    signal_mask : (N,) bool — stubs with trackId != 0

    Returns
    -------
    Scalar loss tensor.
    """
    keep_masks = []
    if valid_mask is not None:
        keep_masks.append(valid_mask)
    if signal_mask is not None:
        keep_masks.append(signal_mask)

    if keep_masks:
        keep = keep_masks[0]
        for m in keep_masks[1:]:
            keep = keep & m
    else:
        keep = torch.ones(len(pred_log_pt), dtype=torch.bool,
                          device=pred_log_pt.device)

    if keep.sum() == 0:
        return pred_log_pt.sum() * 0.0

    log_true = torch.log(true_pt[keep].clamp(min=1e-3))
    return F.mse_loss(pred_log_pt[keep], log_true)


def combined_loss(
    node_logits: Tensor | None,
    node_labels: Tensor | None,
    edge_logits: Tensor | None,
    edge_labels: Tensor | None,
    pt_pred: Tensor | None = None,
    pt_true: Tensor | None = None,
    valid_mask: Tensor | None = None,
    node_ambig: Tensor | None = None,
    edge_ambig: Tensor | None = None,
    signal_mask: Tensor | None = None,
    w_node: float = 1.0,
    w_edge: float = 1.0,
    w_pt: float = 0.1,
    node_pos_weight: float | None = None,
    edge_pos_weight: float | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    Weighted sum of all active loss heads.

    Returns (total_loss, {head_name: scalar_loss}) for logging.
    """
    parts: dict[str, Tensor] = {}
    total = torch.zeros(1, device=_any_device(
        node_logits, edge_logits, pt_pred))

    if node_logits is not None and node_labels is not None:
        l = node_bce_loss(node_logits, node_labels, valid_mask, node_ambig,
                          node_pos_weight)
        parts["node_bce"] = l
        total = total + w_node * l

    if edge_logits is not None and edge_labels is not None:
        l = edge_bce_loss(edge_logits, edge_labels, edge_ambig, edge_pos_weight)
        parts["edge_bce"] = l
        total = total + w_edge * l

    if pt_pred is not None and pt_true is not None:
        l = pt_regression_loss(pt_pred, pt_true, valid_mask, signal_mask)
        parts["pt_reg"] = l
        total = total + w_pt * l

    parts["total"] = total.squeeze()
    return total.squeeze(), parts


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_keep(
    valid_mask: Tensor | None,
    ambiguous_mask: Tensor | None,
    n: int,
    device: torch.device,
) -> Tensor:
    keep = torch.ones(n, dtype=torch.bool, device=device)
    if valid_mask is not None:
        keep = keep & valid_mask
    if ambiguous_mask is not None:
        keep = keep & (~ambiguous_mask)
    return keep


def _any_device(*tensors) -> torch.device:
    for t in tensors:
        if t is not None:
            return t.device
    return torch.device("cpu")
