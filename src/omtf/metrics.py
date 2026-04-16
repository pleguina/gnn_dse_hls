"""
OMTF evaluation metrics — four canonical metrics from the migration plan.

All evaluation tables must include:
> Full trigger rate requires minimum-bias full events through GMT emulation.
> B4 gives a background-acceptance proxy only, not absolute CMS L1 rate.

Metrics
-------
1. stub_recovery_efficiency  — fraction of signal stubs correctly classified as signal
2. candidate_recovery_proxy  — fraction of gen-muons with ≥1 correctly classified stub
3. background_acceptance     — fraction of B4 windows with ≥1 stub above threshold
4. edge_auc                  — ROC-AUC of edge same-track classifier

Auxiliary
---------
- node_auc           — ROC-AUC of node signal/noise classifier
- pt_resolution      — (pred_pt - true_pt) / true_pt distribution summary
- pt_mae             — mean absolute error on pT (GeV)
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

try:
    from sklearn.metrics import roc_auc_score
    _SKLEARN = True
except ImportError:
    _SKLEARN = False


# --------------------------------------------------------------------------
# 1. Stub-to-track recovery efficiency
# --------------------------------------------------------------------------

def stub_recovery_efficiency(
    node_logits: Tensor,
    node_labels: Tensor,
    valid_mask: Tensor,
    threshold: float = 0.0,  # logit threshold (0.0 ≡ prob > 0.5)
) -> float:
    """
    Fraction of signal stubs (label=1, valid) classified as signal (logit > threshold).
    """
    valid = valid_mask.bool()
    signal = (node_labels > 0.5) & valid
    if signal.sum() == 0:
        return float("nan")
    pred_signal = node_logits[signal] > threshold
    return float(pred_signal.float().mean().item())


# --------------------------------------------------------------------------
# 2. Candidate recovery proxy
# --------------------------------------------------------------------------

def candidate_recovery_proxy(
    node_logits: Tensor,          # (B, Nmax) or (N_total,) flat
    node_labels: Tensor,          # same shape
    track_id: Tensor,             # same shape — which muon each stub belongs to
    valid_mask: Tensor,           # same shape
    threshold: float = 0.0,
) -> float:
    """
    Fraction of gen-muons (unique non-zero track_id per entry) for which at
    least one of their stubs is correctly predicted as signal.

    Note: this is a stub-level proxy, not a full candidate-matching metric.
    Full candidate matching requires K-slot output heads (Stage 4+).
    """
    # Flatten
    flat_logits  = node_logits.reshape(-1)
    flat_labels  = node_labels.reshape(-1)
    flat_tid     = track_id.reshape(-1)
    flat_valid   = valid_mask.reshape(-1).bool()
    pred_signal  = (flat_logits > threshold) & flat_valid

    # Group by (batch, track_id) pairs — use a simple approximation:
    # iterate over all stubs and track which muons have at least one recovered stub
    tid_np  = flat_tid.cpu().numpy()
    pred_np = pred_signal.cpu().numpy()
    lbl_np  = flat_labels.cpu().numpy()
    valid_np = flat_valid.cpu().numpy()

    muon_found = {}   # track_id -> bool
    for tid, pred, lbl, v in zip(tid_np, pred_np, lbl_np, valid_np):
        if not v or tid == 0:
            continue
        if tid not in muon_found:
            muon_found[tid] = False
        if pred and lbl > 0.5:
            muon_found[tid] = True

    if not muon_found:
        return float("nan")
    return float(np.mean(list(muon_found.values())))


# --------------------------------------------------------------------------
# 3. Background acceptance proxy  (B4 dataset)
# --------------------------------------------------------------------------

def background_acceptance(
    node_logits: Tensor,   # (B, Nmax) — B4 batch
    valid_mask: Tensor,    # (B, Nmax)
    threshold: float = 0.0,
) -> float:
    """
    Fraction of processor windows (B4, noise-only) where the model predicts
    at least one stub as signal.

    This is a background-acceptance proxy, NOT absolute CMS L1 rate.
    Full trigger rate requires minimum-bias full events through GMT emulation.
    """
    valid = valid_mask.bool()
    # (B, Nmax) — True where model predicts signal
    pred = (node_logits > threshold) & valid
    # per-window: any signal predicted?
    any_signal = pred.any(dim=-1)  # (B,)
    return float(any_signal.float().mean().item())


# --------------------------------------------------------------------------
# 4. Edge AUC
# --------------------------------------------------------------------------

def edge_auc(
    edge_logits: Tensor | list,
    edge_labels: Tensor | list,
    edge_ambig: Tensor | list | None = None,
) -> float:
    """
    ROC-AUC of the edge same-track classifier, excluding ambiguous pairs.
    Accepts flat tensors or lists of per-sample tensors.
    """
    if not _SKLEARN:
        return float("nan")

    if isinstance(edge_logits, list):
        logits = torch.cat(edge_logits)
        labels = torch.cat(edge_labels)
        ambig  = torch.cat(edge_ambig) if edge_ambig is not None else None
    else:
        logits, labels, ambig = edge_logits, edge_labels, edge_ambig

    keep = torch.ones(len(logits), dtype=torch.bool)
    if ambig is not None:
        keep = keep & (~ambig.bool())

    l = labels[keep].cpu().numpy().astype(float)
    s = torch.sigmoid(logits[keep]).cpu().numpy().astype(float)

    if l.sum() == 0 or (1 - l).sum() == 0:
        return float("nan")
    return float(roc_auc_score(l, s))


# --------------------------------------------------------------------------
# Auxiliary: node AUC
# --------------------------------------------------------------------------

def node_auc(
    node_logits: Tensor,
    node_labels: Tensor,
    valid_mask: Tensor,
    ambiguous: Tensor | None = None,
) -> float:
    """ROC-AUC of the node signal/noise classifier on valid, unambiguous stubs."""
    if not _SKLEARN:
        return float("nan")

    keep = valid_mask.reshape(-1).bool()
    if ambiguous is not None:
        keep = keep & (~ambiguous.reshape(-1).bool())

    l = node_labels.reshape(-1)[keep].cpu().numpy().astype(float)
    s = torch.sigmoid(node_logits.reshape(-1)[keep]).cpu().numpy().astype(float)

    if l.sum() == 0 or (1 - l).sum() == 0:
        return float("nan")
    return float(roc_auc_score(l, s))


# --------------------------------------------------------------------------
# Auxiliary: pT resolution and MAE
# --------------------------------------------------------------------------

def pt_metrics(
    pred_log_pt: Tensor,
    true_pt: Tensor,
    valid_mask: Tensor,
    signal_mask: Tensor | None = None,
) -> dict[str, float]:
    """
    pT resolution statistics.

    Returns: mae, median_relative_error, sigma_68 (half-width of central 68%
    of (pred_pt - true_pt) / true_pt distribution).
    """
    keep = valid_mask.reshape(-1).bool()
    if signal_mask is not None:
        keep = keep & signal_mask.reshape(-1).bool()

    if keep.sum() == 0:
        return {"mae": float("nan"), "median_rel": float("nan"), "sigma68": float("nan")}

    pred_pt = torch.exp(pred_log_pt.reshape(-1)[keep]).cpu().numpy()
    true    = true_pt.reshape(-1)[keep].cpu().numpy()

    rel = (pred_pt - true) / np.clip(true, 1e-3, None)
    mae = float(np.mean(np.abs(pred_pt - true)))
    med = float(np.median(rel))
    q16, q84 = np.percentile(rel, [16, 84])
    sigma68 = float((q84 - q16) / 2)

    return {"mae": mae, "median_rel": med, "sigma68": sigma68}


# --------------------------------------------------------------------------
# Summary printer
# --------------------------------------------------------------------------

EVAL_DISCLAIMER = (
    "NOTE: Full trigger rate requires minimum-bias full events through GMT emulation.\n"
    "      B4 gives a background-acceptance proxy only, not absolute CMS L1 rate."
)


def print_metrics_table(results: dict, title: str = "Evaluation"):
    """
    Print a formatted metrics table.

    results: {dataset_name: {metric_name: value}}
    """
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

    all_keys = []
    for v in results.values():
        for k in v:
            if k not in all_keys:
                all_keys.append(k)

    col_w = max(len(k) for k in all_keys) + 2
    ds_w  = max(len(ds) for ds in results) + 2

    header = f"  {'Dataset':{ds_w}s}" + "".join(f"  {k:{col_w}s}" for k in all_keys)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for ds, vals in results.items():
        row = f"  {ds:{ds_w}s}"
        for k in all_keys:
            v = vals.get(k, float("nan"))
            if isinstance(v, float):
                row += f"  {v:{col_w}.4f}"
            else:
                row += f"  {str(v):{col_w}s}"
        print(row)

    print()
    print(EVAL_DISCLAIMER)
    print('='*60)
