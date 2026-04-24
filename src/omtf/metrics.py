"""
OMTF evaluation metrics.

All evaluation tables must include:
> Full trigger rate requires minimum-bias full events through GMT emulation.
> B4 gives a background-acceptance proxy only, not absolute CMS L1 rate.

Stub/node metrics (all models)
------------------------------
1. stub_recovery_efficiency  — fraction of signal stubs classified as signal
2. node_auc                  — ROC-AUC of node signal/noise classifier
3. candidate_recovery_proxy  — gen-muon recovery via stub votes (node-only proxy)

Window/trigger metrics (models with candidate_logits)
------------------------------------------------------
4. trigger_efficiency_vs_pt  — efficiency vs gen pT threshold scan
5. candidate_recovery_slot   — slot k fires for gen muon k (slot-ordered models)
6. background_acceptance      — B4 false-positive window rate (proxy)
7. dimuon_separation_efficiency — both slots fire in 2-muon windows

pT regression (per-slot models with calibrated pt_pred)
-------------------------------------------------------
8. pt_resolution_slots       — (pred_pt - true_pt)/true_pt summary per slot

Edge metrics
------------
9. edge_auc                  — ROC-AUC of edge same-track classifier
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


# --------------------------------------------------------------------------
# 5. Slot-based candidate recovery  (slot-ordered models: slot_model)
# --------------------------------------------------------------------------

def candidate_recovery_slot(
    candidate_logits: Tensor,  # (B, K)
    gen_pt: Tensor,            # (B, K) — 0.0 if gen muon absent at slot k
    threshold: float = 0.0,
) -> float:
    """
    Fraction of occupied gen-muon slots (gen_pt[b,k] > 0) where the
    corresponding slot fires (candidate_logit[b,k] > threshold).

    Uses fixed-index assignment: slot k ↔ gen muon track_id=k+1.
    """
    occupied = (gen_pt > 0)
    if occupied.sum() == 0:
        return float("nan")
    fired = (candidate_logits > threshold) & occupied
    return float(fired.float().sum() / occupied.float().sum())


# --------------------------------------------------------------------------
# 4. Trigger efficiency vs gen pT threshold
# --------------------------------------------------------------------------

_DEFAULT_PT_THRESHOLDS = [0.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 30.0, 50.0]


def trigger_efficiency_vs_pt(
    candidate_logits: Tensor,   # (B, K)
    gen_pt: Tensor,             # (B, K) — 0.0 if absent
    pt_thresholds: list[float] | None = None,
    score_threshold: float = 0.0,
) -> dict:
    """
    Window-level trigger efficiency as a function of gen pT threshold.

    A window "triggers" if any slot fires (max_k candidate_logit > score_threshold).
    For each pT cut, efficiency is computed over windows where at least one gen
    muon has pT > cut.

    Returns {'thresholds': [...], 'efficiency': [...], 'n_windows': [...]}
    """
    if pt_thresholds is None:
        pt_thresholds = _DEFAULT_PT_THRESHOLDS

    # Window fires if any slot exceeds score threshold
    triggers = (candidate_logits > score_threshold).any(dim=-1)    # (B,)
    # Max gen pT per window
    max_gen_pt = gen_pt.max(dim=-1).values                         # (B,)

    thresholds_out, efficiencies, n_windows = [], [], []
    for pt_cut in pt_thresholds:
        mask = max_gen_pt > pt_cut
        n = int(mask.sum())
        if n == 0:
            eff = float("nan")
        else:
            eff = float(triggers[mask].float().mean())
        thresholds_out.append(float(pt_cut))
        efficiencies.append(eff)
        n_windows.append(n)

    return {
        "thresholds":  thresholds_out,
        "efficiency":  efficiencies,
        "n_windows":   n_windows,
    }


# --------------------------------------------------------------------------
# 6. Background acceptance — slot-based  (B4)
# --------------------------------------------------------------------------

def background_acceptance_slots(
    candidate_logits: Tensor,  # (B, K)
    threshold: float = 0.0,
) -> float:
    """
    Fraction of B4 processor windows where at least one slot fires.
    Uses candidate_logits from slot-capable models.
    """
    fires = (candidate_logits > threshold).any(dim=-1)  # (B,)
    return float(fires.float().mean())


# --------------------------------------------------------------------------
# 7. Dimuon / multi-muon separation efficiency
# --------------------------------------------------------------------------

def dimuon_separation_efficiency(
    candidate_logits: Tensor,  # (B, K)
    gen_pt: Tensor,            # (B, K)
    n_required: int = 2,
    threshold: float = 0.0,
) -> float:
    """
    Among windows where >= n_required gen muons are present (gen_pt[b,k] > 0),
    fraction where at least n_required distinct slots fire.

    Uses fixed-index assignment: slot k ↔ gen muon k+1.
    For n_required=2: both slot 0 and slot 1 must fire (for 2-muon windows).
    """
    n_gen_per_window = (gen_pt > 0).sum(dim=-1)        # (B,)
    multi_mask = n_gen_per_window >= n_required         # (B,) windows with enough gen muons

    if multi_mask.sum() == 0:
        return float("nan")

    fires_per_slot = candidate_logits > threshold       # (B, K)
    # For fixed-index: we need slots 0..n_required-1 all to fire
    # in windows that have gen muons at those indices
    occupied = gen_pt > 0                              # (B, K)
    all_recovered = (fires_per_slot & occupied).sum(dim=-1) >= n_required   # (B,)

    return float(all_recovered[multi_mask].float().mean())


# --------------------------------------------------------------------------
# 8. Per-slot pT resolution
# --------------------------------------------------------------------------

def dxy_resolution_slots(
    dxy_pred: Tensor,   # (B, K) — predicted dXY in cm
    gen_pt: Tensor,     # (B, K) — used to identify occupied slots (gen_pt > 0)
    gen_dxy: Tensor,    # (B, K) — true dXY in cm
) -> dict:
    """
    dXY resolution for occupied slots.

    Returns: mae (cm), median (cm), sigma68 (half-width of central 68%),
             n_muons, mae_displaced (|gen_dxy| > 0.1 cm only).
    """
    occupied = gen_pt > 0
    if occupied.sum() == 0:
        return {"mae": float("nan"), "median": float("nan"),
                "sigma68": float("nan"), "n_muons": 0, "mae_displaced": float("nan")}

    pred = dxy_pred[occupied].cpu().numpy()
    true = gen_dxy[occupied].cpu().numpy()
    err  = pred - true

    displaced = np.abs(true) > 0.1  # |dXY| > 1 mm — non-prompt
    mae_disp  = float(np.mean(np.abs(err[displaced]))) if displaced.sum() > 0 else float("nan")

    q16, q84 = np.percentile(err, [16, 84])
    return {
        "mae":           float(np.mean(np.abs(err))),
        "median":        float(np.median(err)),
        "sigma68":       float((q84 - q16) / 2),
        "n_muons":       int(occupied.sum()),
        "mae_displaced": mae_disp,
    }


def pt_resolution_slots(
    pt_pred: Tensor,    # (B, K) — raw log(pT) predictions
    gen_pt: Tensor,     # (B, K) — true pT in GeV (0 if absent)
) -> dict:
    """
    pT resolution for occupied slots (gen_pt > 0).

    pred_pt = exp(pt_pred).  Resolution = (pred_pt - true_pt) / true_pt.

    Returns: mae (GeV), median_rel, sigma68 (half-width of central 68%),
             n_muons (int).
    """
    occupied = gen_pt > 0
    if occupied.sum() == 0:
        return {"mae": float("nan"), "median_rel": float("nan"),
                "sigma68": float("nan"), "n_muons": 0}

    pred_pt = torch.exp(pt_pred[occupied]).cpu().numpy()
    true    = gen_pt[occupied].cpu().numpy()

    rel = (pred_pt - true) / np.clip(true, 1e-3, None)
    mae = float(np.mean(np.abs(pred_pt - true)))
    med = float(np.median(rel))
    q16, q84 = np.percentile(rel, [16, 84])

    return {
        "mae":        mae,
        "median_rel": med,
        "sigma68":    float((q84 - q16) / 2),
        "n_muons":    int(occupied.sum()),
    }
