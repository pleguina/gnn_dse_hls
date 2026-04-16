"""
OMTF derived pair features.

Produces per-pair (i, j) features from raw stub arrays for a single window.
All inputs are Python lists or 1-D numpy arrays of integers.

Raw stub features (per stub): phi, phiB, eta, r, quality, type, layer, bx
Derived pair features (per ordered pair i < j):
  delta_phi, delta_r, delta_r2, kappa_hat, abs_delta_eta, delta_bx, phiB_diff

kappa_hat guard: |dr^2| < 1e-6  => kappa_hat = 0.0  (avoids divide-by-zero)
delta_bx is 0 everywhere in current datasets; retained for completeness.
"""

from __future__ import annotations

import numpy as np

# Ordered list of raw stub features stored in the stub tensor columns.
# Column order in dataset.py must match this.
RAW_FEATURE_NAMES = ["phi", "phiB", "eta", "r", "quality", "type", "layer", "bx"]

# Pair features produced by compute_pair_features_np
PAIR_FEATURE_NAMES = [
    "delta_phi",
    "delta_r",
    "delta_r2",
    "kappa_hat",
    "abs_delta_eta",
    "delta_bx",
    "phiB_diff",
]


def compute_pair_features_np(
    phi: np.ndarray,
    phiB: np.ndarray,
    eta: np.ndarray,
    r: np.ndarray,
    bx: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute pair features for all valid (i < j) stub pairs in one window.

    Parameters
    ----------
    phi, phiB, eta, r, bx : (N,) arrays — raw stub features
    valid_mask             : (N,) bool array; if None, all stubs are valid

    Returns
    -------
    edge_index  : (2, E) int64 — row-major (i, j) with i < j over valid stubs
    edge_attr   : (E, 7) float32 — pair features in PAIR_FEATURE_NAMES order
    """
    n = len(phi)
    if valid_mask is None:
        valid_mask = np.ones(n, dtype=bool)

    valid_idx = np.where(valid_mask)[0]
    nv = len(valid_idx)

    if nv < 2:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, len(PAIR_FEATURE_NAMES)), dtype=np.float32)
        return edge_index, edge_attr

    # All pairs i < j among valid stubs
    ii, jj = np.triu_indices(nv, k=1)
    src = valid_idx[ii]
    dst = valid_idx[jj]

    dphi = phi[src].astype(np.float32) - phi[dst].astype(np.float32)
    dr   = r[src].astype(np.float32)   - r[dst].astype(np.float32)
    dr2  = dr ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(dr2 > 1e-6, 2.0 * dphi / dr2, 0.0).astype(np.float32)
    abs_deta = np.abs(eta[src].astype(np.float32) - eta[dst].astype(np.float32))
    dbx  = bx[src].astype(np.float32)  - bx[dst].astype(np.float32)
    dphiB = phiB[src].astype(np.float32) - phiB[dst].astype(np.float32)

    edge_attr = np.stack([dphi, dr, dr2, kappa, abs_deta, dbx, dphiB], axis=1)
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)
    return edge_index, edge_attr


def clip_kappa_hat(
    edge_attr: np.ndarray,
    clip_val: float = 500.0,
) -> np.ndarray:
    """
    Clip kappa_hat (column index 3) to [-clip_val, +clip_val].

    kappa_hat has long tails (±3000+) due to near-parallel stubs with small dr.
    A clip at ±500 retains the physically meaningful range while suppressing outliers.
    Adjust clip_val based on feature_ranges.json if a tighter range is needed.
    """
    out = edge_attr.copy()
    out[:, 3] = np.clip(out[:, 3], -clip_val, clip_val)
    return out


def build_edge_labels(
    track_id: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    ambiguous: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build binary edge labels: 1 if stub pair shares the same non-zero track_id.

    Parameters
    ----------
    track_id  : (N,) int array — per-stub track assignment (0 = noise)
    src, dst  : (E,) int arrays — edge endpoints
    ambiguous : (N,) uint8 array — per-stub ambiguity flag (optional)

    Returns
    -------
    edge_label      : (E,) float32 — 1.0 if same track, 0.0 otherwise
    edge_ambiguous  : (E,) bool — True if either stub is ambiguous
    """
    ti = track_id[src]
    tj = track_id[dst]
    # Same non-zero track_id => positive pair
    edge_label = ((ti == tj) & (ti != 0)).astype(np.float32)

    if ambiguous is not None:
        edge_ambiguous = (ambiguous[src].astype(bool) | ambiguous[dst].astype(bool))
    else:
        edge_ambiguous = np.zeros(len(src), dtype=bool)

    return edge_label, edge_ambiguous
