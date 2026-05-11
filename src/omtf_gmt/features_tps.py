"""
Node feature schema for the GMT-visible-stub branch — TPS source (Phase B4).

14-dimensional feature vector per stub, same length as the KMTF schema so that
the same model architecture (EdgeCompat, etc.) can be used without code changes.
Features at indices 7–10 differ from the KMTF schema; do not mix checkpoints.

  Continuous (7):
    0  phi_rel          processor-centred phi = offlineCoord1 − proc_centre  [rad]
    1  coord2_feature   offlineCoord2 (second phi for RPC-matched stubs; 0 for CSC-only)
    2  eta1             offlineEta1                                            [1]
    3  eta2             offlineEta2 (masked to 0 if absent)                   [1]
    4  quality_norm     quality / 3  (TPS quality range: 1–3)                 [0,1]
    5  eta_quality_norm etaQuality / 3  (TPS etaQuality range: 0–3)          [0,1]
    6  bx_norm          bxNum / 3  (in-time BX=0 → 0.0)                     [−1,1]

  Ordinal-categorical as normalised int (3):
    7  tf_layer_norm    tfLayer / 4  (TPS: 0–4 → 0.0–1.0)
    8  depth_region_norm (depthRegion − 1) / 3  (stations 1–4 → 0.0–1.0)
    9  stub_type_norm   float(stubType)  (binary {0,1}; 1 = RPC-matched or hybrid)

  Mask (1):
   10  has_coord2       1 if offlineCoord2 is non-zero (RPC match present)

  Domain flags (3):
   11  stub_in_overlap  1.0 if 0.83 ≤ |offlineEta1| ≤ 1.24, else 0.0
   12  abs_eta          |offlineEta1|                                          [1]
   13  eta_dist_to_overlap  signed distance from |offlineEta1| to [0.83, 1.24]:
                              < 0  stub is below the overlap (barrel side)
                              = 0  stub is inside the overlap
                              > 0  stub is above the overlap (endcap side)

  Note: the KMTF schema has has_eta2 at index 7 and has_coord2 at index 8.
  TPS replaces those with tf_layer_norm, depth_region_norm, stub_type_norm, and
  has_coord2 at indices 7–10.  coord2_feature occupies index 1 in both schemas,
  but its physical meaning differs (DT phiBend for KMTF; second phi for TPS).

Total: N_FEATURES_TPS = 14.
"""

from __future__ import annotations

import numpy as np

# Re-export overlap bounds from the KMTF features module for consistency.
from omtf_gmt.features import ETA_OVERLAP_LO, ETA_OVERLAP_HI

# ----- index constants -----------------------------------------------------

F_TPS_PHI_REL          = 0
F_TPS_COORD2           = 1
F_TPS_ETA1             = 2
F_TPS_ETA2             = 3
F_TPS_QUALITY          = 4
F_TPS_ETA_QUALITY      = 5
F_TPS_BX               = 6
F_TPS_TF_LAYER         = 7
F_TPS_DEPTH_REGION     = 8
F_TPS_STUB_TYPE        = 9
F_TPS_HAS_COORD2       = 10
F_TPS_IN_OVERLAP       = 11
F_TPS_ABS_ETA          = 12
F_TPS_ETA_DIST_OVERLAP = 13

N_FEATURES_TPS: int = 14

FEATURE_NAMES_TPS: list[str] = [
    "phi_rel",
    "coord2_feature",
    "eta1",
    "eta2",
    "quality_norm",
    "eta_quality_norm",
    "bx_norm",
    "tf_layer_norm",
    "depth_region_norm",
    "stub_type_norm",
    "has_coord2",
    "stub_in_overlap",
    "abs_eta",
    "eta_dist_to_overlap",
]

# ----- per-stub feature builder --------------------------------------------

def build_node_features_tps(
    phi_rel:      np.ndarray,   # (N,) processor-centred phi [rad]
    coord2:       np.ndarray,   # (N,) offlineCoord2 (second phi or 0)
    eta1:         np.ndarray,   # (N,) offlineEta1
    eta2:         np.ndarray,   # (N,) offlineEta2
    quality:      np.ndarray,   # (N,) raw quality int (TPS range: 1–3)
    eta_quality:  np.ndarray,   # (N,) raw etaQuality int (TPS range: 0–3)
    bx:           np.ndarray,   # (N,) bxNum
    tf_layer:     np.ndarray,   # (N,) tfLayer int (TPS range: 0–4)
    depth_region: np.ndarray,   # (N,) depthRegion int (range: 1–4)
    stub_type:    np.ndarray,   # (N,) stubType int (binary {0,1})
) -> np.ndarray:
    """Return feature matrix of shape (N, N_FEATURES_TPS) in float32."""
    N = len(phi_rel)
    X = np.zeros((N, N_FEATURES_TPS), dtype=np.float32)

    has_coord2 = (coord2 != 0).astype(np.float32)
    abs_e      = np.abs(eta1).astype(np.float32)
    in_ov      = (abs_e >= ETA_OVERLAP_LO) & (abs_e <= ETA_OVERLAP_HI)
    eta_dist   = np.where(abs_e < ETA_OVERLAP_LO,
                          abs_e - ETA_OVERLAP_LO,
                          np.where(in_ov, 0.0, abs_e - ETA_OVERLAP_HI)).astype(np.float32)

    X[:, F_TPS_PHI_REL]          = phi_rel.astype(np.float32)
    X[:, F_TPS_COORD2]            = coord2.astype(np.float32)
    X[:, F_TPS_ETA1]              = eta1.astype(np.float32)
    X[:, F_TPS_ETA2]              = (np.asarray(eta2, dtype=np.float32) *
                                      (np.asarray(eta2) != 0).astype(np.float32))
    X[:, F_TPS_QUALITY]           = np.clip(quality, 0, 3).astype(np.float32) / 3.0
    X[:, F_TPS_ETA_QUALITY]       = np.clip(eta_quality, 0, 3).astype(np.float32) / 3.0
    X[:, F_TPS_BX]                = np.clip(bx, -3, 3).astype(np.float32) / 3.0
    X[:, F_TPS_TF_LAYER]          = np.clip(tf_layer, 0, 4).astype(np.float32) / 4.0
    X[:, F_TPS_DEPTH_REGION]      = (np.clip(depth_region, 1, 4) - 1).astype(np.float32) / 3.0
    X[:, F_TPS_STUB_TYPE]         = np.clip(stub_type, 0, 1).astype(np.float32)
    X[:, F_TPS_HAS_COORD2]        = has_coord2
    X[:, F_TPS_IN_OVERLAP]        = in_ov.astype(np.float32)
    X[:, F_TPS_ABS_ETA]           = abs_e
    X[:, F_TPS_ETA_DIST_OVERLAP]  = eta_dist

    return X
