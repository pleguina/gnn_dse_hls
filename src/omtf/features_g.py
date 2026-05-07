"""
Node feature schema for the OMTF-internal G-dataset study.

Uses reg_stub_* branches directly from OMTFAllInputTree — no NanoAOD stub
collection, no truth transfer.  Native OMTF processor-window representation.

11-dimensional feature vector per stub.

  Physics coordinates (4):
    0  phi_norm           reg_stub_phiHw  × (2π/5400)   [rad, ~−0.5 to +2.2]
    1  phiB_norm          reg_stub_phiBHw × (2π/5400)   [rad, ~−0.7 to +0.8]
    2  eta_hw_norm        reg_stub_etaHw  / 127.0        [0, 1]
    3  r_norm             reg_stub_r      / 800.0        [cm/800]

  Stub quality / type (4):
    4  quality_norm       reg_stub_quality / 15.0
    5  type_norm          reg_stub_type    / 15.0
    6  layer_norm         reg_stub_layer   / 17.0        [0, 1]
    7  bx_norm            reg_stub_bx      / 3.0

  Overlap-domain flags (3):
    8  stub_in_overlap    1.0 if 0.83 ≤ |eta_phys| ≤ 1.24, else 0.0
    9  abs_eta            |reg_stub_etaHw × 0.010875|
   10  eta_dist_to_overlap  signed distance from |eta_phys| to [0.83, 1.24]:
                              < 0  barrel side
                              = 0  inside overlap
                              > 0  endcap side

Normalization notes
-------------------
  phi_norm/phiB_norm use OMTF_PHI_SCALE = 2π/5400; values are not zero-centred
  because reg_stub_phiHw is processor-local but not symmetric around 0 in the
  3-processor production configuration (observed range: ~−0.5 to +2.2 rad).

  quality/type are normalised by 15 (same convention as KMTF) even though
  observed maxima in G datasets are 6 and 9 respectively.

  bx is identically 0 in current G datasets (in-time BX only).

N_FEATURES_G = 11.
"""

from __future__ import annotations

import numpy as np

ETA_OVERLAP_LO: float = 0.83
ETA_OVERLAP_HI: float = 1.24

_OMTF_PHI_SCALE: float = 2.0 * np.pi / 5400.0
_ETA_HW_TO_PHY: float  = 0.010875   # etaHw × this = physics |eta|

# ---- index constants -------------------------------------------------------

F_PHI_NORM          = 0
F_PHIB_NORM         = 1
F_ETA_HW_NORM       = 2
F_R_NORM            = 3
F_QUALITY_NORM      = 4
F_TYPE_NORM         = 5
F_LAYER_NORM        = 6
F_BX_NORM           = 7
F_IN_OVERLAP        = 8
F_ABS_ETA           = 9
F_ETA_DIST_OVERLAP  = 10

N_FEATURES_G: int = 11

FEATURE_NAMES_G: list[str] = [
    "phi_norm",
    "phiB_norm",
    "eta_hw_norm",
    "r_norm",
    "quality_norm",
    "type_norm",
    "layer_norm",
    "bx_norm",
    "stub_in_overlap",
    "abs_eta",
    "eta_dist_to_overlap",
]


def build_node_features_g(
    phi_hw:  np.ndarray,   # (N,) reg_stub_phiHw  (processor-local HW units)
    phib_hw: np.ndarray,   # (N,) reg_stub_phiBHw
    eta_hw:  np.ndarray,   # (N,) reg_stub_etaHw
    r:       np.ndarray,   # (N,) reg_stub_r [cm]
    quality: np.ndarray,   # (N,) reg_stub_quality
    type_:   np.ndarray,   # (N,) reg_stub_type
    layer:   np.ndarray,   # (N,) reg_stub_layer
    bx:      np.ndarray,   # (N,) reg_stub_bx
) -> np.ndarray:
    """Return feature matrix of shape (N, N_FEATURES_G) in float32."""
    N = len(phi_hw)
    X = np.zeros((N, N_FEATURES_G), dtype=np.float32)

    abs_eta  = np.abs(eta_hw.astype(np.float32) * _ETA_HW_TO_PHY)
    in_ov    = (abs_eta >= ETA_OVERLAP_LO) & (abs_eta <= ETA_OVERLAP_HI)
    eta_dist = np.where(
        abs_eta < ETA_OVERLAP_LO,
        abs_eta - ETA_OVERLAP_LO,
        np.where(in_ov, 0.0, abs_eta - ETA_OVERLAP_HI),
    ).astype(np.float32)

    X[:, F_PHI_NORM]         = phi_hw.astype(np.float32)  * _OMTF_PHI_SCALE
    X[:, F_PHIB_NORM]        = phib_hw.astype(np.float32) * _OMTF_PHI_SCALE
    X[:, F_ETA_HW_NORM]      = np.clip(eta_hw, 0, 127).astype(np.float32) / 127.0
    X[:, F_R_NORM]           = np.clip(r,       0, 800).astype(np.float32) / 800.0
    X[:, F_QUALITY_NORM]     = np.clip(quality, 0,  15).astype(np.float32) / 15.0
    X[:, F_TYPE_NORM]        = np.clip(type_,   0,  15).astype(np.float32) / 15.0
    X[:, F_LAYER_NORM]       = np.clip(layer,   0,  17).astype(np.float32) / 17.0
    X[:, F_BX_NORM]          = np.clip(bx,     -3,   3).astype(np.float32) / 3.0
    X[:, F_IN_OVERLAP]       = in_ov.astype(np.float32)
    X[:, F_ABS_ETA]          = abs_eta
    X[:, F_ETA_DIST_OVERLAP] = eta_dist

    return X
