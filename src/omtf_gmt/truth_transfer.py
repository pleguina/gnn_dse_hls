"""
Offline truth transfer: OMTF internal stubs → GMT-visible KMTF barrel stubs.

For each KMTF barrel stub in an OMTF processor window, find the best-matching
OMTF stub and copy its trackId / ambiguous flag.

Matching strategy (composite key, in order of priority):
  1. BX must match (bxNum == reg_stub_bx).
  2. Phi proximity: |phi_kmtf − phi_omtf| < PHI_MATCH_THRESH.
  3. Station compatibility: depthRegion ≈ station inferred from OMTF layer.
  4. Tie-breaker: nearest phi.

If no OMTF stub passes (1)+(2), the KMTF stub is labelled as noise (trackId=0,
ambiguous=0, truth_source="unmatched").

truth_source encodes the confidence of the transferred label:
  "omtf_transfer"   — matched to an OMTF stub with trackId assignment
  "omtf_noise"      — matched to an OMTF stub with trackId=0 (confirmed noise)
  "unmatched"       — no OMTF stub found within matching window
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .regioning import omtf_phi_to_global_rad, angle_diff

# ----- tunable thresholds --------------------------------------------------

# After applying the phiZero(proc) correction, OMTF global phi and KMTF
# offlineCoord1 are in the same coordinate frame.  The residual spread is
# small (< 0.05 rad) for in-time barrel stubs; the threshold is set to
# 0.10 rad to accommodate calibration offsets and curvature at low pT.
PHI_MATCH_THRESH: float = 0.10    # rad
BX_MATCH: bool = True             # require BX agreement

# crude OMTF logic-layer → DT-station mapping for barrel layers
# OMTF layers 0-17; barrel DT layers roughly:
#   0,1 → MB1 (station 1), 2,3 → MB2, 4,5 → MB3, 6,7,8 → MB4 / transition
# Only used as a soft filter; phi is the primary criterion.
_LAYER_TO_STATION: dict[int, int] = {
    0: 1, 1: 1,
    2: 2, 3: 2,
    4: 3, 5: 3,
    6: 4, 7: 4, 8: 4,
    # CSC/RPC layers not relevant for barrel-first study
}


def _layer_to_station(layer: int) -> int:
    return _LAYER_TO_STATION.get(int(layer), -1)


# ----- per-window transfer -------------------------------------------------

@dataclass
class TransferResult:
    track_id:     np.ndarray   # (N_kmtf,) int8
    ambiguous:    np.ndarray   # (N_kmtf,) uint8
    truth_source: list[str]    # (N_kmtf,) "omtf_transfer" | "omtf_noise" | "unmatched"
    match_rate:   float        # fraction of KMTF stubs matched
    n_kmtf:       int
    n_matched:    int


def transfer(
    # --- KMTF barrel stubs (already filtered to processor window) ---
    kmtf_phi_rad:    np.ndarray,   # (N_kmtf,) offlineCoord1 in rad (global)
    kmtf_depth:      np.ndarray,   # (N_kmtf,) depthRegion (DT station 1-4)
    kmtf_bx:         np.ndarray,   # (N_kmtf,) bxNum
    # --- OMTF stubs in same (event, processor) window ---
    omtf_phi_hw:     np.ndarray,   # (M,) reg_stub_phiHw (PROCESSOR-LOCAL HW units)
    omtf_layer:      np.ndarray,   # (M,) reg_stub_layer (logic layer 0-17)
    omtf_bx:         np.ndarray,   # (M,) reg_stub_bx
    omtf_track_id:   np.ndarray,   # (M,) reg_stub_trackId (int8)
    omtf_ambiguous:  np.ndarray,   # (M,) reg_stub_ambiguous (uint8)
    omtf_proc:       int = 0,      # OMTF processor index (needed for phiZero correction)
) -> TransferResult:
    """Assign OMTF truth labels to KMTF barrel stubs."""
    N = len(kmtf_phi_rad)
    out_tid  = np.zeros(N, dtype=np.int8)
    out_amb  = np.zeros(N, dtype=np.uint8)
    out_src  = ["unmatched"] * N

    if len(omtf_phi_hw) == 0:
        return TransferResult(out_tid, out_amb, out_src, 0.0, N, 0)

    # Convert processor-local phiHw to global phi using phiZero correction.
    # omtf_proc is passed so the converter can add the right phiZero(proc).
    omtf_phi_rad = omtf_phi_to_global_rad(np.asarray(omtf_phi_hw, dtype=np.float64), omtf_proc)
    omtf_station = np.array([_layer_to_station(l) for l in omtf_layer], dtype=np.int8)

    n_matched = 0
    for i in range(N):
        k_phi  = float(kmtf_phi_rad[i])
        k_dep  = int(kmtf_depth[i])
        k_bx   = int(kmtf_bx[i])

        # step 1: BX filter
        if BX_MATCH:
            bx_ok = np.asarray(omtf_bx, dtype=np.int8) == k_bx
        else:
            bx_ok = np.ones(len(omtf_bx), dtype=bool)

        # step 2: phi proximity
        dphi = np.abs(angle_diff(omtf_phi_rad, k_phi))
        phi_ok = dphi < PHI_MATCH_THRESH

        # step 3: station compatibility (soft — allow -1 if layer unknown)
        sta_ok = (omtf_station == k_dep) | (omtf_station == -1)

        candidates = bx_ok & phi_ok & sta_ok
        if not candidates.any():
            # relax station filter and retry with phi only
            candidates = bx_ok & phi_ok
        if not candidates.any():
            continue   # unmatched

        # tie-break: nearest phi
        best = int(np.argmin(dphi + np.where(candidates, 0.0, 1e9)))
        out_tid[i]  = int(omtf_track_id[best])
        out_amb[i]  = int(omtf_ambiguous[best])
        out_src[i]  = "omtf_transfer" if omtf_track_id[best] != 0 else "omtf_noise"
        n_matched  += 1

    rate = n_matched / max(1, N)
    return TransferResult(out_tid, out_amb, out_src, rate, N, n_matched)
