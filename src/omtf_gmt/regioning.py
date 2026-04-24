"""
OMTF processor region definitions for the GMT-visible-stub branch.

CRITICAL COORDINATE NOTE (verified in CMSSW source):
  reg_stub_phiHw is PROCESSOR-LOCAL, not global CMS phi.
  Source: OMTFinputMaker.cc calls angleConverter.getProcessorPhi(...),
  which subtracts phiZero(proc) before storing in the tree.

  phiZero(proc) = (PHI_BINS / NPROCESSORS) * proc  +  PHI_BINS / 24
                = 1800 * proc + 225   [5400-bin units, for NPROCESSORS=3]

  To recover global phi from phiHw:
      global_phi_hw  = phiHw + phiZero(proc)
      global_phi_rad = global_phi_hw * (2π / PHI_BINS)

  A naive phiHw × 2π/5400 gives the wrong global phi.

PRODUCTION CONFIGURATION (EOS files used in this study):
  NPROCESSORS = 3 (full-phi coverage with 3 processors instead of 12)
  Processor centres: proc=0 → 15°, proc=1 → 135°, proc=2 → 255°
  Each processor covers a 120° window (±60° around centre).
  No overlap between processors in this configuration.
  For a standard 12-processor OMTF: NPROCESSORS=12, centres at 15°+k×30°.

KMTF offlineCoord1 is already global phi in radians — no correction needed.
"""

from __future__ import annotations

import math
import numpy as np

# ----- production configuration constants ----------------------------------

PHI_BINS: int     = 5400           # OMTF phi bins over 2π (nPhiBins in config XML)
NPROCESSORS: int  = 3              # processors in this EOS production run
_PHI_ZERO_EXTRA: int = PHI_BINS // 24   # = 225 bins = 15°  (nPhiBins/24 in formula)

# window half-width: each processor covers one full sector = 360°/NPROCESSORS
# For NPROCESSORS=3: sector = 120°, half = 60° = π/3 rad
WINDOW_HALF: float = math.pi / NPROCESSORS   # rad; = π/3 for NPROCESSORS=3

OMTF_PHI_SCALE: float = 2.0 * math.pi / PHI_BINS   # rad per HW count


# ----- phiZero: processor local-to-global offset ---------------------------

def phiZero_hw(proc: int) -> int:
    """Processor phi offset in 5400-bin units (add to phiHw to get global phi)."""
    return (PHI_BINS // NPROCESSORS) * proc + _PHI_ZERO_EXTRA


def processor_phi_center(proc: int) -> float:
    """Centre phi of OMTF processor proc in radians (global CMS frame)."""
    return phiZero_hw(proc) * OMTF_PHI_SCALE


def processor_phi_window(proc: int, margin: float = 0.0) -> tuple[float, float]:
    """(lo, hi) global phi window for processor proc in radians."""
    centre = processor_phi_center(proc)
    return centre - WINDOW_HALF - margin, centre + WINDOW_HALF + margin


# ----- vectorised window membership (uses KMTF offlineCoord1 — global rad) -

def angle_diff(phi: np.ndarray | float, ref: float) -> np.ndarray | float:
    """Signed angular difference phi − ref, wrapped to (−π, π]."""
    d = np.asarray(phi, dtype=np.float64) - ref
    return (d + math.pi) % (2.0 * math.pi) - math.pi


def stubs_in_window(
    offlineCoord1: np.ndarray,
    proc: int,
    margin: float = 0.02,   # small buffer for calibration edge effects
) -> np.ndarray:
    """Boolean mask: True for KMTF stubs inside processor proc's phi window.

    Uses KMTF offlineCoord1 (already global phi in radians) directly.
    """
    centre = processor_phi_center(proc)
    half   = WINDOW_HALF + margin
    d = np.abs(angle_diff(np.asarray(offlineCoord1, dtype=np.float64), centre))
    return d <= half


def phi_rel(offlineCoord1: np.ndarray, proc: int) -> np.ndarray:
    """Processor-centred phi: offlineCoord1 − centre, wrapped to (−π, π]."""
    centre = processor_phi_center(proc)
    return angle_diff(np.asarray(offlineCoord1, dtype=np.float64), centre)


# ----- OMTF phi scale conversions (processor-local HW → global rad) --------

def omtf_phi_to_global_rad(phi_hw: np.ndarray | int, proc: int) -> np.ndarray:
    """Convert OMTF processor-local phiHw to global phi in radians.

    Applies the phiZero(proc) correction derived from CMSSW source.
    """
    global_hw = np.asarray(phi_hw, dtype=np.float64) + phiZero_hw(proc)
    # wrap to [0, 2π)
    return (global_hw * OMTF_PHI_SCALE) % (2.0 * math.pi)
