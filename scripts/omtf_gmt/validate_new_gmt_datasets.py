#!/usr/bin/env python
"""
Validation gate for new GMT/OMTF overlap datasets (G1–G10).

Runs 10 physics checks and writes a Markdown report, JSON summary, and CSV.

Usage (quick subset):
    python scripts/omtf_gmt/validate_new_gmt_datasets.py \
        --data-dir data/prod \
        --datasets G1 G2 G3 G4 G5 G6 G7 G8 G9 G10 \
        --files-per-dataset 20 \
        --output build/omtf_gmt/eval/new_dataset_validation.md

Usage (full production):
    python scripts/omtf_gmt/validate_new_gmt_datasets.py \
        --data-dir data/prod \
        --datasets G1 G2 G3 G4 G5 G6 G7 G8 G9 G10 \
        --all-files \
        --output build/omtf_gmt/eval/new_dataset_validation_full.md
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median, mode
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

try:
    import uproot
    import awkward as ak
except ImportError:
    print("uproot and awkward are required.  Run: pip install uproot awkward")
    sys.exit(1)

from omtf_gmt.regioning import stubs_in_window, omtf_phi_to_global_rad, angle_diff
from omtf_gmt.truth_transfer import transfer, PHI_MATCH_THRESH

# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

DATASET_NAMES = {
    "G1": "singlePromptOverlap",
    "G2": "singlePromptOverlap_PU200",
    "G3": "singleDisplacedOverlap",
    "G4": "singleDisplacedOverlap_PU200",
    "G5": "twoDisplacedOverlap_PU200",
    "G6": "triPromptOverlap_PU200",
    "G7": "hardNegLowEtaMuon",
    "G8": "hardNegLowEtaMuon_PU200",
    "G9":  "hardNegHighEtaMuon",
    "G10": "hardNegHighEtaMuon_PU200",
}

EXPECTED_N_GEN = {
    "G1": 1, "G2": 1, "G3": 1, "G4": 1,
    "G5": 2, "G6": 3, "G7": 1, "G8": 1,
    "G9": 1, "G10": 1,
}

EXPECTED_OVERLAP_TARGET_MULT = {
    "G1": 1, "G2": 1, "G3": 1, "G4": 1,
    "G5": 2, "G6": 3, "G7": 0, "G8": 0,
    "G9": 0, "G10": 0,
}

# pT ranges [min, max] in GeV
PT_RANGE = {
    "G1": (2, 200), "G2": (2, 200), "G3": (2, 200), "G4": (2, 200),
    "G5": (2, 200), "G6": (5, 80),  "G7": (2, 200), "G8": (2, 200),
    "G9": (2, 200), "G10": (2, 200),
}

# prompt vs displaced
PROMPT_DATASETS  = {"G1", "G2", "G6", "G7", "G8", "G9", "G10"}
DISPLACED_DATASETS = {"G3", "G4", "G5"}
DXY_RANGE = {
    "G3": (0, 50), "G4": (0, 50), "G5": (0, 30),
}
PU200_DATASETS = {"G2", "G4", "G5", "G6", "G8", "G10"}

IN_OVERLAP_LO = 0.82
IN_OVERLAP_HI = 1.24
LOW_ETA_MAX   = 0.75
HIGH_ETA_MIN  = 1.24   # |eta| lower bound for G9/G10 high-eta hard negatives
SIGNAL_DATASETS   = {"G1", "G2", "G3", "G4", "G5", "G6"}
HARD_NEG_DATASETS = {"G7", "G8", "G9", "G10"}
LOW_ETA_DATASETS  = {"G7", "G8"}    # barrel-side hard negatives
HIGH_ETA_DATASETS = {"G9", "G10"}   # endcap-side hard negatives

# G1–G6 and G9/G10 are split into <DS>_pos and <DS>_neg subdirectories on disk.
# G7/G8 and B4 are single directories.
SPLIT_DATASETS = {"G1", "G2", "G3", "G4", "G5", "G6", "G9", "G10"}


def _dataset_dirs(ds: str, data_dir: Path) -> list[Path]:
    """Return the list of raw-data directories to scan for dataset *ds*."""
    if ds in SPLIT_DATASETS:
        dirs = [data_dir / f"{ds}_pos", data_dir / f"{ds}_neg"]
        return [d for d in dirs if d.exists()]
    return [data_dir / ds]

OMTF_HITS_BRANCHES = [
    "reg_eventNum", "reg_iProcessor",
    "reg_stub_layer", "reg_stub_phiHw", "reg_stub_phiBHw",
    "reg_stub_etaHw", "reg_stub_r", "reg_stub_quality",
    "reg_stub_type", "reg_stub_bx", "reg_stub_trackId",
    "reg_stub_ambiguous",
]
OMTF_TRANSFER_BRANCHES = [
    "reg_eventNum", "reg_iProcessor",
    "reg_stub_phiHw", "reg_stub_layer", "reg_stub_bx",
    "reg_stub_trackId", "reg_stub_ambiguous",
]
NANO_GEN_BRANCHES = [
    "event", "nGenMuon",
    "GenMuon_pt", "GenMuon_eta", "GenMuon_phi",
    "GenMuon_charge", "GenMuon_pdgId", "GenMuon_status",
]
NANO_GEN_OPTIONAL = ["GenMuon_dXY", "GenMuon_lXY",
                     "GenMuon_etaSt1", "GenMuon_etaSt2",
                     "GenMuon_phiSt1", "GenMuon_phiSt2"]
NANO_KMTF_BRANCHES = [
    "event",
    "nMuonStubKmtf",
    "MuonStubKmtf_isBarrel",
    "MuonStubKmtf_offlineCoord1", "MuonStubKmtf_offlineCoord2",
    "MuonStubKmtf_offlineEta1",   "MuonStubKmtf_offlineEta2",
    "MuonStubKmtf_quality",       "MuonStubKmtf_etaQuality",
    "MuonStubKmtf_bxNum",         "MuonStubKmtf_tfLayer",
    "MuonStubKmtf_depthRegion",
]

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

PASS  = "PASS"
WARN  = "WARN"
FAIL  = "FAIL"


@dataclass
class CheckResult:
    status: str          # PASS / WARN / FAIL
    details: str = ""    # human-readable details


@dataclass
class DatasetResult:
    name: str
    n_hits_files: int  = 0
    n_nano_files: int  = 0
    n_matched_pairs: int = 0
    missing_nano: list = field(default_factory=list)
    missing_hits: list = field(default_factory=list)

    # 3.2 gen multiplicity
    n_events:       int    = 0
    mode_n_gen:     int    = -1
    frac_expected:  float  = 0.0
    min_n_gen:      int    = 0
    max_n_gen:      int    = 0

    # 3.3 eta
    frac_vertex_in_overlap: float = 0.0
    frac_st2_in_overlap:    float = 0.0
    frac_low_eta:           float = 0.0
    frac_high_eta:          float = 0.0

    # 3.4 pT / dXY
    pt_stats: dict  = field(default_factory=dict)
    dxy_stats: dict = field(default_factory=dict)

    # 3.5 charge
    n_muplus:  int = 0
    n_muminus: int = 0

    # 3.6 OMTF struct
    n_omtf_windows:       int = 0
    bad_vector_length:    int = 0
    bad_processor_rows:   int = 0
    bad_trackid_rows:     int = 0
    bad_ambiguous_rows:   int = 0

    # 3.7 event join
    n_omtf_events: int = 0
    n_nano_events: int = 0
    n_matched_events: int = 0

    # 3.8 KMTF stubs
    n_total_kmtf_stubs: int = 0
    mean_kmtf_stubs_per_event: float = 0.0
    frac_kmtf_in_overlap: float = 0.0

    # 3.9 truth transfer
    tt_n_stubs:    int   = 0
    tt_n_matched:  int   = 0
    tt_match_pct:  float = 0.0
    tt_sig_transferred: int = 0
    tt_noise_confirmed: int = 0
    tt_phi_residuals: list = field(default_factory=list)

    # 3.10 target multiplicity
    target_mult_counts: dict = field(default_factory=dict)

    # per-check verdicts
    checks: dict = field(default_factory=dict)

    @property
    def overall_status(self) -> str:
        statuses = [c.status for c in self.checks.values()]
        if FAIL in statuses:
            return FAIL
        if WARN in statuses:
            return WARN
        return PASS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _in_overlap(eta: np.ndarray) -> np.ndarray:
    a = np.abs(eta)
    return (a >= IN_OVERLAP_LO) & (a <= IN_OVERLAP_HI)


def _percentile(arr: np.ndarray, p: float) -> float:
    if len(arr) == 0:
        return float("nan")
    return float(np.percentile(arr, p))


def _safe_mode(counts: Counter) -> int:
    if not counts:
        return -1
    return counts.most_common(1)[0][0]


def _nano_path(hits_path: Path) -> Path:
    return hits_path.parent / hits_path.name.replace("omtf_hits_", "omtf_nano_")


def _hits_path(nano_path: Path) -> Path:
    return nano_path.parent / nano_path.name.replace("omtf_nano_", "omtf_hits_")


def _load_gen_branches(nano_path: Path) -> tuple[dict, set]:
    """Return (array_dict, available_optional_set)."""
    tree = uproot.open(str(nano_path))["Events"]
    avail = set(tree.keys())
    opts = [b for b in NANO_GEN_OPTIONAL if b in avail]
    branches = NANO_GEN_BRANCHES + opts
    arr = tree.arrays(branches, library="ak")
    return arr, set(opts)


def _load_kmtf_branches(nano_path: Path) -> ak.Array:
    tree = uproot.open(str(nano_path))["Events"]
    avail = set(tree.keys())
    branches = [b for b in NANO_KMTF_BRANCHES if b in avail]
    return tree.arrays(branches, library="ak"), set(avail)


# ---------------------------------------------------------------------------
# Check 3.1: file-pair integrity
# ---------------------------------------------------------------------------

def check_file_pairs(ds: str, data_dir: Path, hits_files: list[Path],
                     all_files: bool = False) -> DatasetResult:
    r = DatasetResult(name=ds)
    ds_dirs = _dataset_dirs(ds, data_dir)

    r.n_hits_files = len(hits_files)
    r.n_nano_files = sum(len(list(d.glob("omtf_nano_*.root"))) for d in ds_dirs)

    # Every selected hits file must have a matching nano file.
    for hf in hits_files:
        nf = _nano_path(hf)
        if nf.exists():
            r.n_matched_pairs += 1
        else:
            r.missing_nano.append(hf.name)

    # In full-scan mode, also check the reverse: every nano file in the
    # directory should have a hits counterpart.  In subset mode this check
    # would produce false positives for files not in the subset.
    if all_files:
        for d in ds_dirs:
            for nf in sorted(d.glob("omtf_nano_*.root")):
                hf = _hits_path(nf)
                if not hf.exists():
                    r.missing_hits.append(nf.name)

    n_missing = len(r.missing_nano)
    if n_missing == 0:
        r.checks["file_pairs"] = CheckResult(PASS)
    else:
        r.checks["file_pairs"] = CheckResult(
            FAIL, f"{n_missing} hits files have no matching nano file"
        )
    return r


# ---------------------------------------------------------------------------
# Check 3.2–3.5: gen-level properties (per-file, accumulate)
# ---------------------------------------------------------------------------

def accumulate_gen_and_charge(r: DatasetResult, ds: str, hits_files: list[Path]) -> None:
    n_gen_vals: list[int] = []
    vertex_eta_all: list[float] = []
    st2_eta_all:    list[float] = []
    pt_all:         list[float] = []
    dxy_all:        list[float] = []
    n_plus  = 0
    n_minus = 0

    for hf in hits_files:
        nf = _nano_path(hf)
        if not nf.exists():
            continue
        try:
            arr, opts = _load_gen_branches(nf)
        except Exception as exc:
            print(f"    WARN: could not read gen branches from {nf.name}: {exc}")
            continue

        has_dxy  = "GenMuon_dXY" in opts
        has_etaSt2 = "GenMuon_etaSt2" in opts

        for i in range(len(arr)):
            n = int(arr["nGenMuon"][i])
            n_gen_vals.append(n)

            etas   = ak.to_numpy(arr["GenMuon_eta"][i]).astype(np.float64)
            pts    = ak.to_numpy(arr["GenMuon_pt"][i]).astype(np.float64)
            charges = ak.to_numpy(arr["GenMuon_charge"][i]).astype(np.float64)

            vertex_eta_all.extend(etas.tolist())
            pt_all.extend(pts.tolist())

            if has_etaSt2:
                eta_st2 = ak.to_numpy(arr["GenMuon_etaSt2"][i]).astype(np.float64)
                st2_eta_all.extend(eta_st2.tolist())

            if has_dxy:
                dxys = ak.to_numpy(arr["GenMuon_dXY"][i]).astype(np.float64)
                dxy_all.extend(dxys.tolist())

            for c in charges:
                if c > 0:
                    n_plus += 1
                else:
                    n_minus += 1

    r.n_events   = len(n_gen_vals)
    counts       = Counter(n_gen_vals)
    r.mode_n_gen = _safe_mode(counts)
    expected_n   = EXPECTED_N_GEN[ds]
    r.frac_expected = counts[expected_n] / max(1, len(n_gen_vals))
    r.min_n_gen  = min(n_gen_vals) if n_gen_vals else 0
    r.max_n_gen  = max(n_gen_vals) if n_gen_vals else 0

    # --- 3.2 verdict ---
    if r.frac_expected >= 0.999:
        r.checks["gen_multiplicity"] = CheckResult(PASS)
    elif r.frac_expected >= 0.95:
        r.checks["gen_multiplicity"] = CheckResult(
            WARN, f"frac_expected={r.frac_expected:.4f} < 0.999"
        )
    else:
        r.checks["gen_multiplicity"] = CheckResult(
            FAIL, f"frac_expected={r.frac_expected:.4f} — PartID may be wrong"
        )

    # --- 3.3 eta domain ---
    v_eta = np.array(vertex_eta_all)
    frac_v = float(np.mean(_in_overlap(v_eta))) if len(v_eta) else 0.0
    r.frac_vertex_in_overlap = frac_v

    if st2_eta_all:
        s2_eta = np.array(st2_eta_all)
        # Filter out sentinel fill values: etaSt2 = -9.0 (or abs > 5) indicates that
        # station-2 extrapolation was not attempted or failed.  Only use physically
        # plausible values (|eta| < 5) for the overlap fraction.
        s2_valid = s2_eta[np.abs(s2_eta) < 5.0]
        frac_s2 = float(np.mean(_in_overlap(s2_valid))) if len(s2_valid) else float("nan")
    else:
        frac_s2 = float("nan")
    r.frac_st2_in_overlap = frac_s2

    frac_low  = float(np.mean(np.abs(v_eta) < LOW_ETA_MAX))  if len(v_eta) else 0.0
    frac_high = float(np.mean(np.abs(v_eta) > HIGH_ETA_MIN)) if len(v_eta) else 0.0
    r.frac_low_eta  = frac_low
    r.frac_high_eta = frac_high

    # Use station-2 (propagated) eta as the primary overlap metric when available and
    # reliable; fall back to vertex eta.  Station-2 eta is more relevant for trigger
    # primitives, but its propagation can fail (producing sentinel zeros, filtered above).
    s2_available = not (frac_s2 != frac_s2)   # not NaN
    if s2_available:
        eta_metric = frac_s2
        eta_src = "station-2"
    else:
        eta_metric = frac_v
        eta_src = "vertex"

    # Thresholds differ by metric source:
    # - Vertex eta is the generator-level truth (eta filter acts on it directly).
    #   Expect ≥ 0.90 pass, ≥ 0.70 warn.
    # - Station-2 propagated eta has natural shortfall: sentinel fill (-9) occurs when
    #   propagation fails (~40% of muons in 2-file tests), muons near the eta boundary
    #   can propagate out of the overlap band, and displaced muons are more affected.
    #   Use a lower pass threshold (0.65) and fail threshold (0.45) accordingly.
    pass_thresh_eta  = 0.65 if s2_available else 0.90
    warn_thresh_eta  = 0.45 if s2_available else 0.70

    if ds in SIGNAL_DATASETS:
        if eta_metric >= pass_thresh_eta:
            r.checks["eta_domain"] = CheckResult(PASS)
        elif eta_metric >= warn_thresh_eta:
            r.checks["eta_domain"] = CheckResult(
                WARN, f"{eta_src} overlap fraction={eta_metric:.3f} — check eta filter"
            )
        else:
            r.checks["eta_domain"] = CheckResult(
                FAIL, f"{eta_src} overlap fraction={eta_metric:.3f} — eta filter likely missing"
            )
    elif ds in LOW_ETA_DATASETS:  # G7/G8: barrel-side hard negatives
        # Low-eta metric uses vertex eta (station-2 propagation for barrel muons
        # at |eta|<0.75 is unreliable).  Overlap veto uses the propagated metric.
        if eta_metric <= 0.05 and frac_low >= 0.80:
            r.checks["eta_domain"] = CheckResult(PASS)
        elif eta_metric <= 0.20:
            r.checks["eta_domain"] = CheckResult(
                WARN,
                f"{eta_src} overlap fraction={eta_metric:.3f}, "
                f"low-eta (vertex) fraction={frac_low:.3f}"
            )
        else:
            r.checks["eta_domain"] = CheckResult(
                FAIL,
                f"hard-neg has {eta_src} overlap fraction={eta_metric:.3f} "
                f"— eta range overlaps acceptance"
            )
    else:  # G9/G10: endcap-side hard negatives (|eta| > 1.24)
        # High-eta muons should not reach the OMTF overlap band (0.82–1.24).
        # Confirm with vertex eta that most muons are above the overlap edge.
        if eta_metric <= 0.05 and frac_high >= 0.80:
            r.checks["eta_domain"] = CheckResult(PASS)
        elif eta_metric <= 0.20:
            r.checks["eta_domain"] = CheckResult(
                WARN,
                f"{eta_src} overlap fraction={eta_metric:.3f}, "
                f"high-eta (vertex, |eta|>{HIGH_ETA_MIN}) fraction={frac_high:.3f}"
            )
        else:
            r.checks["eta_domain"] = CheckResult(
                FAIL,
                f"hard-neg has {eta_src} overlap fraction={eta_metric:.3f} "
                f"— eta range overlaps OMTF acceptance"
            )

    # --- 3.4 pT / dXY ---
    pts = np.array(pt_all)
    pt_lo, pt_hi = PT_RANGE[ds]
    if len(pts):
        r.pt_stats = {
            "min":    float(pts.min()),
            "p1":     _percentile(pts, 1),
            "median": float(np.median(pts)),
            "p99":    _percentile(pts, 99),
            "max":    float(pts.max()),
        }
        pt_ok = (r.pt_stats["p1"] >= pt_lo * 0.9) and (r.pt_stats["p99"] <= pt_hi * 1.1)
    else:
        r.pt_stats = {}
        pt_ok = False

    dxys = np.array(dxy_all)
    if len(dxys):
        r.dxy_stats = {
            "min":    float(dxys.min()),
            "median": float(np.median(dxys)),
            "max":    float(dxys.max()),
        }
    else:
        r.dxy_stats = {}

    dxy_ok = True
    dxy_issue = ""
    if ds in DISPLACED_DATASETS:
        if len(dxys) == 0:
            dxy_ok = False
            dxy_issue = "GenMuon_dXY branch missing or empty for displaced dataset"
        else:
            dmin, dmax = DXY_RANGE[ds]
            if dxys.max() < 0.5:
                dxy_ok = False
                dxy_issue = (f"displaced sample has max |dXY|={dxys.max():.2f} cm "
                             f"(expected up to {dmax} cm)")
    elif ds in PROMPT_DATASETS:
        if len(dxys) == 0:
            # Missing dXY for prompt: log a note but don't fail — prompt samples
            # in older productions may not store dXY.
            dxy_issue = "GenMuon_dXY branch absent — cannot verify promptness"
        elif np.median(np.abs(dxys)) > 1.0:
            dxy_ok = False
            dxy_issue = f"prompt sample has median |dXY|={np.median(np.abs(dxys)):.2f} cm"

    if pt_ok and dxy_ok and not dxy_issue:
        r.checks["pt_dxy"] = CheckResult(PASS)
    elif not dxy_ok:
        sev = FAIL if ds in DISPLACED_DATASETS else WARN
        if not pt_ok:
            r.checks["pt_dxy"] = CheckResult(FAIL, f"pT out of range; {dxy_issue}")
        else:
            r.checks["pt_dxy"] = CheckResult(sev, dxy_issue)
    elif not pt_ok:
        r.checks["pt_dxy"] = CheckResult(WARN, "pT range outside expected bounds")
    else:
        # dxy_issue is set but dxy_ok is True (missing-but-prompt case)
        r.checks["pt_dxy"] = CheckResult(WARN, dxy_issue)

    # --- 3.5 charge balance ---
    r.n_muplus  = n_plus
    r.n_muminus = n_minus
    total_mu = n_plus + n_minus
    if total_mu == 0:
        r.checks["charge"] = CheckResult(WARN, "no muons found to check charge")
    else:
        frac_plus = n_plus / total_mu
        if 0.40 <= frac_plus <= 0.60:
            r.checks["charge"] = CheckResult(PASS)
        elif 0.30 <= frac_plus <= 0.70:
            r.checks["charge"] = CheckResult(
                WARN, f"charge asymmetry: frac_mu+={frac_plus:.3f} — check if intentional"
            )
        else:
            r.checks["charge"] = CheckResult(
                WARN,
                f"strongly asymmetric charge: frac_mu+={frac_plus:.3f} "
                "(may be intentional single-charge production)"
            )


# ---------------------------------------------------------------------------
# Check 3.6: OMTFAllInputTree integrity
# ---------------------------------------------------------------------------

EXPECTED_PROC_MIN = 0
EXPECTED_PROC_MAX = 2   # NPROCESSORS-1 for 3-processor config

def check_omtf_integrity(r: DatasetResult, hits_files: list[Path]) -> None:
    bad_vec  = 0
    bad_proc = 0
    bad_tid  = 0
    bad_amb  = 0
    n_windows = 0

    vector_branch_names = [b for b in OMTF_HITS_BRANCHES
                           if b.startswith("reg_stub_")]

    for hf in hits_files:
        if not hf.exists():
            continue
        try:
            tree = uproot.open(str(hf))["simOmtfPhase2Digis/OMTFAllInputTree"]
            arr  = tree.arrays(OMTF_HITS_BRANCHES, library="ak")
        except Exception as exc:
            print(f"    WARN: cannot open {hf.name}: {exc}")
            r.checks["omtf_integrity"] = CheckResult(FAIL, f"cannot open file: {exc}")
            return

        for i in range(len(arr)):
            n_windows += 1
            lengths = []
            for b in vector_branch_names:
                lengths.append(len(arr[b][i]))

            if len(set(lengths)) > 1:
                bad_vec += 1

            proc = int(arr["reg_iProcessor"][i])
            if proc < EXPECTED_PROC_MIN or proc > EXPECTED_PROC_MAX:
                bad_proc += 1

            tids = ak.to_numpy(arr["reg_stub_trackId"][i])
            if np.any(tids < 0):
                bad_tid += 1

            ambs = ak.to_numpy(arr["reg_stub_ambiguous"][i])
            if not np.all((ambs == 0) | (ambs == 1)):
                bad_amb += 1

    r.n_omtf_windows    = n_windows
    r.bad_vector_length = bad_vec
    r.bad_processor_rows = bad_proc
    r.bad_trackid_rows   = bad_tid
    r.bad_ambiguous_rows = bad_amb

    if bad_vec + bad_proc + bad_amb == 0:
        r.checks["omtf_integrity"] = CheckResult(PASS)
    elif bad_vec > 0:
        r.checks["omtf_integrity"] = CheckResult(
            FAIL, f"{bad_vec} windows with mismatched stub-vector lengths"
        )
    else:
        r.checks["omtf_integrity"] = CheckResult(
            WARN,
            f"bad_proc={bad_proc}, bad_trackId={bad_tid}, bad_ambiguous={bad_amb}"
        )


# ---------------------------------------------------------------------------
# Check 3.7: Nano ↔ hits event join
# ---------------------------------------------------------------------------

def check_event_join(r: DatasetResult, hits_files: list[Path]) -> None:
    omtf_events_total = 0
    nano_events_total = 0
    matched_total     = 0

    for hf in hits_files:
        nf = _nano_path(hf)
        if not nf.exists():
            continue
        try:
            tree  = uproot.open(str(hf))["simOmtfPhase2Digis/OMTFAllInputTree"]
            arr_h = tree.arrays(["reg_eventNum"], library="ak")
            tree_n = uproot.open(str(nf))["Events"]
            arr_n  = tree_n.arrays(["event"], library="ak")
        except Exception as exc:
            print(f"    WARN: join check failed for {hf.name}: {exc}")
            continue

        omtf_evs = set(int(arr_h["reg_eventNum"][i]) & 0xFFFFFFFF
                       for i in range(len(arr_h)))
        nano_evs = set(int(arr_n["event"][i]) & 0xFFFFFFFF
                       for i in range(len(arr_n)))

        omtf_events_total += len(omtf_evs)
        nano_events_total += len(nano_evs)
        matched_total     += len(omtf_evs & nano_evs)

    r.n_omtf_events    = omtf_events_total
    r.n_nano_events    = nano_events_total
    r.n_matched_events = matched_total

    frac = matched_total / max(1, omtf_events_total)
    if frac >= 0.99:
        r.checks["event_join"] = CheckResult(PASS)
    elif frac >= 0.90:
        r.checks["event_join"] = CheckResult(
            WARN, f"match fraction={frac:.4f} — check file pairing"
        )
    else:
        r.checks["event_join"] = CheckResult(
            FAIL, f"match fraction={frac:.4f} — event number mismatch or wrong file pairs"
        )


# ---------------------------------------------------------------------------
# Check 3.8: KMTF stub presence and eta coverage
# ---------------------------------------------------------------------------

def check_kmtf_stubs(r: DatasetResult, ds: str, hits_files: list[Path]) -> None:
    total_stubs   = 0
    in_overlap    = 0
    n_events_seen = 0

    for hf in hits_files:
        nf = _nano_path(hf)
        if not nf.exists():
            continue
        try:
            arr, avail = _load_kmtf_branches(nf)
        except Exception as exc:
            print(f"    WARN: KMTF branch read failed {nf.name}: {exc}")
            continue

        has_eta1 = "MuonStubKmtf_offlineEta1" in avail

        for i in range(len(arr)):
            n_events_seen += 1
            is_bar = ak.to_numpy(arr["MuonStubKmtf_isBarrel"][i]).astype(bool)
            n_bar  = int(is_bar.sum())
            total_stubs += n_bar

            if has_eta1 and n_bar > 0:
                eta1 = ak.to_numpy(arr["MuonStubKmtf_offlineEta1"][i])[is_bar].astype(np.float64)
                in_overlap += int(_in_overlap(eta1).sum())

    r.n_total_kmtf_stubs = total_stubs
    r.mean_kmtf_stubs_per_event = total_stubs / max(1, n_events_seen)
    r.frac_kmtf_in_overlap = in_overlap / max(1, total_stubs)

    if ds in LOW_ETA_DATASETS:
        # G7/G8: barrel muons must have real barrel stubs, mostly outside overlap
        if total_stubs == 0:
            r.checks["kmtf_stubs"] = CheckResult(
                FAIL, "no KMTF barrel stubs found — barrel hard negatives must have real barrel stubs"
            )
        elif r.frac_kmtf_in_overlap > 0.30:
            r.checks["kmtf_stubs"] = CheckResult(
                WARN,
                f"frac_in_overlap={r.frac_kmtf_in_overlap:.3f} is high for hard negatives"
            )
        else:
            r.checks["kmtf_stubs"] = CheckResult(PASS)
    elif ds in HIGH_ETA_DATASETS:
        # G9/G10: endcap muons — zero or few barrel stubs is expected
        if total_stubs == 0:
            r.checks["kmtf_stubs"] = CheckResult(
                WARN, "no KMTF barrel stubs (expected for endcap muons at |eta|>1.24)"
            )
        elif r.frac_kmtf_in_overlap > 0.30:
            r.checks["kmtf_stubs"] = CheckResult(
                WARN,
                f"frac_in_overlap={r.frac_kmtf_in_overlap:.3f} is high for high-eta hard negatives"
            )
        else:
            r.checks["kmtf_stubs"] = CheckResult(PASS)
    else:  # G1–G6
        if total_stubs == 0:
            r.checks["kmtf_stubs"] = CheckResult(FAIL, "no KMTF barrel stubs found")
        elif r.frac_kmtf_in_overlap < 0.30:
            r.checks["kmtf_stubs"] = CheckResult(
                WARN, f"only {r.frac_kmtf_in_overlap:.3f} of KMTF stubs in overlap — expected more"
            )
        else:
            r.checks["kmtf_stubs"] = CheckResult(PASS)


# ---------------------------------------------------------------------------
# Check 3.9: truth-transfer quality
# ---------------------------------------------------------------------------

def check_truth_transfer(r: DatasetResult, ds: str, hits_files: list[Path]) -> None:
    n_stubs    = 0
    n_matched  = 0
    n_sig_xfer = 0
    # for hard-neg: count signal transfers in the overlap eta band specifically
    n_sig_xfer_overlap = 0
    n_noise_c  = 0
    phi_res: list[float] = []

    for hf in hits_files:
        nf = _nano_path(hf)
        if not nf.exists():
            continue
        try:
            tree  = uproot.open(str(hf))["simOmtfPhase2Digis/OMTFAllInputTree"]
            arr_h = tree.arrays(OMTF_TRANSFER_BRANCHES, library="ak")
            arr_n, avail_n = _load_kmtf_branches(nf)
        except Exception as exc:
            print(f"    WARN: truth-transfer read failed {hf.name}: {exc}")
            continue

        has_eta1 = "MuonStubKmtf_offlineEta1" in avail_n

        nano_map: dict[int, dict] = {}
        for i in range(len(arr_n)):
            ev    = int(arr_n["event"][i]) & 0xFFFFFFFF
            is_b  = ak.to_numpy(arr_n["MuonStubKmtf_isBarrel"][i]).astype(bool)
            entry: dict = {
                "c1":  ak.to_numpy(arr_n["MuonStubKmtf_offlineCoord1"][i])[is_b].astype(np.float64),
                "dep": ak.to_numpy(arr_n["MuonStubKmtf_depthRegion"][i])[is_b].astype(np.int8),
                "bx":  ak.to_numpy(arr_n["MuonStubKmtf_bxNum"][i])[is_b].astype(np.int8),
            }
            if has_eta1:
                entry["eta1"] = ak.to_numpy(arr_n["MuonStubKmtf_offlineEta1"][i])[is_b].astype(np.float64)
            nano_map[ev] = entry

        for i in range(len(arr_h)):
            ev   = int(arr_h["reg_eventNum"][i]) & 0xFFFFFFFF
            proc = int(arr_h["reg_iProcessor"][i])
            if ev not in nano_map:
                continue

            phi_hw = ak.to_numpy(arr_h["reg_stub_phiHw"][i]).astype(np.int32)
            layer  = ak.to_numpy(arr_h["reg_stub_layer"][i]).astype(np.int8)
            bx_o   = ak.to_numpy(arr_h["reg_stub_bx"][i]).astype(np.int8)
            tid_o  = ak.to_numpy(arr_h["reg_stub_trackId"][i]).astype(np.int8)
            amb_o  = ak.to_numpy(arr_h["reg_stub_ambiguous"][i]).astype(np.uint8)

            nm   = nano_map[ev]
            mask = stubs_in_window(nm["c1"], proc)
            k_c1 = nm["c1"][mask]
            k_dep = nm["dep"][mask]
            k_bx  = nm["bx"][mask]
            if not mask.any():
                continue

            n_stubs += int(mask.sum())
            tr = transfer(k_c1, k_dep, k_bx, phi_hw, layer, bx_o, tid_o, amb_o,
                          omtf_proc=proc)
            n_matched  += tr.n_matched
            n_sig_xfer += int((tr.track_id > 0).sum())
            n_noise_c  += int(
                (np.array(tr.truth_source) == "omtf_noise").sum()
            )

            # For hard-neg datasets: count signal transfers only in overlap eta.
            # Barrel muons (|eta|<0.75) legitimately produce OMTF stubs and get
            # non-zero trackIds — that is expected. The diagnostic is whether any
            # of those stubs fall in the overlap band (0.82-1.24).
            if has_eta1 and "eta1" in nm:
                k_eta = nm["eta1"][mask]
                overlap_mask_k = _in_overlap(k_eta)
                sig_mask = tr.track_id > 0
                n_sig_xfer_overlap += int((sig_mask & overlap_mask_k).sum())

            if len(phi_hw) > 0:
                omtf_phi_r = omtf_phi_to_global_rad(phi_hw, proc)
                for j in range(len(k_c1)):
                    if tr.truth_source[j] != "unmatched":
                        dphi = float(np.min(np.abs(angle_diff(omtf_phi_r, float(k_c1[j])))))
                        phi_res.append(dphi)

    r.tt_n_stubs         = n_stubs
    r.tt_n_matched       = n_matched
    r.tt_match_pct       = 100.0 * n_matched / max(1, n_stubs)
    r.tt_sig_transferred = n_sig_xfer
    r.tt_noise_confirmed = n_noise_c
    r.tt_phi_residuals   = phi_res

    match_rate = n_matched / max(1, n_stubs)
    phi_med_mrad = float(np.median(phi_res) * 1000) if phi_res else float("nan")

    # PU200 datasets have extra pile-up KMTF stubs that never match an OMTF stub,
    # so the expected match rate is lower.  Use relaxed thresholds for those.
    is_pu = ds in PU200_DATASETS
    pass_thresh = 0.55 if is_pu else 0.80
    fail_thresh = 0.35 if is_pu else 0.60

    if ds in LOW_ETA_DATASETS:
        # G7/G8: barrel muons produce real stubs; absence of stubs is a hard failure
        if n_stubs == 0:
            r.checks["truth_transfer"] = CheckResult(
                FAIL, "no KMTF stubs — cannot validate truth transfer"
            )
        elif n_sig_xfer_overlap / max(1, n_stubs) > 0.10:
            r.checks["truth_transfer"] = CheckResult(
                FAIL,
                f"hard-neg has {n_sig_xfer_overlap} signal-transferred stubs "
                f"in overlap eta — overlap targets should be 0"
            )
        else:
            r.checks["truth_transfer"] = CheckResult(PASS)
    elif ds in HIGH_ETA_DATASETS:
        # G9/G10: endcap muons may have zero KMTF barrel stubs — that is expected
        if n_stubs == 0:
            r.checks["truth_transfer"] = CheckResult(
                WARN, "no KMTF barrel stubs for high-eta sample (expected for endcap muons)"
            )
        elif n_sig_xfer_overlap / max(1, n_stubs) > 0.10:
            r.checks["truth_transfer"] = CheckResult(
                FAIL,
                f"hard-neg has {n_sig_xfer_overlap} signal-transferred stubs "
                f"in overlap eta — overlap targets should be 0"
            )
        else:
            r.checks["truth_transfer"] = CheckResult(PASS)
    else:
        if match_rate >= pass_thresh and phi_med_mrad < 10.0:
            r.checks["truth_transfer"] = CheckResult(PASS)
        elif match_rate >= fail_thresh:
            r.checks["truth_transfer"] = CheckResult(
                WARN,
                f"match_rate={match_rate:.3f}, phi_median={phi_med_mrad:.1f} mrad"
            )
        else:
            r.checks["truth_transfer"] = CheckResult(
                FAIL,
                f"match_rate={match_rate:.3f} — truth transfer not reliable"
            )


# ---------------------------------------------------------------------------
# Check 3.10: target multiplicity
# ---------------------------------------------------------------------------

def check_target_multiplicity(r: DatasetResult, ds: str, hits_files: list[Path]) -> None:
    target_counts: Counter = Counter()
    expected_mult = EXPECTED_OVERLAP_TARGET_MULT[ds]

    for hf in hits_files:
        nf = _nano_path(hf)
        if not nf.exists():
            continue
        try:
            tree  = uproot.open(str(hf))["simOmtfPhase2Digis/OMTFAllInputTree"]
            arr_h = tree.arrays(OMTF_TRANSFER_BRANCHES, library="ak")
            arr_n, avail_n = _load_kmtf_branches(nf)
        except Exception as exc:
            print(f"    WARN: target mult read failed {hf.name}: {exc}")
            continue

        has_eta1 = "MuonStubKmtf_offlineEta1" in avail_n

        # Build nano event map with full KMTF stubs
        nano_map: dict[int, dict] = {}
        for i in range(len(arr_n)):
            ev   = int(arr_n["event"][i]) & 0xFFFFFFFF
            is_b = ak.to_numpy(arr_n["MuonStubKmtf_isBarrel"][i]).astype(bool)
            entry: dict = {
                "c1":  ak.to_numpy(arr_n["MuonStubKmtf_offlineCoord1"][i])[is_b].astype(np.float64),
                "dep": ak.to_numpy(arr_n["MuonStubKmtf_depthRegion"][i])[is_b].astype(np.int8),
                "bx":  ak.to_numpy(arr_n["MuonStubKmtf_bxNum"][i])[is_b].astype(np.int8),
            }
            if has_eta1:
                entry["eta1"] = ak.to_numpy(arr_n["MuonStubKmtf_offlineEta1"][i])[is_b].astype(np.float64)
            nano_map[ev] = entry

        for i in range(len(arr_h)):
            ev   = int(arr_h["reg_eventNum"][i]) & 0xFFFFFFFF
            proc = int(arr_h["reg_iProcessor"][i])
            if ev not in nano_map:
                continue

            phi_hw = ak.to_numpy(arr_h["reg_stub_phiHw"][i]).astype(np.int32)
            layer  = ak.to_numpy(arr_h["reg_stub_layer"][i]).astype(np.int8)
            bx_o   = ak.to_numpy(arr_h["reg_stub_bx"][i]).astype(np.int8)
            tid_o  = ak.to_numpy(arr_h["reg_stub_trackId"][i]).astype(np.int8)
            amb_o  = ak.to_numpy(arr_h["reg_stub_ambiguous"][i]).astype(np.uint8)

            nm   = nano_map[ev]
            mask = stubs_in_window(nm["c1"], proc)
            k_c1 = nm["c1"][mask]
            if not mask.any():
                # window with no KMTF stubs — count as 0 targets
                target_counts[0] += 1
                continue

            k_dep = nm["dep"][mask]
            k_bx  = nm["bx"][mask]

            tr = transfer(k_c1, k_dep, k_bx, phi_hw, layer, bx_o, tid_o, amb_o,
                          omtf_proc=proc)

            # "Overlap target" = signal stub in the overlap eta band (0.82–1.24).
            # Barrel-muon signal stubs (G7/G8) will be outside this band and
            # correctly yield 0 overlap targets even though they have trackId>0.
            if has_eta1 and "eta1" in nm:
                k_eta = nm["eta1"][mask]
                overlap_signal = (tr.track_id > 0) & _in_overlap(k_eta)
            else:
                # no eta info — fall back to all signal stubs (may over-count for G7/G8)
                overlap_signal = tr.track_id > 0

            unique_ids = set(int(x) for x in tr.track_id[overlap_signal] if x > 0)
            target_counts[len(unique_ids)] += 1

    r.target_mult_counts = dict(sorted(target_counts.items()))
    total_windows = sum(target_counts.values())

    if total_windows == 0:
        if ds in HARD_NEG_DATASETS:
            # No OMTF windows at all: acceptable only if KMTF stubs confirm no
            # overlap signal (checked separately in 3.8/3.9).
            r.checks["target_multiplicity"] = CheckResult(
                WARN,
                "no OMTF windows found; check KMTF stub presence (3.8) and truth transfer (3.9)"
            )
        else:
            r.checks["target_multiplicity"] = CheckResult(
                FAIL, "no OMTF windows found — file-pair or join issue"
            )
        return

    frac_expected = target_counts[expected_mult] / max(1, total_windows)

    # With NPROCESSORS=3, a single muon occupies exactly 1 of 3 windows per event.
    # The maximum achievable frac_target=1 for a single-muon sample is ~1/3 ≈ 0.33.
    # For 0-target (G7/G8): frac_target=0 should be ≥ 0.80 (barrel muons leave
    # some residual overlap stubs due to geometry edge effects at |eta|≈0.82).
    # Hard failure: the multiplicity is systematically higher than expected (PartID bug).

    # Fraction of windows with MORE than expected targets
    frac_too_high = sum(
        cnt for mult, cnt in target_counts.items() if mult > expected_mult
    ) / max(1, total_windows)

    wrong_high = frac_too_high > 0.10

    if ds in HARD_NEG_DATASETS:
        # G7/G8: expect nearly all windows to have 0 overlap targets
        if target_counts.get(0, 0) / max(1, total_windows) >= 0.80:
            r.checks["target_multiplicity"] = CheckResult(PASS)
        elif wrong_high:
            r.checks["target_multiplicity"] = CheckResult(
                FAIL,
                f"hard-neg has {frac_too_high:.3f} of windows with ≥1 overlap targets "
                f"— eta range may overlap acceptance"
            )
        else:
            r.checks["target_multiplicity"] = CheckResult(
                WARN, f"frac_target=0: {frac_expected:.3f}"
            )
    elif expected_mult == 1:
        # Single-muon datasets: in 3-processor OMTF, ~1/3 of windows have 1 target.
        # Accept if the fraction with 2+ targets is low (PartID check).
        if wrong_high:
            r.checks["target_multiplicity"] = CheckResult(
                FAIL,
                f"1-target sample has {frac_too_high:.3f} of windows with ≥2 overlap targets "
                f"— PartID error suspected"
            )
        elif frac_expected >= 0.15:
            r.checks["target_multiplicity"] = CheckResult(PASS)
        else:
            r.checks["target_multiplicity"] = CheckResult(
                WARN, f"frac_target=1: {frac_expected:.3f} — unusually low signal windows"
            )
    else:
        # Multi-muon datasets (G5: 2, G6: 3)
        if wrong_high:
            r.checks["target_multiplicity"] = CheckResult(
                FAIL,
                f"sample has {frac_too_high:.3f} of windows with more targets than expected "
                f"(expected {expected_mult}) — PartID error suspected"
            )
        elif frac_expected >= 0.10:
            r.checks["target_multiplicity"] = CheckResult(PASS)
        else:
            r.checks["target_multiplicity"] = CheckResult(
                WARN, f"frac_target={expected_mult}: {frac_expected:.3f}"
            )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def validate_dataset(
    ds: str,
    data_dir: Path,
    n_files: int | None,
    all_files: bool = False,
) -> DatasetResult:
    ds_dirs = _dataset_dirs(ds, data_dir)
    missing = [str(d) for d in ds_dirs if not d.exists()]
    if missing or not ds_dirs:
        r = DatasetResult(name=ds)
        for key in ["file_pairs", "gen_multiplicity", "eta_domain", "pt_dxy",
                    "charge", "omtf_integrity", "event_join", "kmtf_stubs",
                    "truth_transfer", "target_multiplicity"]:
            r.checks[key] = CheckResult(FAIL, f"dataset directories not found: {missing or ds_dirs}")
        return r

    hits_files = sorted(
        hf for d in ds_dirs for hf in d.glob("omtf_hits_*.root")
    )
    if n_files is not None:
        hits_files = hits_files[:n_files]

    if not hits_files:
        r = DatasetResult(name=ds)
        for key in ["file_pairs", "gen_multiplicity", "eta_domain", "pt_dxy",
                    "charge", "omtf_integrity", "event_join", "kmtf_stubs",
                    "truth_transfer", "target_multiplicity"]:
            r.checks[key] = CheckResult(FAIL, "no hits files found")
        return r

    print(f"  [{ds}] {len(hits_files)} files", flush=True)

    # 3.1 file pairs
    r = check_file_pairs(ds, data_dir, hits_files, all_files=all_files)
    # filter to matched pairs only for subsequent checks
    good_files = [hf for hf in hits_files if _nano_path(hf).exists()]

    # 3.2–3.5 gen-level
    print(f"  [{ds}] gen branches ...", end="", flush=True)
    accumulate_gen_and_charge(r, ds, good_files)
    print(f" events={r.n_events}")

    # 3.6 OMTF struct
    print(f"  [{ds}] OMTF integrity ...", end="", flush=True)
    check_omtf_integrity(r, good_files)
    print(f" windows={r.n_omtf_windows}")

    # 3.7 event join
    print(f"  [{ds}] event join ...", end="", flush=True)
    check_event_join(r, good_files)
    print(f" matched={r.n_matched_events}/{r.n_omtf_events}")

    # 3.8 KMTF stubs
    print(f"  [{ds}] KMTF stubs ...", end="", flush=True)
    check_kmtf_stubs(r, ds, good_files)
    print(f" stubs={r.n_total_kmtf_stubs}")

    # 3.9 truth transfer
    print(f"  [{ds}] truth transfer ...", end="", flush=True)
    check_truth_transfer(r, ds, good_files)
    print(f" match={r.tt_match_pct:.1f}%")

    # 3.10 target multiplicity
    print(f"  [{ds}] target multiplicity ...", end="", flush=True)
    check_target_multiplicity(r, ds, good_files)
    print(f" counts={r.target_mult_counts}")

    return r


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _status_icon(s: str) -> str:
    return {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(s, s)


def _fmt_float(v: Any, fmt: str = ".3f") -> str:
    if isinstance(v, float) and (v != v):   # NaN check
        return "n/a"
    try:
        return format(v, fmt)
    except Exception:
        return str(v)


def render_markdown(results: list[DatasetResult], n_files_used: int | None) -> str:
    date = datetime.date.today().isoformat()
    scope = f"{n_files_used} files/dataset" if n_files_used else "full production"
    lines: list[str] = [
        "# GMT/OMTF Overlap Dataset Validation Report",
        "",
        f"Validation date: {date}",
        f"Scope: {scope}",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Dataset | Description | Events | Overall |",
        "| --- | --- | --- | --- |",
    ]
    for r in results:
        icon = _status_icon(r.overall_status)
        lines.append(
            f"| {r.name} | {DATASET_NAMES.get(r.name, '')} | {r.n_events:,} | {icon} {r.overall_status} |"
        )

    # --- 3.1 ---
    lines += ["", "---", "", "## 3.1 File-pair integrity", "",
              "| Dataset | hits files | nano files | matched pairs | missing nano | missing hits |",
              "| --- | --- | --- | --- | --- | --- |"]
    for r in results:
        icon = _status_icon(r.checks.get("file_pairs", CheckResult(WARN)).status)
        lines.append(
            f"| {r.name} | {r.n_hits_files} | {r.n_nano_files} | {r.n_matched_pairs} "
            f"| {len(r.missing_nano)} | {len(r.missing_hits)} | {icon} |"
        )

    # --- 3.2 ---
    lines += ["", "---", "", "## 3.2 Generator-level multiplicity", "",
              "| Dataset | Events | mode(nGenMuon) | frac expected | min | max | Status |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in results:
        c = r.checks.get("gen_multiplicity", CheckResult(WARN))
        icon = _status_icon(c.status)
        exp = EXPECTED_N_GEN.get(r.name, "?")
        lines.append(
            f"| {r.name} | {r.n_events:,} | {r.mode_n_gen} (exp {exp}) "
            f"| {r.frac_expected:.4f} | {r.min_n_gen} | {r.max_n_gen} | {icon} {c.status} |"
        )

    # --- 3.3 ---
    lines += ["", "---", "", "## 3.3 Eta-domain validation", "",
              "| Dataset | frac vertex in overlap | frac station-2 in overlap | frac low-eta | frac high-eta | Status |",
              "| --- | --- | --- | --- | --- | --- |"]
    for r in results:
        c = r.checks.get("eta_domain", CheckResult(WARN))
        icon = _status_icon(c.status)
        st2 = _fmt_float(r.frac_st2_in_overlap)
        lines.append(
            f"| {r.name} | {r.frac_vertex_in_overlap:.4f} | {st2} "
            f"| {r.frac_low_eta:.4f} | {r.frac_high_eta:.4f} | {icon} {c.status} |"
        )

    # --- 3.4 ---
    lines += ["", "---", "", "## 3.4 pT and displacement validation", "",
              "| Dataset | pt min | pt p1 | pt median | pt p99 | pt max | dXY min | dXY median | dXY max | Status |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in results:
        c = r.checks.get("pt_dxy", CheckResult(WARN))
        icon = _status_icon(c.status)
        ps = r.pt_stats
        ds = r.dxy_stats
        def _g(d, k): return _fmt_float(d.get(k, float("nan")), ".1f")
        lines.append(
            f"| {r.name} | {_g(ps,'min')} | {_g(ps,'p1')} | {_g(ps,'median')} "
            f"| {_g(ps,'p99')} | {_g(ps,'max')} "
            f"| {_g(ds,'min')} | {_g(ds,'median')} | {_g(ds,'max')} | {icon} {c.status} |"
        )

    # --- 3.5 ---
    lines += ["", "---", "", "## 3.5 Charge balance", "",
              "| Dataset | n mu+ | n mu- | frac mu+ | frac mu- | Status |",
              "| --- | --- | --- | --- | --- | --- |"]
    for r in results:
        c = r.checks.get("charge", CheckResult(WARN))
        icon = _status_icon(c.status)
        total = r.n_muplus + r.n_muminus
        fp = r.n_muplus / max(1, total)
        fm = r.n_muminus / max(1, total)
        lines.append(
            f"| {r.name} | {r.n_muplus:,} | {r.n_muminus:,} "
            f"| {fp:.3f} | {fm:.3f} | {icon} {c.status} |"
        )

    # --- 3.6 ---
    lines += ["", "---", "", "## 3.6 OMTFAllInputTree integrity", "",
              "_Assumes `reg_iProcessor` ∈ [0, 2], corresponding to the 3-processor OMTF"
              " configuration per eta side.  If a future production stores global 0–11"
              " indices this check will incorrectly flag bad-processor rows._",
              "",
              "| Dataset | windows | bad vector-length | bad processor | bad trackId | bad ambiguous | Status |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in results:
        c = r.checks.get("omtf_integrity", CheckResult(WARN))
        icon = _status_icon(c.status)
        lines.append(
            f"| {r.name} | {r.n_omtf_windows:,} | {r.bad_vector_length} "
            f"| {r.bad_processor_rows} | {r.bad_trackid_rows} "
            f"| {r.bad_ambiguous_rows} | {icon} {c.status} |"
        )

    # --- 3.7 ---
    lines += ["", "---", "", "## 3.7 Nano ↔ hits event join", "",
              "| Dataset | OMTF events | Nano events | matched events | match fraction | Status |",
              "| --- | --- | --- | --- | --- | --- |"]
    for r in results:
        c = r.checks.get("event_join", CheckResult(WARN))
        icon = _status_icon(c.status)
        frac = r.n_matched_events / max(1, r.n_omtf_events)
        lines.append(
            f"| {r.name} | {r.n_omtf_events:,} | {r.n_nano_events:,} "
            f"| {r.n_matched_events:,} | {frac:.4f} | {icon} {c.status} |"
        )

    # --- 3.8 ---
    lines += ["", "---", "", "## 3.8 KMTF stub presence and eta coverage", "",
              "| Dataset | total KMTF stubs | mean stubs/event | frac in overlap | frac out overlap | Status |",
              "| --- | --- | --- | --- | --- | --- |"]
    for r in results:
        c = r.checks.get("kmtf_stubs", CheckResult(WARN))
        icon = _status_icon(c.status)
        f_out = 1.0 - r.frac_kmtf_in_overlap
        lines.append(
            f"| {r.name} | {r.n_total_kmtf_stubs:,} "
            f"| {r.mean_kmtf_stubs_per_event:.2f} "
            f"| {r.frac_kmtf_in_overlap:.4f} | {f_out:.4f} | {icon} {c.status} |"
        )

    # --- 3.9 ---
    lines += ["", "---", "", "## 3.9 Truth-transfer validation", "",
              "_Note: PU200 datasets (G2/G4/G5/G6/G8) have lower match rates due to"
              " pile-up KMTF stubs with no OMTF counterpart; thresholds are relaxed for those._",
              "",
              "| Dataset | PU | KMTF stubs | matched | match % | signal xfer | noise conf "
              "| phi res median [mrad] | phi res p95 [mrad] | Status |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in results:
        c = r.checks.get("truth_transfer", CheckResult(WARN))
        icon = _status_icon(c.status)
        res = np.array(r.tt_phi_residuals)
        phi_med = f"{float(np.median(res)*1000):.1f}" if len(res) else "n/a"
        phi_p95 = f"{float(np.percentile(res,95)*1000):.1f}" if len(res) else "n/a"
        pu_flag = "PU200" if r.name in PU200_DATASETS else "no PU"
        lines.append(
            f"| {r.name} | {pu_flag} | {r.tt_n_stubs:,} | {r.tt_n_matched:,} "
            f"| {r.tt_match_pct:.1f}% | {r.tt_sig_transferred:,} | {r.tt_noise_confirmed:,} "
            f"| {phi_med} | {phi_p95} | {icon} {c.status} |"
        )

    # --- 3.10 ---
    lines += ["", "---", "", "## 3.10 Target multiplicity validation", "",
              "| Dataset | frac 0 | frac 1 | frac 2 | frac 3 | Status |",
              "| --- | --- | --- | --- | --- | --- |"]
    for r in results:
        c = r.checks.get("target_multiplicity", CheckResult(WARN))
        icon = _status_icon(c.status)
        total = sum(r.target_mult_counts.values())
        def _fc(k): return f"{r.target_mult_counts.get(k,0)/max(1,total):.4f}"
        lines.append(
            f"| {r.name} | {_fc(0)} | {_fc(1)} | {_fc(2)} | {_fc(3)} | {icon} {c.status} |"
        )

    # --- failure notes ---
    failures = [(r, k, c) for r in results
                for k, c in r.checks.items() if c.status in (WARN, FAIL)]
    if failures:
        lines += ["", "---", "", "## Notes on warnings and failures", ""]
        for r, k, c in failures:
            lines.append(f"- **{r.name} / {k}** ({c.status}): {c.details}")

    # --- signoff ---
    all_pass = all(r.overall_status == PASS for r in results)
    proceed = "YES" if all_pass else "NO"

    check_keys = [
        ("file_pairs",          "File-pair integrity passed"),
        ("gen_multiplicity",    "Generator multiplicity passed"),
        ("eta_domain",          "Eta-domain validation passed"),
        ("pt_dxy",              "pT/dXY validation passed"),
        ("omtf_integrity",      "OMTFAllInputTree integrity passed"),
        ("event_join",          "Nano ↔ hits join passed"),
        ("kmtf_stubs",          "KMTF stub presence passed"),
        ("truth_transfer",      "Truth-transfer validation passed"),
        ("target_multiplicity", "Target multiplicity validation passed"),
    ]

    lines += ["", "---", "", "## Signoff", ""]
    for key, label in check_keys:
        all_ok = all(r.checks.get(key, CheckResult(FAIL)).status == PASS for r in results)
        box = "x" if all_ok else " "
        lines.append(f"- [{box}] {label}")

    lines += ["", f"Proceed to cache building: **{proceed}**", ""]
    return "\n".join(lines)


def render_json(results: list[DatasetResult]) -> dict:
    out = {"datasets": {}}
    for r in results:
        out["datasets"][r.name] = {
            "overall": r.overall_status,
            "n_events": r.n_events,
            "checks": {k: {"status": c.status, "details": c.details}
                       for k, c in r.checks.items()},
            "stats": {
                "n_hits_files": r.n_hits_files,
                "n_matched_pairs": r.n_matched_pairs,
                "mode_n_gen": r.mode_n_gen,
                "frac_expected_n_gen": r.frac_expected,
                "frac_vertex_in_overlap": r.frac_vertex_in_overlap,
                "frac_st2_in_overlap": r.frac_st2_in_overlap,
                "frac_low_eta": r.frac_low_eta,
                "frac_high_eta": r.frac_high_eta,
                "pt_stats": r.pt_stats,
                "dxy_stats": r.dxy_stats,
                "n_muplus": r.n_muplus,
                "n_muminus": r.n_muminus,
                "n_omtf_windows": r.n_omtf_windows,
                "bad_vector_length": r.bad_vector_length,
                "n_matched_events": r.n_matched_events,
                "n_omtf_events": r.n_omtf_events,
                "n_total_kmtf_stubs": r.n_total_kmtf_stubs,
                "mean_kmtf_stubs_per_event": r.mean_kmtf_stubs_per_event,
                "frac_kmtf_in_overlap": r.frac_kmtf_in_overlap,
                "tt_match_pct": r.tt_match_pct,
                "tt_sig_transferred": r.tt_sig_transferred,
                "target_mult_counts": r.target_mult_counts,
            },
        }
    return out


def render_csv(results: list[DatasetResult]) -> list[dict]:
    rows = []
    for r in results:
        total = sum(r.target_mult_counts.values())
        def _fc(k): return r.target_mult_counts.get(k, 0) / max(1, total)
        row = {
            "dataset": r.name,
            "description": DATASET_NAMES.get(r.name, ""),
            "overall": r.overall_status,
            "n_events": r.n_events,
            "mode_n_gen": r.mode_n_gen,
            "frac_expected_n_gen": f"{r.frac_expected:.4f}",
            "frac_vertex_in_overlap": f"{r.frac_vertex_in_overlap:.4f}",
            "frac_low_eta": f"{r.frac_low_eta:.4f}",
            "frac_high_eta": f"{r.frac_high_eta:.4f}",
            "pt_median": _fmt_float(r.pt_stats.get("median", float("nan")), ".2f"),
            "dxy_median": _fmt_float(r.dxy_stats.get("median", float("nan")), ".2f"),
            "frac_mu_plus": f"{r.n_muplus / max(1, r.n_muplus+r.n_muminus):.3f}",
            "omtf_bad_vec": r.bad_vector_length,
            "event_join_frac": f"{r.n_matched_events / max(1, r.n_omtf_events):.4f}",
            "n_kmtf_stubs": r.n_total_kmtf_stubs,
            "mean_kmtf_stubs_per_event": f"{r.mean_kmtf_stubs_per_event:.2f}",
            "frac_kmtf_in_overlap": f"{r.frac_kmtf_in_overlap:.4f}",
            "tt_match_pct": f"{r.tt_match_pct:.1f}",
            "frac_target_0": f"{_fc(0):.4f}",
            "frac_target_1": f"{_fc(1):.4f}",
            "frac_target_2": f"{_fc(2):.4f}",
            "frac_target_3": f"{_fc(3):.4f}",
        }
        for k, c in r.checks.items():
            row[f"check_{k}"] = c.status
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate new GMT/OMTF overlap datasets (G1–G10)"
    )
    p.add_argument("--data-dir", type=Path, default=Path("data/prod"))
    p.add_argument("--datasets", nargs="+",
                   default=["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10"])
    p.add_argument("--files-per-dataset", type=int, default=20,
                   help="Number of hits files per dataset (ignored if --all-files)")
    p.add_argument("--all-files", action="store_true",
                   help="Use all available files (overrides --files-per-dataset)")
    p.add_argument("--output", type=Path,
                   default=Path("build/omtf_gmt/eval/new_dataset_validation.md"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n_files = None if args.all_files else args.files_per_dataset

    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output.with_suffix(".json")
    csv_path  = args.output.parent / (args.output.stem.replace("validation", "summary") + ".csv")

    results: list[DatasetResult] = []
    for ds in args.datasets:
        print(f"\n=== {ds} ({DATASET_NAMES.get(ds, '')}) ===")
        r = validate_dataset(ds, args.data_dir, n_files, all_files=args.all_files)
        results.append(r)
        icon = _status_icon(r.overall_status)
        print(f"  → {icon} {r.overall_status}")

    print("\nWriting reports...")
    md = render_markdown(results, n_files)
    args.output.write_text(md)
    print(f"  Markdown: {args.output}")

    jdata = render_json(results)
    json_path.write_text(json.dumps(jdata, indent=2))
    print(f"  JSON:     {json_path}")

    csv_rows = render_csv(results)
    if csv_rows:
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
    print(f"  CSV:      {csv_path}")

    overall = FAIL if any(r.overall_status == FAIL for r in results) else \
              WARN if any(r.overall_status == WARN for r in results) else PASS
    print(f"\nOverall: {_status_icon(overall)} {overall}")
    sys.exit(0 if overall != FAIL else 1)


if __name__ == "__main__":
    main()
