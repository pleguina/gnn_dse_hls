#!/usr/bin/env python
"""
TPS input audit — Step 1 of Phase B4 (OMTF_ALTERNATIVE_STUDY_TPS).

For each dataset, loads (hits, nano) file pairs and reports:
  - Event / window / stub occupancy
  - TPS stub distributions: BX, eta, phi, quality, stubType, tfLayer, depthRegion
  - Fraction of TPS stubs inside the OMTF overlap eta band
  - Truth-transfer quality for a Δphi threshold scan (5/10/20/40 mrad)

The audit must pass (≥1 useful TPS stub/window in G1–G6, sane distributions)
before building the TPS cache (make_gmt_dataset_tps.py).

Usage
-----
  python scripts/omtf_gmt/audit_tps_inputs.py \\
      --datasets G1_pos G1_neg G2_pos G2_neg G7 G8 B4 \\
      --data-dir data/prod \\
      --max-files 5 \\
      --output build/omtf_gmt/eval/TPS_INPUT_AUDIT.md
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from omtf_gmt.regioning import (
    stubs_in_window, phi_rel, omtf_phi_to_global_rad,
    angle_diff, processor_phi_center,
)
from omtf_gmt.features import ETA_OVERLAP_LO, ETA_OVERLAP_HI

try:
    import uproot
    import awkward as ak
except ImportError:
    print("uproot and awkward are required.  pip install uproot awkward")
    sys.exit(1)

# ----- constants -----------------------------------------------------------

ALL_DATASETS = [
    "G1_pos", "G1_neg",
    "G2_pos", "G2_neg",
    "G3_pos", "G3_neg",
    "G4_pos", "G4_neg",
    "G5_pos", "G5_neg",
    "G6_pos", "G6_neg",
    "G7", "G8", "B4",
]
HARD_NEG_DATASETS = {"G7", "G8"}

OMTF_HITS_BRANCHES = [
    "reg_eventNum", "reg_iProcessor",
    "reg_stub_phiHw", "reg_stub_layer", "reg_stub_bx",
    "reg_stub_trackId", "reg_stub_ambiguous",
]
TPS_BRANCHES = [
    "event", "nMuonStubTps",
    "MuonStubTps_offlineCoord1", "MuonStubTps_offlineCoord2",
    "MuonStubTps_offlineEta1",   "MuonStubTps_offlineEta2",
    "MuonStubTps_bxNum",
    "MuonStubTps_quality",       "MuonStubTps_etaQuality",
    "MuonStubTps_tfLayer",       "MuonStubTps_depthRegion",
    "MuonStubTps_stubType",
    "MuonStubTps_isBarrel",      "MuonStubTps_isEndcap",
]

# Phi threshold scan values [rad]
PHI_THRESH_SCAN = [0.005, 0.010, 0.020, 0.040]


# ----- per-file processing --------------------------------------------------

def _load_nano_event_map(nano_path: Path) -> dict[int, dict]:
    """Return dict: uint32(event) → TPS stub arrays."""
    tree = uproot.open(str(nano_path))["Events"]
    available = set(tree.keys())
    branches  = [b for b in TPS_BRANCHES if b in available]
    arr = tree.arrays(branches, library="ak")

    event_map: dict[int, dict] = {}
    for i in range(len(arr)):
        ev = int(arr["event"][i]) & 0xFFFFFFFF
        c1  = ak.to_numpy(arr["MuonStubTps_offlineCoord1"][i]).astype(np.float64)
        c2  = ak.to_numpy(arr["MuonStubTps_offlineCoord2"][i]).astype(np.float32)
        e1  = ak.to_numpy(arr["MuonStubTps_offlineEta1"][i]).astype(np.float32)
        bx  = ak.to_numpy(arr["MuonStubTps_bxNum"][i]).astype(np.int8)
        q   = ak.to_numpy(arr["MuonStubTps_quality"][i]).astype(np.int16)
        eq  = ak.to_numpy(arr["MuonStubTps_etaQuality"][i]).astype(np.int16)
        lay = ak.to_numpy(arr["MuonStubTps_tfLayer"][i]).astype(np.int8)
        dep = ak.to_numpy(arr["MuonStubTps_depthRegion"][i]).astype(np.int8)
        sty = ak.to_numpy(arr["MuonStubTps_stubType"][i]).astype(np.int8)
        ieb = ak.to_numpy(arr["MuonStubTps_isBarrel"][i]).astype(bool) if "MuonStubTps_isBarrel" in available else np.zeros(len(c1), bool)
        iec = ak.to_numpy(arr["MuonStubTps_isEndcap"][i]).astype(bool) if "MuonStubTps_isEndcap" in available else np.zeros(len(c1), bool)
        event_map[ev] = {
            "c1": c1, "c2": c2, "e1": e1, "bx": bx,
            "q": q, "eq": eq, "lay": lay, "dep": dep, "sty": sty,
            "is_barrel": ieb, "is_endcap": iec,
        }
    return event_map


def _truth_transfer_phi_scan(
    tps_phi:    np.ndarray,   # (N,) global phi [rad]
    tps_bx:    np.ndarray,   # (N,) bxNum
    omtf_phi_hw: np.ndarray, # (M,) processor-local phiHw
    omtf_bx:   np.ndarray,   # (M,) reg_stub_bx
    omtf_tid:  np.ndarray,   # (M,) reg_stub_trackId
    proc:      int,
    thresholds: list[float],
) -> dict[float, dict]:
    """Count truth-transfer matches for each phi threshold."""
    if len(omtf_phi_hw) == 0 or len(tps_phi) == 0:
        return {t: {"matched": 0, "signal": 0, "noise": 0, "total": len(tps_phi)}
                for t in thresholds}

    omtf_phi_rad = omtf_phi_to_global_rad(omtf_phi_hw.astype(np.float64), proc)

    results = {}
    for thresh in thresholds:
        matched = signal = noise = 0
        for i in range(len(tps_phi)):
            bx_ok  = (omtf_bx == tps_bx[i])
            dphi   = np.abs(angle_diff(omtf_phi_rad, float(tps_phi[i])))
            phi_ok = dphi < thresh
            cands  = bx_ok & phi_ok
            if cands.any():
                best = int(np.argmin(dphi + np.where(cands, 0.0, 1e9)))
                matched += 1
                if omtf_tid[best] != 0:
                    signal += 1
                else:
                    noise += 1
        results[thresh] = {
            "matched": matched,
            "signal":  signal,
            "noise":   noise,
            "total":   len(tps_phi),
        }
    return results


def audit_file_pair(
    hits_path:  Path,
    nano_path:  Path,
    is_hard_neg: bool = False,
) -> dict:
    """Collect per-window TPS stub statistics for one file pair."""
    nano_map = _load_nano_event_map(nano_path)

    hits_tree = uproot.open(str(hits_path))["simOmtfPhase2Digis/OMTFAllInputTree"]
    arr = hits_tree.arrays(OMTF_HITS_BRANCHES, library="ak")

    stats = {
        "n_events_hits":    0,
        "n_windows":        0,
        "n_windows_with_tps": 0,
        "n_stubs_total":    0,    # all TPS stubs in window
        "n_stubs_barrel":   0,
        "n_stubs_endcap":   0,
        "n_stubs_in_overlap": 0,
        "stubs_per_window": [],   # list of counts (windows with ≥1 TPS stub)
        # distributions (flat lists)
        "bx_vals":          [],
        "eta1_vals":        [],
        "phi_vals":         [],
        "quality_vals":     [],
        "tf_layer_vals":    [],
        "depth_region_vals":[],
        "stub_type_vals":   [],
        # truth-transfer per threshold
        "transfer": {t: {"matched": 0, "signal": 0, "noise": 0, "total": 0}
                     for t in PHI_THRESH_SCAN},
    }

    seen_events: set = set()
    for i in range(len(arr)):
        ev   = int(arr["reg_eventNum"][i]) & 0xFFFFFFFF
        proc = int(arr["reg_iProcessor"][i])

        seen_events.add(ev)
        stats["n_windows"] += 1

        if ev not in nano_map:
            continue

        nano_ev = nano_map[ev]
        c1_all  = nano_ev["c1"]
        if len(c1_all) == 0:
            continue

        # select TPS stubs in this processor's phi window
        mask = stubs_in_window(c1_all, proc)
        if not mask.any():
            continue

        c1  = c1_all[mask]
        e1  = nano_ev["e1"][mask]
        bx  = nano_ev["bx"][mask]
        q   = nano_ev["q"][mask]
        lay = nano_ev["lay"][mask]
        dep = nano_ev["dep"][mask]
        sty = nano_ev["sty"][mask]
        ieb = nano_ev["is_barrel"][mask]
        iec = nano_ev["is_endcap"][mask]

        n = len(c1)
        abs_e = np.abs(e1)
        n_in_ov = int(((abs_e >= ETA_OVERLAP_LO) & (abs_e <= ETA_OVERLAP_HI)).sum())

        stats["n_windows_with_tps"] += 1
        stats["n_stubs_total"]      += n
        stats["n_stubs_barrel"]     += int(ieb.sum())
        stats["n_stubs_endcap"]     += int(iec.sum())
        stats["n_stubs_in_overlap"] += n_in_ov
        stats["stubs_per_window"].append(n)

        # distributions (sample at most 500 stubs per file to bound memory)
        idx_sample = np.arange(n) if n <= 500 else np.random.choice(n, 500, replace=False)
        stats["bx_vals"].extend(bx[idx_sample].tolist())
        stats["eta1_vals"].extend(e1[idx_sample].tolist())
        stats["phi_vals"].extend(c1[idx_sample].tolist())
        stats["quality_vals"].extend(q[idx_sample].tolist())
        stats["tf_layer_vals"].extend(lay[idx_sample].tolist())
        stats["depth_region_vals"].extend(dep[idx_sample].tolist())
        stats["stub_type_vals"].extend(sty[idx_sample].tolist())

        # truth-transfer scan
        phi_hw = ak.to_numpy(arr["reg_stub_phiHw"][i]).astype(np.int32)
        bx_o   = ak.to_numpy(arr["reg_stub_bx"][i]).astype(np.int8)
        tid_o  = ak.to_numpy(arr["reg_stub_trackId"][i]).astype(np.int8)

        tr = _truth_transfer_phi_scan(c1, bx, phi_hw, bx_o, tid_o, proc,
                                      PHI_THRESH_SCAN)
        for thresh in PHI_THRESH_SCAN:
            for key in ("matched", "signal", "noise", "total"):
                stats["transfer"][thresh][key] += tr[thresh][key]

    stats["n_events_hits"] = len(seen_events)
    return stats


def aggregate_stats(file_stats: list[dict]) -> dict:
    if not file_stats:
        return {}
    agg = {
        "n_events_hits":     sum(s["n_events_hits"]     for s in file_stats),
        "n_windows":         sum(s["n_windows"]         for s in file_stats),
        "n_windows_with_tps":sum(s["n_windows_with_tps"]for s in file_stats),
        "n_stubs_total":     sum(s["n_stubs_total"]     for s in file_stats),
        "n_stubs_barrel":    sum(s["n_stubs_barrel"]    for s in file_stats),
        "n_stubs_endcap":    sum(s["n_stubs_endcap"]    for s in file_stats),
        "n_stubs_in_overlap":sum(s["n_stubs_in_overlap"]for s in file_stats),
        "stubs_per_window":  [],
        "bx_vals":           [],
        "eta1_vals":         [],
        "phi_vals":          [],
        "quality_vals":      [],
        "tf_layer_vals":     [],
        "depth_region_vals": [],
        "stub_type_vals":    [],
        "transfer": {t: {"matched": 0, "signal": 0, "noise": 0, "total": 0}
                     for t in PHI_THRESH_SCAN},
    }
    for s in file_stats:
        agg["stubs_per_window"].extend(s["stubs_per_window"])
        agg["bx_vals"].extend(s["bx_vals"])
        agg["eta1_vals"].extend(s["eta1_vals"])
        agg["phi_vals"].extend(s["phi_vals"])
        agg["quality_vals"].extend(s["quality_vals"])
        agg["tf_layer_vals"].extend(s["tf_layer_vals"])
        agg["depth_region_vals"].extend(s["depth_region_vals"])
        agg["stub_type_vals"].extend(s["stub_type_vals"])
        for t in PHI_THRESH_SCAN:
            for k in ("matched", "signal", "noise", "total"):
                agg["transfer"][t][k] += s["transfer"][t][k]
    return agg


# ----- report rendering ----------------------------------------------------

def render_report(results: dict[str, dict]) -> str:
    lines = ["# TPS Input Audit — Phase B4\n",
             "Checks TPS stub occupancy and truth-transfer quality before building the TPS cache.\n"]

    # --- Table 1: occupancy overview ---
    lines.append("## Table 1 — Occupancy overview\n")
    lines.append("| Dataset | Events | Windows | Win w/ TPS | Mean stubs/win | Frac in overlap |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for ds, r in results.items():
        if not r:
            continue
        spw = np.array(r["stubs_per_window"]) if r["stubs_per_window"] else np.array([0])
        frac_win = r["n_windows_with_tps"] / max(1, r["n_windows"])
        mean_spw = float(spw.mean()) if len(spw) > 0 else 0.0
        frac_ov  = r["n_stubs_in_overlap"] / max(1, r["n_stubs_total"])
        lines.append(
            f"| {ds} | {r['n_events_hits']:,} | {r['n_windows']:,} | "
            f"{r['n_windows_with_tps']:,} ({100*frac_win:.0f}%) | "
            f"{mean_spw:.1f} | {100*frac_ov:.1f}% |"
        )

    # --- Table 2: barrel vs endcap split ---
    lines.append("\n## Table 2 — Barrel vs endcap split\n")
    lines.append("| Dataset | Total stubs | Barrel | Endcap | Frac endcap |")
    lines.append("| --- | --- | --- | --- | --- |")
    for ds, r in results.items():
        if not r:
            continue
        tot = r["n_stubs_total"]
        frac_ec = r["n_stubs_endcap"] / max(1, tot)
        lines.append(
            f"| {ds} | {tot:,} | {r['n_stubs_barrel']:,} | "
            f"{r['n_stubs_endcap']:,} | {100*frac_ec:.1f}% |"
        )

    # --- Table 3: truth-transfer threshold scan ---
    lines.append("\n## Table 3 — Truth-transfer threshold scan (Δφ < threshold)\n")
    lines.append(
        "Match rate = fraction of TPS stubs in window matched to any OMTF stub. "
        "Signal% = fraction of matched stubs where OMTF trackId > 0.\n"
    )
    lines.append("| Dataset | Δφ < 5 mrad | Δφ < 10 mrad | Δφ < 20 mrad | Δφ < 40 mrad |")
    lines.append("| --- | --- | --- | --- | --- |")
    for ds, r in results.items():
        if not r:
            continue
        cells = []
        for t in PHI_THRESH_SCAN:
            tr = r["transfer"][t]
            total = max(1, tr["total"])
            match_pct = 100.0 * tr["matched"] / total
            if tr["matched"] > 0:
                sig_pct = 100.0 * tr["signal"] / tr["matched"]
                cells.append(f"{match_pct:.0f}% ({sig_pct:.0f}% sig)")
            else:
                cells.append("0%")
        lines.append(f"| {ds} | " + " | ".join(cells) + " |")

    # --- Distribution tables ---
    lines.append("\n## Table 4 — BX distribution\n")
    lines.append("| Dataset | " + " | ".join(f"BX={v}" for v in range(-3, 4)) + " |")
    lines.append("| --- | " + " | ".join(["---"] * 7) + " |")
    for ds, r in results.items():
        if not r:
            continue
        bx = np.array(r["bx_vals"])
        total = max(1, len(bx))
        cells = [f"{100*np.mean(bx == v):.1f}%" for v in range(-3, 4)]
        lines.append(f"| {ds} | " + " | ".join(cells) + " |")

    lines.append("\n## Table 5 — Quality distribution\n")
    lines.append("| Dataset | q=0 | q=1 | q=2 | q=3 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for ds, r in results.items():
        if not r:
            continue
        q = np.array(r["quality_vals"])
        total = max(1, len(q))
        cells = [f"{100*np.mean(q == v):.1f}%" for v in range(4)]
        lines.append(f"| {ds} | " + " | ".join(cells) + " |")

    lines.append("\n## Table 6 — stubType distribution\n")
    lines.append("| Dataset | type=0 | type=1 |")
    lines.append("| --- | --- | --- |")
    for ds, r in results.items():
        if not r:
            continue
        sty = np.array(r["stub_type_vals"])
        lines.append(
            f"| {ds} | {100*np.mean(sty == 0):.1f}% | {100*np.mean(sty == 1):.1f}% |"
        )

    lines.append("\n## Table 7 — tfLayer distribution\n")
    lines.append("| Dataset | " + " | ".join(f"lay={v}" for v in range(5)) + " |")
    lines.append("| --- | " + " | ".join(["---"] * 5) + " |")
    for ds, r in results.items():
        if not r:
            continue
        lay = np.array(r["tf_layer_vals"])
        cells = [f"{100*np.mean(lay == v):.1f}%" for v in range(5)]
        lines.append(f"| {ds} | " + " | ".join(cells) + " |")

    lines.append("\n## Table 8 — depthRegion distribution\n")
    lines.append("| Dataset | " + " | ".join(f"dep={v}" for v in range(1, 5)) + " |")
    lines.append("| --- | " + " | ".join(["---"] * 4) + " |")
    for ds, r in results.items():
        if not r:
            continue
        dep = np.array(r["depth_region_vals"])
        cells = [f"{100*np.mean(dep == v):.1f}%" for v in range(1, 5)]
        lines.append(f"| {ds} | " + " | ".join(cells) + " |")

    lines.append("\n## Table 9 — |η| distribution percentiles\n")
    lines.append("| Dataset | p10 | p25 | p50 | p75 | p90 | p99 | max |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for ds, r in results.items():
        if not r:
            continue
        e1 = np.abs(np.array(r["eta1_vals"]))
        if len(e1) == 0:
            lines.append(f"| {ds} | — | — | — | — | — | — | — |")
            continue
        pcts = np.percentile(e1, [10, 25, 50, 75, 90, 99])
        lines.append(
            f"| {ds} | {pcts[0]:.3f} | {pcts[1]:.3f} | {pcts[2]:.3f} | "
            f"{pcts[3]:.3f} | {pcts[4]:.3f} | {pcts[5]:.3f} | {e1.max():.3f} |"
        )

    return "\n".join(lines) + "\n"


# ----- main -----------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TPS input audit — Phase B4")
    p.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    p.add_argument("--data-dir", type=Path, default=Path("data/prod"))
    p.add_argument("--max-files", type=int, default=None,
                   help="Max hits files per dataset (None = all)")
    p.add_argument("--output", type=Path,
                   default=Path("build/omtf_gmt/eval/TPS_INPUT_AUDIT.md"))
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    verbose = not args.quiet
    results: dict[str, dict] = {}

    for ds in args.datasets:
        ds_dir     = args.data_dir / ds
        hits_files = sorted(ds_dir.glob("omtf_hits_*.root"))
        if args.max_files:
            hits_files = hits_files[:args.max_files]
        if not hits_files:
            print(f"  [{ds}] no hits files found in {ds_dir}, skipping")
            results[ds] = {}
            continue

        base_ds     = ds.removesuffix("_pos").removesuffix("_neg")
        is_hard_neg = base_ds in HARD_NEG_DATASETS

        file_stats: list[dict] = []
        t0 = time.time()
        for fi, hits_path in enumerate(hits_files):
            nano_path = hits_path.parent / hits_path.name.replace("omtf_hits_", "omtf_nano_")
            if not nano_path.exists():
                if verbose:
                    print(f"  [{ds}] missing nano for {hits_path.name}, skipping")
                continue
            file_stats.append(audit_file_pair(hits_path, nano_path, is_hard_neg))
            if verbose:
                r = file_stats[-1]
                print(
                    f"  [{ds}] {fi+1}/{len(hits_files)}  "
                    f"win_with_tps={r['n_windows_with_tps']}  "
                    f"stubs={r['n_stubs_total']}",
                    end="\r",
                )

        agg = aggregate_stats(file_stats)
        results[ds] = agg
        if verbose and agg:
            spw = np.array(agg["stubs_per_window"]) if agg["stubs_per_window"] else np.array([0])
            print(
                f"\n  [{ds}] done ({time.time()-t0:.0f}s) — "
                f"win_with_tps={agg['n_windows_with_tps']:,}/{agg['n_windows']:,}  "
                f"mean_stubs/win={spw.mean():.2f}  "
                f"frac_in_ov={100*agg['n_stubs_in_overlap']/max(1,agg['n_stubs_total']):.1f}%"
            )

    if not any(results.values()):
        print("No valid data found.")
        return

    report = render_report(results)
    print("\n" + report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"Report written: {args.output}")


if __name__ == "__main__":
    main()
