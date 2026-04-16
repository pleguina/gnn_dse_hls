"""
Stage 1.5 audit — physical variable validation across all datasets.

Goes beyond range statistics to check physical consistency:

  A. Per-variable value inventories
     - phi, phiB, eta, r, quality, type, layer, bx unique value sets
     - Flags any value outside the expected OMTF hardware envelope

  B. Layer-r consistency
     - r should be FIXED per layer (detector geometry is rigid)
     - Reports mean r per layer; flags layers with r spread > tolerance

  C. Type composition and quality-per-type
     - DT/CSC/RPC stub fractions per dataset
     - Quality distribution per stub type
     - phiB == 0 check for RPC stubs (no bending measurement)

  D. iProcessor distribution
     - Should be roughly uniform across sectors 0–11

  E. GenMuon pT from NanoAOD
     - Joins hits to nano, reads GenMuon_pt for signal stubs
     - Reports pT distribution per dataset (needed for regression target)
     - Also reads GenMuon_charge, GenMuon_eta, GenMuon_phi

  F. Signal stub pT coverage
     - For each signal stub: what pT does the matched muon have?
     - Displacement (d0) if available

Output: build/audit/variable_check_report.md + build/audit/gen_pt_distributions.json

Usage:
    python src/audit/check_variables.py --all --max-files 3 --max-entries 300
    python src/audit/check_variables.py --dataset S1 --max-files 5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "prod"
OUTPUT_DIR = PROJECT_ROOT / "build" / "audit"

DATASETS = ["S1", "S2", "S3", "S4", "S5", "B1", "B2", "B3", "B4"]

# --------------------------------------------------------------------------
# OMTF hardware envelope (inclusive).
# Values outside these ranges are reported as unexpected — not errors, but
# worth flagging before assuming a variable is ready for model input.
# --------------------------------------------------------------------------
EXPECTED_BOUNDS = {
    # Validated against all 9 datasets — see docs/omtf/VARIABLE_SIGNOFF.md
    "phi":     (-500,  2000),   # processor-window phi in HW units; seen [-463, 1854]
    "phiB":    (-1100, 1100),   # bending; DT/CSC non-zero, RPC always 0; ~1% tails beyond ±512
    "eta":     (50,    130),    # HW eta code (positive-eta region; seen [58,127])
    "r":       (250,   700),    # radial HW code; seen [281,680]; constant per ring
    "quality": (0,     15),     # up to 4-bit quality; seen [1,6]
    "type":    (0,     15),     # DT=3, RPC=5, CSC=9 in Phase-2 OMTF encoding
    "layer":   (0,     17),     # 18 OMTF layers (0-indexed); OK
    "bx":      (-3,    3),      # bunch crossing; all 0 in current datasets
}

# Tolerance: a layer's r values should not vary by more than this.
# Single-ring layers (RPC barrel) have std=0. Multi-ring layers (CSC, RPC endcap)
# legitimately show spread of 30-110 units. Flag only catastrophic variance.
R_SPREAD_TOLERANCE = 120


# --------------------------------------------------------------------------
# Section A+B+C+D: per-file variable scan
# --------------------------------------------------------------------------

def scan_file(path: Path, max_entries: int) -> dict | None:
    import ROOT
    from audit.root_utils import open_hits_tree, read_entry

    f, t = open_hits_tree(path)
    if t is None:
        return None

    # Collectors
    raw_vals   = defaultdict(list)   # feature -> all values
    layer_r    = defaultdict(list)   # layer -> list of r values
    type_qual  = defaultdict(list)   # type -> list of quality values
    type_phiB  = defaultdict(list)   # type -> list of phiB values
    iproc_counts = defaultdict(int)

    n = min(int(t.GetEntries()), max_entries)
    for i in range(n):
        e = read_entry(t, i)
        ns = e["n_stubs"]
        if ns == 0:
            continue

        iproc_counts[e["i_processor"]] += 1

        for feat in ("phi", "phiB", "eta", "r", "quality", "type", "layer", "bx"):
            raw_vals[feat].extend(e[feat])

        for j in range(ns):
            lay = e["layer"][j]
            r_j = e["r"][j]
            t_j = e["type"][j]
            q_j = e["quality"][j]
            p_j = e["phiB"][j]

            layer_r[lay].append(r_j)
            type_qual[t_j].append(q_j)
            type_phiB[t_j].append(p_j)

    f.Close()
    return {
        "n_entries": n,
        "raw_vals":  {k: v for k, v in raw_vals.items()},
        "layer_r":   {k: v for k, v in layer_r.items()},
        "type_qual": {k: v for k, v in type_qual.items()},
        "type_phiB": {k: v for k, v in type_phiB.items()},
        "iproc_counts": dict(iproc_counts),
    }


def merge_scans(scans: list[dict]) -> dict:
    out = {
        "n_entries": 0,
        "raw_vals":  defaultdict(list),
        "layer_r":   defaultdict(list),
        "type_qual": defaultdict(list),
        "type_phiB": defaultdict(list),
        "iproc_counts": defaultdict(int),
    }
    for s in scans:
        if s is None:
            continue
        out["n_entries"] += s["n_entries"]
        for k, v in s["raw_vals"].items():
            out["raw_vals"][k].extend(v)
        for k, v in s["layer_r"].items():
            out["layer_r"][k].extend(v)
        for k, v in s["type_qual"].items():
            out["type_qual"][k].extend(v)
        for k, v in s["type_phiB"].items():
            out["type_phiB"][k].extend(v)
        for k, v in s["iproc_counts"].items():
            out["iproc_counts"][k] += v
    return out


# --------------------------------------------------------------------------
# Section E: GenMuon pT via NanoAOD join
# --------------------------------------------------------------------------

def scan_gen_muon(hits_path: Path, nano_path: Path, max_entries: int) -> dict | None:
    import ROOT
    from audit.root_utils import open_hits_tree, open_nano_tree, read_entry

    fh, th = open_hits_tree(hits_path)
    if th is None:
        return None
    fn, tn = open_nano_tree(nano_path)
    if tn is None:
        fh.Close()
        return None

    # Build nano event map
    nano_map = {}
    for i in range(tn.GetEntries()):
        tn.GetEntry(i)
        key = int(tn.event) & 0xFFFFFFFF
        nano_map[key] = i

    pt_values   = []
    charge_vals = []
    eta_vals    = []
    d0_vals     = []
    has_d0      = None   # probe once
    has_charge  = None

    n = min(int(th.GetEntries()), max_entries)
    for i in range(n):
        e = read_entry(th, i)
        key = e["event_num"]
        if key not in nano_map:
            continue
        nano_idx = nano_map[key]
        tn.GetEntry(nano_idx)
        n_gen = int(tn.nGenMuon)

        # Probe for optional branches once
        # NanoAOD uses GenMuon_dXY (not d0)
        if has_d0 is None:
            try:
                _ = tn.GenMuon_dXY
                has_d0 = True
            except Exception:
                has_d0 = False
        if has_charge is None:
            try:
                _ = tn.GenMuon_charge
                has_charge = True
            except Exception:
                has_charge = False

        for tid in e["track_id"]:
            if tid == 0:
                continue
            gen_idx = tid - 1
            if gen_idx < 0 or gen_idx >= n_gen:
                continue
            pt_values.append(float(tn.GenMuon_pt[gen_idx]))
            if has_charge:
                charge_vals.append(int(tn.GenMuon_charge[gen_idx]))
            if has_d0:
                try:
                    d0_vals.append(float(tn.GenMuon_dXY[gen_idx]))
                except Exception:
                    pass
            try:
                eta_vals.append(float(tn.GenMuon_eta[gen_idx]))
            except Exception:
                pass

    fh.Close()
    fn.Close()

    if not pt_values:
        return {"pt": [], "charge": [], "eta": [], "d0": [], "has_d0": False, "has_charge": False}

    return {
        "pt": pt_values,
        "charge": charge_vals,
        "eta": eta_vals,
        "d0": d0_vals,
        "has_d0": has_d0,
        "has_charge": has_charge,
    }


def pt_stats(values: list) -> dict:
    if not values:
        return {}
    arr = np.array(values)
    return {
        "n": len(arr),
        "min": float(arr.min()),
        "p5":  float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


# --------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------

def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def report_bounds(merged: dict, dataset: str) -> list[str]:
    issues = []
    for feat, vals in merged["raw_vals"].items():
        if not vals:
            continue
        lo, hi = EXPECTED_BOUNDS.get(feat, (-999999, 999999))
        arr = np.array(vals)
        n_oob = int(np.sum((arr < lo) | (arr > hi)))
        pct = 100.0 * n_oob / len(arr)
        flag = " <-- OUT-OF-BOUNDS" if n_oob > 0 else ""
        unique_count = len(np.unique(arr))
        print(f"    {feat:8s}: min={int(arr.min()):6d}  max={int(arr.max()):6d}  "
              f"unique={unique_count:4d}  OOB={n_oob} ({pct:.1f}%){flag}")
        if n_oob > 0:
            oob_vals = arr[(arr < lo) | (arr > hi)]
            issues.append(f"{dataset}.{feat}: {n_oob} values outside [{lo},{hi}] "
                          f"(e.g. {sorted(set(oob_vals.tolist()))[:5]})")
    return issues


def report_layer_r(merged: dict, dataset: str) -> list[str]:
    issues = []
    print(f"\n  Layer-r consistency (r should be fixed per layer):")
    print(f"    {'Layer':6s}  {'N stubs':>8s}  {'r mean':>8s}  "
          f"{'r std':>7s}  {'r unique':>10s}  Status")
    print("    " + "-" * 58)

    for lay in sorted(merged["layer_r"].keys()):
        r_arr = np.array(merged["layer_r"][lay])
        r_unique = sorted(set(r_arr.tolist()))
        spread = float(r_arr.std())
        flag = ""
        if spread > R_SPREAD_TOLERANCE:
            flag = " <-- SPREAD"
            issues.append(f"{dataset} layer {lay}: r std={spread:.1f} > tolerance {R_SPREAD_TOLERANCE}")
        print(f"    {lay:6d}  {len(r_arr):>8d}  {r_arr.mean():>8.1f}  "
              f"{spread:>7.2f}  {str(r_unique[:6]):>10s}{flag}")
    return issues


def report_type_composition(merged: dict, dataset: str):
    total = sum(len(v) for v in merged["type_qual"].values())
    if total == 0:
        return
    print(f"\n  Stub type composition  (total stubs: {total}):")
    print(f"    {'Type':5s}  {'Count':>8s}  {'Frac':>7s}  "
          f"{'Quality vals (unique)':25s}  {'phiB mean':>10s}  {'phiB==0 %':>10s}")
    print("    " + "-" * 80)
    for typ in sorted(merged["type_qual"].keys()):
        qual_arr = np.array(merged["type_qual"][typ])
        phiB_arr = np.array(merged["type_phiB"][typ])
        n = len(qual_arr)
        frac = 100.0 * n / total
        q_unique = sorted(set(qual_arr.tolist()))
        phiB_zero_pct = 100.0 * np.sum(phiB_arr == 0) / n
        phiB_mean = float(phiB_arr.mean())
        print(f"    {typ:5d}  {n:>8d}  {frac:>6.1f}%  "
              f"{str(q_unique):25s}  {phiB_mean:>10.1f}  {phiB_zero_pct:>9.1f}%")


def report_iprocessor(merged: dict):
    counts = merged["iproc_counts"]
    if not counts:
        return
    total = sum(counts.values())
    print(f"\n  iProcessor distribution  (windows, not stubs):")
    sectors = sorted(counts.keys())
    for sec in sectors:
        bar = "#" * int(30 * counts[sec] / max(counts.values()))
        print(f"    proc {sec:2d}: {counts[sec]:5d}  ({100*counts[sec]/total:.1f}%)  {bar}")


def report_gen_pt(pt_data: dict | None, dataset: str):
    if pt_data is None or not pt_data["pt"]:
        print(f"  No GenMuon pT data (no signal stubs or join failed)")
        return

    arr = np.array(pt_data["pt"])
    s = pt_stats(pt_data["pt"])
    print(f"  GenMuon pT ({len(arr)} signal-stub matched muons):")
    print(f"    min={s['min']:.1f}  p5={s['p5']:.1f}  p25={s['p25']:.1f}  "
          f"p50={s['p50']:.1f}  p75={s['p75']:.1f}  p95={s['p95']:.1f}  max={s['max']:.1f} GeV")

    if pt_data.get("has_charge") and pt_data["charge"]:
        charges = pt_data["charge"]
        pos = sum(1 for c in charges if c > 0)
        neg = len(charges) - pos
        print(f"    charge: +1={pos}  -1={neg}  (ratio {pos/max(neg,1):.2f})")

    if pt_data.get("has_d0") and pt_data["d0"]:
        d0_arr = np.array(pt_data["d0"])
        print(f"    dXY: min={d0_arr.min():.3f}  median={np.median(d0_arr):.3f}  "
              f"p95={np.percentile(d0_arr,95):.3f}  max={d0_arr.max():.3f}  "
              f"|dXY|>1cm={np.sum(np.abs(d0_arr)>1)} ({100*np.mean(np.abs(d0_arr)>1):.1f}%)")
    else:
        print(f"    GenMuon_dXY: not available in these files")

    return s


# --------------------------------------------------------------------------
# Per-dataset runner
# --------------------------------------------------------------------------

def check_dataset(dataset: str, max_files: int, max_entries: int) -> dict:
    d = DATA_ROOT / dataset
    if not d.exists():
        print(f"  [SKIP] {d} not found")
        return {}

    hits_files = sorted(d.glob(f"omtf_hits_{dataset}_*.root"))[:max_files]
    if not hits_files:
        print(f"  [SKIP] No files")
        return {}

    # Scan raw variables
    scans = []
    for hf in hits_files:
        scans.append(scan_file(hf, max_entries))
    merged = merge_scans([s for s in scans if s is not None])

    all_issues = []

    print(f"\n--- A. Variable bounds ---")
    issues = report_bounds(merged, dataset)
    all_issues.extend(issues)

    print(f"\n--- B. Layer-r consistency ---")
    issues = report_layer_r(merged, dataset)
    all_issues.extend(issues)

    print(f"\n--- C. Type composition and phiB ---")
    report_type_composition(merged, dataset)

    print(f"\n--- D. iProcessor distribution ---")
    report_iprocessor(merged)

    # GenMuon pT: use first file pair only
    pt_data = None
    pt_stats_out = None
    if dataset != "B4":
        hf = hits_files[0]
        nf = hf.parent / hf.name.replace("hits", "nano")
        if not nf.exists():
            # Try alternative naming
            idx = hf.stem.split("_")[-1]
            nf = hf.parent / f"omtf_nano_{dataset}_{idx}.root"
        if nf.exists():
            print(f"\n--- E. GenMuon pT (NanoAOD join) ---")
            pt_data = scan_gen_muon(hf, nf, max_entries)
            pt_stats_out = report_gen_pt(pt_data, dataset)
        else:
            print(f"\n--- E. GenMuon pT ---  [SKIP] nano file not found at {nf}")
    else:
        print(f"\n--- E. GenMuon pT ---  [SKIP] B4 is noise-only")

    if all_issues:
        print(f"\n  ISSUES ({len(all_issues)}):")
        for iss in all_issues:
            print(f"    ! {iss}")
    else:
        print(f"\n  No issues found.")

    # Build summary for JSON export
    summary = {
        "dataset": dataset,
        "n_entries": merged["n_entries"],
        "variable_bounds": {},
        "layer_r_spread": {},
        "type_fractions": {},
        "issues": all_issues,
    }
    for feat, vals in merged["raw_vals"].items():
        if vals:
            arr = np.array(vals)
            lo, hi = EXPECTED_BOUNDS.get(feat, (-999999, 999999))
            summary["variable_bounds"][feat] = {
                "min": int(arr.min()), "max": int(arr.max()),
                "n_unique": int(len(np.unique(arr))),
                "n_oob": int(np.sum((arr < lo) | (arr > hi))),
                "expected_lo": lo, "expected_hi": hi,
            }
    for lay, r_list in merged["layer_r"].items():
        arr = np.array(r_list)
        summary["layer_r_spread"][str(lay)] = {
            "mean_r": float(arr.mean()),
            "std_r":  float(arr.std()),
            "unique_r": sorted(set(arr.tolist())),
        }
    total_stubs = sum(len(v) for v in merged["type_qual"].values())
    for typ, quals in merged["type_qual"].items():
        summary["type_fractions"][str(typ)] = {
            "n": len(quals),
            "fraction": len(quals) / max(total_stubs, 1),
            "quality_unique": sorted(set(quals)),
        }
    if pt_stats_out:
        summary["gen_pt"] = pt_stats_out
        summary["has_d0"]     = pt_data.get("has_d0", False) if pt_data else False
        summary["has_charge"] = pt_data.get("has_charge", False) if pt_data else False

    return summary


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OMTF physical variable validation")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max-files", type=int, default=3)
    parser.add_argument("--max-entries", type=int, default=300)
    args = parser.parse_args()

    import ROOT
    ROOT.gROOT.SetBatch(True)
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    datasets = DATASETS if args.all else ([args.dataset] if args.dataset else ["S1"])

    all_summaries = {}
    for ds in datasets:
        print_section(f"Dataset: {ds}")
        summary = check_dataset(ds, args.max_files, args.max_entries)
        if summary:
            all_summaries[ds] = summary

    # Save JSON
    if all_summaries:
        out_json = OUTPUT_DIR / "variable_check.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"\nSaved: {out_json}")

    # Global summary of issues
    total_issues = sum(len(s.get("issues", [])) for s in all_summaries.values())
    print(f"\n{'='*60}")
    print(f"  GLOBAL SUMMARY: {len(all_summaries)} datasets, {total_issues} issues")
    if total_issues:
        for ds, s in all_summaries.items():
            for iss in s.get("issues", []):
                print(f"    [{ds}] {iss}")
    else:
        print("  All variables within expected bounds.")
    print('='*60)

    return 0 if total_issues == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
