"""
Stage 1 audit — feature range analysis for raw and derived pair features.

Exports build/audit/feature_ranges.json for quantization range-setting.

Usage:
    python src/audit/feature_ranges.py --all --max-files 5
    python src/audit/feature_ranges.py --dataset S1 --plot
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "prod"
OUTPUT_DIR = PROJECT_ROOT / "build" / "audit"

DATASETS = ["S1", "S2", "S3", "S4", "S5", "B1", "B2", "B3", "B4"]

RAW_FEATURES = ["phi", "phiB", "eta", "r", "quality", "type", "layer", "bx"]
PAIR_FEATURES = ["delta_phi", "delta_r", "delta_r2", "kappa_hat",
                 "abs_delta_eta", "delta_bx", "phiB_diff"]


def compute_pair_features(phi, phiB, eta, r, bx) -> dict:
    n = len(phi)
    out = {k: [] for k in PAIR_FEATURES}
    for i in range(n):
        for j in range(i + 1, n):
            dphi = phi[i] - phi[j]
            dr = r[i] - r[j]
            dr2 = float(dr) ** 2
            kappa = (2.0 * dphi / dr2) if abs(dr2) > 1e-6 else 0.0
            out["delta_phi"].append(dphi)
            out["delta_r"].append(dr)
            out["delta_r2"].append(dr2)
            out["kappa_hat"].append(kappa)
            out["abs_delta_eta"].append(abs(eta[i] - eta[j]))
            out["delta_bx"].append(bx[i] - bx[j])
            out["phiB_diff"].append(phiB[i] - phiB[j])
    return out


def collect_features(path: Path, max_entries: int = 500) -> dict:
    import ROOT
    from audit.root_utils import open_hits_tree, read_entry

    f, t = open_hits_tree(path)
    if t is None:
        return None

    raw = {k: [] for k in RAW_FEATURES}
    pair = {k: [] for k in PAIR_FEATURES}

    n = min(int(t.GetEntries()), max_entries)
    for i in range(n):
        e = read_entry(t, i)
        if e["n_stubs"] == 0:
            continue

        raw["phi"].extend(e["phi"])
        raw["phiB"].extend(e["phiB"])
        raw["eta"].extend(e["eta"])
        raw["r"].extend(e["r"])
        raw["quality"].extend(e["quality"])
        raw["type"].extend(e["type"])
        raw["layer"].extend(e["layer"])
        raw["bx"].extend(e["bx"])

        pf = compute_pair_features(e["phi"], e["phiB"], e["eta"], e["r"], e["bx"])
        for k in PAIR_FEATURES:
            pair[k].extend(pf[k])

    f.Close()
    return {"raw": raw, "pair": pair}


def compute_stats(values: list) -> dict:
    if not values:
        return {}
    arr = np.array(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    outlier_frac = float(np.mean(np.abs(arr - mean) > 3 * std))
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": mean,
        "std": std,
        "outlier_fraction": outlier_frac,
        "n_samples": int(len(arr)),
    }


def analyze_dataset(dataset: str, max_files: int = 5,
                    max_entries: int = 500) -> dict:
    d = DATA_ROOT / dataset
    if not d.exists():
        return None

    files = sorted(d.glob(f"omtf_hits_{dataset}_*.root"))[:max_files]
    if not files:
        return None

    combined_raw = {k: [] for k in RAW_FEATURES}
    combined_pair = {k: [] for k in PAIR_FEATURES}

    for path in files:
        r = collect_features(path, max_entries=max_entries)
        if r is None:
            continue
        for k in RAW_FEATURES:
            combined_raw[k].extend(r["raw"][k])
        for k in PAIR_FEATURES:
            combined_pair[k].extend(r["pair"][k])

    return {
        "dataset": dataset,
        "raw": {k: compute_stats(combined_raw[k]) for k in RAW_FEATURES},
        "pair": {k: compute_stats(combined_pair[k]) for k in PAIR_FEATURES},
    }


def print_table(stats: dict, label: str):
    print(f"\n  {label}:")
    print(f"    {'Feature':20s}  {'Min':>9s}  {'Max':>9s}  "
          f"{'Mean':>9s}  {'Std':>9s}  {'Outlier%':>8s}")
    print("    " + "-" * 68)
    for name, s in stats.items():
        if not s:
            continue
        print(f"    {name:20s}  {s['min']:>9.2f}  {s['max']:>9.2f}  "
              f"{s['mean']:>9.2f}  {s['std']:>9.2f}  "
              f"{100*s['outlier_fraction']:>7.2f}%")


def main():
    parser = argparse.ArgumentParser(description="OMTF feature range analysis")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max-files", type=int, default=5)
    parser.add_argument("--max-entries", type=int, default=500)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    import ROOT
    ROOT.gROOT.SetBatch(True)
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    datasets = DATASETS if args.all else ([args.dataset] if args.dataset else ["S1"])

    all_stats = {}
    for ds in datasets:
        print(f"\n=== {ds} ===")
        stats = analyze_dataset(ds, max_files=args.max_files, max_entries=args.max_entries)
        if stats is None:
            print("  [SKIP]")
            continue
        print_table(stats["raw"], "Raw stub features")
        print_table(stats["pair"], "Derived pair features")
        all_stats[ds] = stats

    if all_stats:
        out = OUTPUT_DIR / "feature_ranges.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        # Remove numpy arrays before serialising
        serialisable = {}
        for ds, s in all_stats.items():
            serialisable[ds] = {
                "raw": s["raw"],
                "pair": s["pair"],
            }
        with open(out, "w") as f:
            json.dump(serialisable, f, indent=2)
        print(f"\nSaved: {out}")
        print("feature_ranges.json ready for quantization range-setting.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
