"""
Stage 1 audit — occupancy analysis per dataset.

Measures per processor-window:
  - Stubs per window: mean, p95, p99, max
  - Per-type and per-layer stub distributions
  - BX distribution
  - Recommended Nmax

Output: build/audit/occupancy_summary.csv, optional plots

Usage:
    python src/audit/occupancy.py --all
    python src/audit/occupancy.py --dataset B1 --max-files 10
    python src/audit/occupancy.py --all --max-files 5 --plot
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "prod"
OUTPUT_DIR = PROJECT_ROOT / "build" / "audit"

DATASETS = ["S1", "S2", "S3", "S4", "S5", "B1", "B2", "B3", "B4"]


def collect_occupancy(path: Path, max_entries: int = 1000) -> dict:
    import ROOT
    from audit.root_utils import open_hits_tree, read_entry

    f, t = open_hits_tree(path)
    if t is None:
        return None

    stubs_per_window = []
    bx_counts = defaultdict(int)
    type_counts = defaultdict(int)
    layer_counts = defaultdict(int)

    n = min(int(t.GetEntries()), max_entries)
    for i in range(n):
        e = read_entry(t, i)
        ns = e["n_stubs"]
        stubs_per_window.append(ns)
        for bx in e["bx"]:
            bx_counts[bx] += 1
        for typ in e["type"]:
            type_counts[typ] += 1
        for lay in e["layer"]:
            layer_counts[lay] += 1

    f.Close()
    return {
        "stubs_per_window": stubs_per_window,
        "bx_counts": dict(bx_counts),
        "type_counts": dict(type_counts),
        "layer_counts": dict(layer_counts),
    }


def aggregate_dataset(dataset: str, max_files: int = 10,
                      max_entries: int = 1000) -> dict:
    d = DATA_ROOT / dataset
    if not d.exists():
        return None

    files = sorted(d.glob(f"omtf_hits_{dataset}_*.root"))[:max_files]
    if not files:
        return None

    all_stubs = []
    bx_total = defaultdict(int)
    type_total = defaultdict(int)
    layer_total = defaultdict(int)

    for path in files:
        r = collect_occupancy(path, max_entries=max_entries)
        if r is None:
            continue
        all_stubs.extend(r["stubs_per_window"])
        for k, v in r["bx_counts"].items():
            bx_total[k] += v
        for k, v in r["type_counts"].items():
            type_total[k] += v
        for k, v in r["layer_counts"].items():
            layer_total[k] += v

    if not all_stubs:
        return None

    arr = np.array(all_stubs)
    return {
        "dataset": dataset,
        "n_windows": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": int(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(np.max(arr)),
        "bx_counts": dict(sorted(bx_total.items())),
        "type_counts": dict(sorted(type_total.items())),
        "layer_counts": dict(sorted(layer_total.items())),
        "raw": arr,
    }


def recommend_nmax(summaries: list) -> int:
    p99_values = [s["p99"] for s in summaries if s is not None]
    if not p99_values:
        return 64
    max_p99 = max(p99_values)
    nmax = int(np.ceil(max_p99 / 8)) * 8
    return max(nmax, 8)


def save_csv(summaries: list, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "n_windows", "mean", "std", "min", "p50", "p95", "p99", "max"]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in summaries:
            if s is None:
                continue
            w.writerow({k: s[k] for k in fields})
    print(f"\nSaved: {output_path}")


def plot_occupancy(summaries: list, output_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    for s in summaries:
        if s is None:
            continue
        arr = s["raw"]
        max_val = int(s["max"])
        bins = range(0, max_val + 2)
        ax.hist(arr, bins=bins, alpha=0.6, label=s["dataset"],
                density=True, histtype="step", linewidth=1.5)

    ax.set_xlabel("Stubs per processor window")
    ax.set_ylabel("Density")
    ax.set_title("Stub occupancy per processor window")
    ax.legend(ncol=3, fontsize=8)
    ax.set_xlim(0, None)

    out = output_dir / "occupancy_distributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="OMTF occupancy analysis")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max-files", type=int, default=10)
    parser.add_argument("--max-entries", type=int, default=1000)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    import ROOT
    ROOT.gROOT.SetBatch(True)
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    datasets = DATASETS if args.all else ([args.dataset] if args.dataset else ["B1"])

    summaries = []
    print(f"\n{'Dataset':8s}  {'N windows':>10s}  {'Mean':>6s}  {'p50':>6s}  "
          f"{'p95':>6s}  {'p99':>6s}  {'Max':>6s}")
    print("-" * 62)

    for ds in datasets:
        s = aggregate_dataset(ds, max_files=args.max_files, max_entries=args.max_entries)
        if s is None:
            print(f"{ds:8s}  [SKIP]")
            summaries.append(None)
            continue
        print(f"{ds:8s}  {s['n_windows']:>10d}  {s['mean']:>6.1f}  {s['p50']:>6.1f}  "
              f"{s['p95']:>6.1f}  {s['p99']:>6.1f}  {s['max']:>6d}")
        summaries.append(s)

    valid = [s for s in summaries if s is not None]
    if valid:
        nmax = recommend_nmax(valid)
        print(f"\nRecommended Nmax: {nmax}  "
              f"(max p99 rounded up to nearest multiple of 8)")
        print("Update configs/model_config.yaml omtf.Nmax and DATASET_SIGNOFF.md")

        save_csv(valid, OUTPUT_DIR / "occupancy_summary.csv")

        if args.plot:
            plot_occupancy(valid, OUTPUT_DIR / "plots")

    return 0


if __name__ == "__main__":
    sys.exit(main())
